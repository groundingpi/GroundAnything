#!/usr/bin/env python3
"""OpenAI-compatible, single-GPU GAM-Qwen3.5 DLM evaluation backend."""

from __future__ import annotations

from infer.request_contract import validate_thinking, validate_budget

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration


GAM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GAM_ROOT))

from models.dlm.hybrid import GAMQwen35DLM, MASK_TOKEN  # noqa: E402
from models.qwen35_runtime import install_qwen35_fastpath  # noqa: E402
from infer.dlm.stop_contract import single_token_stop_ids  # noqa: E402


BASE_MODEL = Path(os.environ["BASE_MODEL"]).resolve()
DLM_CHECKPOINT = Path(os.environ["DLM_CHECKPOINT"]).resolve()
PORT = int(os.environ.get("PORT", "8101"))
MODEL_NAME = os.environ.get("MODEL_VERSION", str(DLM_CHECKPOINT))
INFERENCE_BLOCK_SIZE = int(os.environ.get("DLM_INFERENCE_BLOCK_SIZE", "16"))
MAX_MODEL_LEN = int(os.environ.get("DLM_MAX_MODEL_LEN", "16384"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("DLM_MAX_NEW_TOKENS_CAP", "16000"))
VISION_ATTN = os.environ.get("DLM_VISION_ATTN_IMPLEMENTATION", "flash_attention_2")
TEXT_ATTN = os.environ.get("DLM_TEXT_ATTN_IMPLEMENTATION", "eager")
CACHE_CLEANUP_INTERVAL_TOKENS = int(os.environ.get("DLM_CUDA_CACHE_CLEANUP_INTERVAL_TOKENS", "128"))
CACHE_CLEANUP_FRACTION = float(os.environ.get("DLM_CUDA_CACHE_CLEANUP_FRACTION", "0.25"))
VALIDATION_ORACLE_ENABLED = os.environ.get("DLM_ENABLE_VALIDATION_ORACLE", "0") == "1"
DEFAULT_SKIP_SPECIAL_TOKENS = (
    os.environ.get("DLM_DEFAULT_SKIP_SPECIAL_TOKENS", "0") == "1"
)
DEFAULT_SPACES_BETWEEN_SPECIAL_TOKENS = (
    os.environ.get("DLM_DEFAULT_SPACES_BETWEEN_SPECIAL_TOKENS", "0") == "1"
)

_DEVICE = torch.device("cuda:0")
_LOCK = threading.Lock()
_MODEL: GAMQwen35DLM | None = None
_PROCESSOR: Any = None
_TOKENIZER: Any = None
_REQUESTS = 0
_FAILURES = 0
_TOTAL_SECONDS = 0.0
_TOTAL_OUTPUT_TOKENS = 0
_TOTAL_NFE = 0
_STOP_REQUESTS = 0
_STOP_TERMINATIONS = 0
_LAST_REQUEST_STOP: list[str] = []
_LAST_REQUEST_STOP_TOKEN_IDS: list[int] = []
_LAST_GENERATION_PARAMETERS: dict[str, Any] = {}
_LOGGED_STOP_CONTRACTS: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _log_stop_contract(
    stop: tuple[str, ...], stop_token_ids: tuple[int, ...]
) -> None:
    contract = (stop, stop_token_ids)
    if contract in _LOGGED_STOP_CONTRACTS:
        return
    _LOGGED_STOP_CONTRACTS.add(contract)
    print(
        json.dumps(
            {
                "event": "openai_stop_contract",
                "stop": list(stop),
                "stop_token_ids": list(stop_token_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def load_model() -> None:
    global _MODEL, _PROCESSOR, _TOKENIZER
    _require((BASE_MODEL / "config.json").is_file(), f"base model is incomplete: {BASE_MODEL}")
    weights = DLM_CHECKPOINT / "model.safetensors"
    _require(weights.is_file(), f"DLM weights are missing: {weights}")

    fastpath = install_qwen35_fastpath()
    print(json.dumps({"qwen35_fastpath": fastpath}, sort_keys=True), flush=True)
    _PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=False)
    _TOKENIZER = AutoTokenizer.from_pretrained(DLM_CHECKPOINT, trust_remote_code=False)
    _PROCESSOR.tokenizer = _TOKENIZER
    native = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        # The PPU FA2 text kernel for Qwen3.5 head_dim=256 has an illegal-page
        # failure on causal verification.  Keep FA2 on the ViT, while the eight
        # text full-attention layers use the mathematically equivalent eager
        # implementation.  Linear-attention layers still use FLA Triton.
        attn_implementation={
            "vision_config": VISION_ATTN,
            "text_config": TEXT_ATTN,
        },
        trust_remote_code=False,
    )
    if native.get_input_embeddings().weight.shape[0] != len(_TOKENIZER):
        native.resize_token_embeddings(len(_TOKENIZER), mean_resizing=False)
    mask_id = int(_TOKENIZER.convert_tokens_to_ids(MASK_TOKEN))
    _require(mask_id >= 0 and _TOKENIZER.encode(MASK_TOKEN, add_special_tokens=False) == [mask_id],
             "final checkpoint tokenizer has no atomic DLM mask token")
    im_end_id = int(_TOKENIZER.convert_tokens_to_ids("<|im_end|>"))
    _MODEL = GAMQwen35DLM(native, mask_id, im_end_id, block_size=32).eval()

    state = load_file(str(weights), device="cpu")
    _require(state and all(key.startswith("base_model.") for key in state),
             "checkpoint is not a GAMQwen35DLM wrapper state dict")
    incompatible = _MODEL.load_state_dict(state, strict=False)
    _require(not incompatible.missing_keys, f"missing DLM keys: {incompatible.missing_keys[:10]}")
    _require(not incompatible.unexpected_keys, f"unexpected DLM keys: {incompatible.unexpected_keys[:10]}")
    del state
    _MODEL.to(_DEVICE)
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "READY",
                "base_model": str(BASE_MODEL),
                "checkpoint": str(DLM_CHECKPOINT),
                "model": MODEL_NAME,
                "device": str(_DEVICE),
                "dtype": "bfloat16",
                "inference_block_size": INFERENCE_BLOCK_SIZE,
                "vision_attention": VISION_ATTN,
                "text_attention": TEXT_ATTN,
                "vocab_size": len(_TOKENIZER),
                "cuda_cache_cleanup_interval_tokens": CACHE_CLEANUP_INTERVAL_TOKENS,
                "cuda_cache_cleanup_fraction": CACHE_CLEANUP_FRACTION,
                "validation_oracle_enabled": VALIDATION_ORACLE_ENABLED,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _decode_image_url(value: Any) -> Image.Image:
    url = value.get("url") if isinstance(value, dict) else value
    _require(isinstance(url, str), "image_url.url must be a string")
    payload = url.split(",", 1)[1] if url.startswith("data:") else url
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": message.get("role", "user"), "content": content})
            continue
        items: list[dict[str, Any]] = []
        for item in content:
            kind = item.get("type")
            if kind == "text":
                items.append({"type": "text", "text": item.get("text", "")})
            elif kind == "image_url":
                items.append({"type": "image", "image": _decode_image_url(item.get("image_url"))})
            else:
                raise ValueError(f"unsupported OpenAI content type: {kind!r}")
        normalized.append({"role": message.get("role", "user"), "content": items})
    return normalized


def _prepare_inputs(messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    normalized = _normalize_messages(messages)
    inputs = _PROCESSOR.apply_chat_template(
        normalized,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    prepared: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            value = value.to(dtype=torch.bfloat16)
        prepared[key] = value.to(_DEVICE)
    return prepared


def run_inference(
    messages: list[dict[str, Any]],
    requested_max_tokens: int,
    repetition_penalty: float = 1.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = False,
    decoding: str = "speculative",
    stop: object = None,
    no_stop_trim: bool = False,
) -> tuple[str, dict[str, Any]]:
    _require(_MODEL is not None, "model is not loaded")
    inputs = _prepare_inputs(messages)
    max_new_tokens = max(1, min(int(requested_max_tokens), MAX_NEW_TOKENS_CAP))
    normalized_request_stop, request_stop_ids = single_token_stop_ids(
        _TOKENIZER, stop
    )
    _log_stop_contract(normalized_request_stop, request_stop_ids)
    stop_ids = tuple(
        dict.fromkeys(
            token
            for token in (
                int(_TOKENIZER.eos_token_id) if _TOKENIZER.eos_token_id is not None else -1,
                int(_TOKENIZER.convert_tokens_to_ids("<|im_end|>")),
                *request_stop_ids,
            )
            if token >= 0
        )
    )
    generation_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
        "max_new_tokens": max_new_tokens,
        "stop_token_ids": stop_ids,
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
        "repetition_penalty": repetition_penalty,
    }
    if temperature != 0.0 or top_p != 1.0 or top_k is not None:
        raise ValueError(
            "legacy speculative DLM backend supports greedy decoding only; "
            "use infer/dlm/openai_server.py for sampling"
        )
    if decoding == "speculative":
        response_ids = _MODEL.speculative_generate(
            **generation_args,
            inference_block_size=INFERENCE_BLOCK_SIZE,
            cuda_cache_cleanup_interval_tokens=CACHE_CLEANUP_INTERVAL_TOKENS,
            cuda_cache_cleanup_fraction=CACHE_CLEANUP_FRACTION,
        )
    elif decoding == "causal_full_prefix_oracle" and VALIDATION_ORACLE_ENABLED:
        response_ids = _MODEL.causal_generate_full_prefix(**generation_args)
    else:
        raise ValueError(f"unsupported DLM decoding mode: {decoding!r}")
    ids = response_ids[0].tolist()
    terminated_by_request_stop = bool(ids and ids[-1] in request_stop_ids)
    finish_reason = (
        "stop"
        if ids and ids[-1] in stop_ids
        else "length"
        if len(ids) >= max_new_tokens
        else "stop"
    )
    while ids and ids[-1] in stop_ids:
        if no_stop_trim and ids[-1] in request_stop_ids:
            break
        ids.pop()
    text = _TOKENIZER.decode(
        ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=spaces_between_special_tokens,
    ).strip()
    stats = dict(getattr(_MODEL, "_last_generation_stats", {}))
    stats["prompt_tokens"] = int(inputs["input_ids"].shape[1])
    stats["completion_tokens"] = len(ids)
    stats["repetition_penalty"] = repetition_penalty
    stats["decoding"] = decoding
    stats["request_stop"] = list(normalized_request_stop)
    stats["request_stop_token_ids"] = list(request_stop_ids)
    stats["terminated_by_request_stop"] = terminated_by_request_stop
    stats["no_stop_trim"] = no_stop_trim
    stats["finish_reason"] = finish_reason
    return text, stats


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:
        pass

    def _send(self, code: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send(
                200,
                {
                    "status": "ok",
                    "service_contract": "native-v1",
                    "non_thinking": True,
                    "max_new_tokens": MAX_NEW_TOKENS_CAP,
                    "max_model_len": MAX_MODEL_LEN,
                    "requests": _REQUESTS,
                    "failures": _FAILURES,
                    "total_seconds": _TOTAL_SECONDS,
                    "total_output_tokens": _TOTAL_OUTPUT_TOKENS,
                    "total_nfe": _TOTAL_NFE,
                    "stop_requests": _STOP_REQUESTS,
                    "stop_terminations": _STOP_TERMINATIONS,
                    "last_request_stop": _LAST_REQUEST_STOP,
                    "last_request_stop_token_ids": _LAST_REQUEST_STOP_TOKEN_IDS,
                    "last_generation_parameters": _LAST_GENERATION_PARAMETERS,
                    "gpu_allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
                    "gpu_reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / (1 << 30),
                    "inflight_generation": dict(
                        getattr(_MODEL, "_inflight_generation_stats", {}) if _MODEL is not None else {}
                    ),
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        global _REQUESTS, _FAILURES, _TOTAL_SECONDS, _TOTAL_OUTPUT_TOKENS, _TOTAL_NFE
        global _STOP_REQUESTS, _STOP_TERMINATIONS
        global _LAST_REQUEST_STOP, _LAST_REQUEST_STOP_TOKEN_IDS
        global _LAST_GENERATION_PARAMETERS
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) if length else b"{}")
            if not isinstance(request, dict): raise ValueError('request must be an object')
            if request.get('model', MODEL_NAME) != MODEL_NAME:
                raise ValueError('unknown model; see /v1/models')
            validate_thinking(request, True)
            validate_budget(request.get('max_tokens', 512), MAX_NEW_TOKENS_CAP)
            temperature = float(request.get("temperature", 0.0))
            top_p = float(request.get("top_p", 1.0))
            raw_top_k = request.get("top_k")
            top_k = None if raw_top_k is None else int(raw_top_k)
            skip_special_tokens = request.get(
                "skip_special_tokens", DEFAULT_SKIP_SPECIAL_TOKENS
            )
            spaces_between_special_tokens = request.get(
                "spaces_between_special_tokens",
                DEFAULT_SPACES_BETWEEN_SPECIAL_TOKENS,
            )
            no_stop_trim = request.get("no_stop_trim", False)
            repetition_penalty = float(request.get("repetition_penalty", 1.0))
            decoding = str(request.get("dlm_decoding", "speculative"))
            # The plain-Qwen3 benchmark may opt into request-scoped hierarchy
            # controls.  The default evaluator leaves this envelope untouched,
            # preserving its historical process-level decode contract.
            decode_overrides = {}
            if os.environ.get("DLM_ENABLE_REQUEST_DECODE_OVERRIDES", "0") == "1":
                custom = request.get("custom_params")
                if isinstance(custom, dict) and isinstance(custom.get("gam_dlm_decode"), dict):
                    decode_overrides = dict(custom["gam_dlm_decode"])
            _require(temperature >= 0.0, "temperature must be non-negative")
            _require(0.0 < top_p <= 1.0, "top_p must be in (0, 1]")
            _require(
                top_k is None
                or (
                    not isinstance(raw_top_k, bool)
                    and float(raw_top_k).is_integer()
                    and top_k > 0
                ),
                "top_k must be null/omitted or a positive integer",
            )
            _require(repetition_penalty > 0.0, "repetition_penalty must be strictly positive")
            _require(
                isinstance(skip_special_tokens, bool),
                "skip_special_tokens must be boolean",
            )
            _require(
                isinstance(spaces_between_special_tokens, bool),
                "spaces_between_special_tokens must be boolean",
            )
            _require(isinstance(no_stop_trim, bool), "no_stop_trim must be boolean")
            _require(not bool(request.get("stream", False)), "streaming responses are not supported")
            _require(int(request.get("n", 1)) == 1, "DLM evaluation supports exactly one completion")
            with _LOCK:
                text, stats = run_inference(
                    request.get("messages", []),
                    request.get("max_tokens", 512),
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    decoding=decoding,
                    stop=request.get("stop"),
                    no_stop_trim=no_stop_trim,
                    decode_overrides=decode_overrides,
                )
            elapsed = time.monotonic() - started
            _REQUESTS += 1
            _TOTAL_SECONDS += elapsed
            _TOTAL_OUTPUT_TOKENS += int(stats.get("completion_tokens", 0))
            _TOTAL_NFE += int(stats.get("nfe", 0))
            _LAST_GENERATION_PARAMETERS = {
                "max_tokens": int(request.get("max_tokens", 512)),
                "effective_max_tokens": int(stats.get("requested_max_tokens", request.get("max_tokens", 512))),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
                "decoding": str(stats.get("decoding", decoding)),
                "logits_processing_order": stats.get("logits_processing_order"),
            }
            request_stop = list(stats.get("request_stop", []))
            if request_stop:
                _STOP_REQUESTS += 1
                _LAST_REQUEST_STOP = request_stop
                _LAST_REQUEST_STOP_TOKEN_IDS = list(
                    stats.get("request_stop_token_ids", [])
                )
            if stats.get("terminated_by_request_stop") is True:
                _STOP_TERMINATIONS += 1
            completion_tokens = int(stats.get("completion_tokens", 0))
            prompt_tokens = int(stats.get("prompt_tokens", 0))
            self._send(
                200,
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": stats.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "dlm_stats": {**stats, "latency_seconds": elapsed},
                },
            )
        except (ValueError, TypeError) as exc:
            _FAILURES += 1
            self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except Exception as exc:
            _FAILURES += 1
            traceback.print_exc()
            self._send(500, {"error": {"message": str(exc), "type": type(exc).__name__}})


def main() -> None:
    load_model()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"[server] listening on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
