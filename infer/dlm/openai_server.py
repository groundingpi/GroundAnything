#!/usr/bin/env python3
"""OpenAI backend exposing all three GAM DLM inference modes.

This module reuses only the established request/image/checkpoint plumbing from
the shared inference backend. Decoding itself is dispatched through ``infer.dlm`` so
the canonical evaluation task definitions remain unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


GAM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GAM_ROOT))

from infer.dlm import DLMInferenceConfig, DLMInferenceEngine  # noqa: E402
from infer.dlm.causal import causal_generate_cached  # noqa: E402
from infer.dlm.prefix_cache import Qwen35PrefixDraftRunner  # noqa: E402
from infer.dlm.prompt_contract import infer_coordinate_group_width  # noqa: E402
from infer.dlm.stop_contract import (  # noqa: E402
    first_token_sequence_match,
    token_stop_sequences,
)
from infer.dlm import legacy_backend as backend  # noqa: E402


DEFAULT_MODE = os.environ.get("DLM_INFERENCE_MODE", "speculative")
LOAD_DLM_WRAPPER = os.environ.get("DLM_LOAD_WRAPPER", "1") == "1"
BLOCK_SIZE = int(os.environ.get("DLM_INFERENCE_BLOCK_SIZE", "32"))
SUB_BLOCK_SIZE = int(os.environ.get("DLM_SUB_BLOCK_SIZE", "8"))
DENOISE_STEPS = int(os.environ.get("DLM_DENOISE_STEPS", "8"))
CONFIDENCE_THRESHOLD = float(os.environ.get("DLM_CONFIDENCE_THRESHOLD", "0.9"))
ACCEPTANCE_POLICY = os.environ.get("DLM_ACCEPTANCE_POLICY", "confidence")
ENTROPY_THRESHOLD = float(os.environ.get("DLM_ENTROPY_THRESHOLD", "0.8"))
USE_PREFIX_CACHE = os.environ.get("DLM_USE_PREFIX_CACHE", "0") == "1"
HIERARCHY_FULL_BLOCK = os.environ.get("DLM_HIERARCHY_FULL_BLOCK", "0") == "1"
ENFORCE_STRUCTURED_TERMINATION = (
    os.environ.get("DLM_ENFORCE_STRUCTURED_TERMINATION", "0") == "1"
)
STRUCTURED_MAX_COORDINATE_GROUPS = int(
    os.environ.get("DLM_STRUCTURED_MAX_COORDINATE_GROUPS", "0")
)
COMMIT_STRUCTURAL_BOUNDARIES = (
    os.environ.get("DLM_COMMIT_STRUCTURAL_BOUNDARIES", "0") == "1"
)
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("DLM_NO_REPEAT_NGRAM_SIZE", "32"))
# Request-scoped decoder overrides are opt-in.  The historical evaluator uses
# immutable process-level settings; keeping this gate off by default preserves
# that contract while allowing the COCO A/B harness to compare step/entropy
# cells without starting a new 10-GB model process for every request.
ENABLE_REQUEST_DECODE_OVERRIDES = (
    os.environ.get("DLM_ENABLE_REQUEST_DECODE_OVERRIDES", "0") == "1"
)
SEMANTIC_STOP_TOKENS = tuple(
    token.strip()
    for token in os.environ.get(
        "DLM_SEMANTIC_STOP_TOKENS", "<|box_end|>,<|quad_end|>"
    ).split(",")
    if token.strip()
)
_ENGINE: DLMInferenceEngine | None = None
_POST_REQUEST_CUDA_CACHE_CLEANUPS = 0
_POST_REQUEST_PEAK_RESERVED_BYTES = 0


def _release_fragmented_cuda_cache_after_request(
    response_ids: torch.Tensor,
) -> dict[str, int | float | bool]:
    """Bound cross-request allocator fragmentation after inference unwinds."""

    global _POST_REQUEST_CUDA_CACHE_CLEANUPS, _POST_REQUEST_PEAK_RESERVED_BYTES
    if backend.CACHE_CLEANUP_INTERVAL_TOKENS == 0 or not response_ids.is_cuda:
        return {
            "post_request_cuda_cache_cleanup": False,
            "post_request_cuda_cache_cleanups_total": _POST_REQUEST_CUDA_CACHE_CLEANUPS,
            "post_request_peak_reserved_gib": (
                _POST_REQUEST_PEAK_RESERVED_BYTES / (1 << 30)
            ),
        }
    reserved = int(torch.cuda.memory_reserved(response_ids.device))
    allocated = int(torch.cuda.memory_allocated(response_ids.device))
    _POST_REQUEST_PEAK_RESERVED_BYTES = max(
        _POST_REQUEST_PEAK_RESERVED_BYTES, reserved
    )
    total = int(torch.cuda.get_device_properties(response_ids.device).total_memory)
    should_release = (
        reserved >= int(total * backend.CACHE_CLEANUP_FRACTION)
        and reserved > 2 * allocated
    )
    if should_release:
        torch.cuda.empty_cache()
        _POST_REQUEST_CUDA_CACHE_CLEANUPS += 1
    return {
        "post_request_cuda_cache_cleanup": should_release,
        "post_request_cuda_cache_cleanups_total": _POST_REQUEST_CUDA_CACHE_CLEANUPS,
        "post_request_peak_reserved_gib": (
            _POST_REQUEST_PEAK_RESERVED_BYTES / (1 << 30)
        ),
        "post_request_reserved_before_gib": reserved / (1 << 30),
        "post_request_allocated_gib": allocated / (1 << 30),
        "post_request_reserved_after_gib": (
            torch.cuda.memory_reserved(response_ids.device) / (1 << 30)
        ),
    }


def _load_native_vlm() -> None:
    """Load the pre-conversion VLM for a fair cached-AR speed baseline."""

    backend.install_qwen35_fastpath()
    backend._PROCESSOR = backend.AutoProcessor.from_pretrained(
        backend.BASE_MODEL, trust_remote_code=False
    )
    backend._TOKENIZER = backend.AutoTokenizer.from_pretrained(
        backend.BASE_MODEL, trust_remote_code=False
    )
    backend._PROCESSOR.tokenizer = backend._TOKENIZER
    native = backend.Qwen3_5ForConditionalGeneration.from_pretrained(
        backend.BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation={
            "vision_config": backend.VISION_ATTN,
            "text_config": backend.TEXT_ATTN,
        },
        trust_remote_code=False,
    )
    if native.get_input_embeddings().weight.shape[0] != len(backend._TOKENIZER):
        native.resize_token_embeddings(len(backend._TOKENIZER), mean_resizing=False)
    im_end_id = int(backend._TOKENIZER.convert_tokens_to_ids("<|im_end|>"))
    backend._MODEL = backend.GAMQwen35DLM(
        native,
        mask_token_id=0,
        im_end_token_id=im_end_id,
        block_size=32,
    ).eval()
    backend._MODEL.to(backend._DEVICE)
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "READY",
                "runtime": "native_preconversion_vlm",
                "base_model": str(backend.BASE_MODEL),
                "device": str(backend._DEVICE),
                "vocab_size": len(backend._TOKENIZER),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _normalized_mode(requested: str) -> str:
    # The legacy handler supplies "speculative" when the request has no mode;
    # an explicitly configured server must still use its selected default.
    mode = DEFAULT_MODE if requested == "speculative" and DEFAULT_MODE != "speculative" else requested
    aliases = {
        "mdm_step": "hierarchy_step",
        "mdm_dynamic": "hierarchy_dynamic",
        "hierarchy": "hierarchy_dynamic",
    }
    return aliases.get(mode, mode)


def _request_decode_overrides(request: object) -> dict[str, Any]:
    """Read the narrow, validated benchmark-only decode override envelope."""

    if not ENABLE_REQUEST_DECODE_OVERRIDES or not isinstance(request, dict):
        return {}
    custom = request.get("custom_params")
    if not isinstance(custom, dict):
        return {}
    value = custom.get("gam_dlm_decode")
    if not isinstance(value, dict):
        return {}
    allowed = {
        "mode", "block_size", "sub_block_size", "denoise_steps",
        "confidence_threshold", "acceptance_policy", "entropy_threshold",
        "use_prefix_cache", "hierarchy_full_block",
        "enforce_structured_termination", "structured_max_coordinate_groups",
        "commit_structural_boundaries",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown request DLM override parameters: {unknown}")
    return dict(value)


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
    decode_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if backend._MODEL is None or _ENGINE is None:
        raise RuntimeError("DLM inference engine is not loaded")
    inputs = backend._prepare_inputs(messages)
    coordinate_group_width = infer_coordinate_group_width(messages)
    max_new_tokens = backend.validate_budget(requested_max_tokens, backend.MAX_NEW_TOKENS_CAP,
                                              inputs["input_ids"].shape[-1], backend.MAX_MODEL_LEN)
    normalized_request_stop, request_stop_sequences = token_stop_sequences(
        backend._TOKENIZER, stop
    )
    request_stop_ids = tuple(
        sequence[0] for sequence in request_stop_sequences if len(sequence) == 1
    )
    backend._log_stop_contract(normalized_request_stop, request_stop_ids)
    stop_ids = tuple(
        dict.fromkeys(
            token
            for token in (
                int(backend._TOKENIZER.eos_token_id)
                if backend._TOKENIZER.eos_token_id is not None
                else -1,
                int(backend._TOKENIZER.convert_tokens_to_ids("<|im_end|>")),
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
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    overrides = dict(decode_overrides or {})
    # ``mode`` is intentionally restricted to the two hierarchy routes here;
    # speculative requests cannot consume entropy/step controls and use the
    # explicit legacy path below.
    requested_mode = str(overrides.get("mode", decoding))
    mode = _normalized_mode(requested_mode)
    if mode == "causal_full_prefix_oracle" and backend.VALIDATION_ORACLE_ENABLED:
        response_ids = backend._MODEL.causal_generate_full_prefix(**generation_args)
    elif mode == "causal_cached":
        response_ids = causal_generate_cached(backend._MODEL, **generation_args)
    else:
        config = DLMInferenceConfig(
            mode=mode,
            block_size=int(overrides.get("block_size", BLOCK_SIZE)),
            sub_block_size=int(overrides.get("sub_block_size", SUB_BLOCK_SIZE)),
            denoise_steps=int(overrides.get("denoise_steps", DENOISE_STEPS)),
            confidence_threshold=float(overrides.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
            acceptance_policy=str(overrides.get("acceptance_policy", ACCEPTANCE_POLICY)),
            entropy_threshold=float(overrides.get("entropy_threshold", ENTROPY_THRESHOLD)),
            use_prefix_cache=bool(overrides.get("use_prefix_cache", USE_PREFIX_CACHE)),
            hierarchy_full_block=bool(overrides.get("hierarchy_full_block", HIERARCHY_FULL_BLOCK)),
            enforce_structured_termination=bool(overrides.get("enforce_structured_termination", ENFORCE_STRUCTURED_TERMINATION)),
            structured_max_coordinate_groups=int(overrides.get("structured_max_coordinate_groups", STRUCTURED_MAX_COORDINATE_GROUPS)),
            commit_structural_boundaries=bool(overrides.get("commit_structural_boundaries", COMMIT_STRUCTURAL_BOUNDARIES)),
        )
        generation_args.update(
            {
                "cuda_cache_cleanup_interval_tokens": backend.CACHE_CLEANUP_INTERVAL_TOKENS,
                "cuda_cache_cleanup_fraction": backend.CACHE_CLEANUP_FRACTION,
                "stop_token_sequences": request_stop_sequences,
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            }
        )
        if mode != "speculative":
            semantic_stop_ids = tuple(
                int(backend._TOKENIZER.convert_tokens_to_ids(token))
                for token in SEMANTIC_STOP_TOKENS
            )
            if any(token_id < 0 for token_id in semantic_stop_ids):
                raise RuntimeError(
                    f"tokenizer is missing semantic stop tokens: {SEMANTIC_STOP_TOKENS}"
                )
            generation_args["semantic_stop_token_ids"] = semantic_stop_ids
            if config.enforce_structured_termination:
                structural_tokens = {
                    name: int(backend._TOKENIZER.convert_tokens_to_ids(token))
                    for name, token in (
                        ("object_ref_start", "<|object_ref_start|>"),
                        ("object_ref_end", "<|object_ref_end|>"),
                        ("box_start", "<|box_start|>"),
                        ("box_end", "<|box_end|>"),
                        ("quad_start", "<|quad_start|>"),
                        ("quad_end", "<|quad_end|>"),
                    )
                }
                coordinate_min = int(backend._TOKENIZER.convert_tokens_to_ids("<0>"))
                coordinate_max = int(backend._TOKENIZER.convert_tokens_to_ids("<999>"))
                coordinate_separator = backend._TOKENIZER.encode(
                    ",", add_special_tokens=False
                )
                if any(value < 0 for value in structural_tokens.values()) or not (
                    0 <= coordinate_min <= coordinate_max
                ) or len(coordinate_separator) != 1:
                    raise RuntimeError("tokenizer is missing the structured grounding contract")
                generation_args.update(
                    {
                        "structural_token_ids": structural_tokens,
                        "coordinate_token_id_range": (coordinate_min, coordinate_max),
                        "coordinate_separator_token_ids": tuple(
                            int(value) for value in coordinate_separator
                        ),
                        "coordinate_group_width": coordinate_group_width,
                    }
                )
        response_ids = _ENGINE.generate(config, **generation_args)

    ids = response_ids[0].tolist()
    request_stop_match = first_token_sequence_match(ids, request_stop_sequences)
    terminated_by_request_stop = request_stop_match is not None
    finish_reason = (
        "stop"
        if terminated_by_request_stop or (ids and ids[-1] in stop_ids)
        else "length"
        if len(ids) >= max_new_tokens
        else "stop"
    )
    if request_stop_match is not None:
        stop_end = request_stop_match[1] if no_stop_trim else request_stop_match[0]
        ids = ids[:stop_end]
    else:
        while ids and ids[-1] in stop_ids:
            if no_stop_trim and ids[-1] in request_stop_ids:
                break
            ids.pop()
    text = backend._TOKENIZER.decode(
        ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=spaces_between_special_tokens,
    ).strip()
    cleanup_stats = _release_fragmented_cuda_cache_after_request(response_ids)
    stats = dict(getattr(backend._MODEL, "_last_generation_stats", {}))
    stats.update(cleanup_stats)
    stats.update(
        {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": len(ids),
            "requested_max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
            "decoding": mode,
            "request_stop": list(normalized_request_stop),
            "request_stop_token_ids": list(request_stop_ids),
            "request_stop_token_sequences": [
                list(sequence) for sequence in request_stop_sequences
            ],
            "terminated_by_request_stop": terminated_by_request_stop,
            "no_stop_trim": no_stop_trim,
            "finish_reason": finish_reason,
            "coordinate_group_width": coordinate_group_width,
            "request_decode_overrides_enabled": ENABLE_REQUEST_DECODE_OVERRIDES,
            "effective_decode_overrides": overrides,
        }
    )
    backend._MODEL._last_generation_stats = dict(stats)
    backend._MODEL._inflight_generation_stats = {**stats, "active": False}
    return text, stats


def main() -> None:
    global _ENGINE
    if LOAD_DLM_WRAPPER:
        backend.load_model()
    else:
        _load_native_vlm()
    _ENGINE = DLMInferenceEngine(backend._MODEL)
    prefix_cache_source_provenance = None
    if USE_PREFIX_CACHE:
        # Validate pinned sources and model layer interfaces before announcing
        # READY. Runtime cache length/layout is checked again on every block.
        Qwen35PrefixDraftRunner(backend._MODEL, BLOCK_SIZE)
        prefix_cache_source_provenance = (
            Qwen35PrefixDraftRunner.qwen_source_provenance()
        )
    backend.run_inference = run_inference
    print(
        json.dumps(
            {
                "status": "DLM_INFERENCE_READY",
                "load_dlm_wrapper": LOAD_DLM_WRAPPER,
                "mode": DEFAULT_MODE,
                "block_size": BLOCK_SIZE,
                "sub_block_size": SUB_BLOCK_SIZE,
                "denoise_steps": DENOISE_STEPS,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "use_prefix_cache": USE_PREFIX_CACHE,
                "hierarchy_full_block": HIERARCHY_FULL_BLOCK,
                "enforce_structured_termination": ENFORCE_STRUCTURED_TERMINATION,
                "structured_max_coordinate_groups": STRUCTURED_MAX_COORDINATE_GROUPS,
                "coordinate_group_width": "task-aware-point2-bbox4",
                "commit_structural_boundaries": COMMIT_STRUCTURAL_BOUNDARIES,
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "prefix_cache_source_provenance": prefix_cache_source_provenance,
                "semantic_stop_tokens": SEMANTIC_STOP_TOKENS,
                "cuda_cache_cleanup_interval_tokens": backend.CACHE_CLEANUP_INTERVAL_TOKENS,
                "cuda_cache_cleanup_fraction": backend.CACHE_CLEANUP_FRACTION,
                "cuda_cache_cleanup_scope": "hierarchy_block_and_post_request",
                "openai_stop_contract": "decoded_token_sequence",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server = backend.ThreadingHTTPServer(("127.0.0.1", backend.PORT), backend.Handler)
    server.daemon_threads = True
    print(f"[server] listening on 127.0.0.1:{backend.PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
