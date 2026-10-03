#!/usr/bin/env python3
"""Native reference server for GroundAnything/Kimi-K3 + Qwen3 DLM.

The production H800 evaluator uses the custom SGLang ``GAMDecodeV2Block``
backend.  PPU images do not provide a compatible ``sgl-kernel`` binary, so
this entry point loads the same full wrapper checkpoint into the canonical
PyTorch ``GAMQwen3DLM`` model and reuses ``infer.dlm.openai_server`` for the
HTTP and hierarchy-decoding contracts.  It is intentionally a reference
backend, not a silent SGLang fallback.
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer

from infer.dependency_guard import bundled_transformers

# Must run before Torch, Transformers and indirect model/backend imports.
_TRANSFORMERS, _TRANSFORMERS_REVISION = bundled_transformers()

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from infer.dlm import openai_server as server
from infer.dlm.model_contract import validate_qwen3_base_config
from models.dlm.hybrid import MASK_TOKEN
from models.dlm.vlm import GAMQwen3DLM


def load_qwen3_model() -> None:
    """Load the immutable Stage-I model and exact full DLM wrapper weights."""

    backend = server.backend
    base_model = backend.BASE_MODEL
    checkpoint = backend.DLM_CHECKPOINT
    weights = checkpoint / "model.safetensors"
    backend._require((base_model / "config.json").is_file(), f"base model is incomplete: {base_model}")
    backend._require(weights.is_file(), f"DLM weights are missing: {weights}")

    validate_qwen3_base_config(base_model)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    processor.tokenizer = tokenizer
    attention = os.environ.get("DLM_QWEN3_ATTN_IMPLEMENTATION", "flash_attention_2")
    native = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation=attention,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    backend._require(native.config.model_type in {"groundinganything", "groundinganything_vlm"}, "expected GroundAnything-VLM config")
    backend._require(native.config.text_config.model_type == "qwen3", "expected plain Qwen3 language tower")
    if native.get_input_embeddings().weight.shape[0] != len(tokenizer):
        native.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    backend._require(
        mask_id >= 0 and tokenizer.encode(MASK_TOKEN, add_special_tokens=False) == [mask_id],
        "checkpoint tokenizer has no atomic DLM mask token",
    )
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    wrapper = GAMQwen3DLM(
        native,
        mask_token_id=mask_id,
        im_end_token_id=im_end_id,
        block_size=server.BLOCK_SIZE,
    ).eval()

    state = load_file(str(weights), device="cpu")
    backend._require(
        state and all(key.startswith("base_model.") for key in state),
        "checkpoint is not a full GAMQwen3DLM wrapper state dict",
    )
    incompatible = wrapper.load_state_dict(state, strict=False)
    backend._require(not incompatible.missing_keys, f"missing DLM keys: {incompatible.missing_keys[:10]}")
    backend._require(not incompatible.unexpected_keys, f"unexpected DLM keys: {incompatible.unexpected_keys[:10]}")
    del state

    backend._PROCESSOR = processor
    backend._TOKENIZER = tokenizer
    backend._MODEL = wrapper.to(backend._DEVICE)
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "READY",
                "runtime": "pytorch_qwen3_dlm_reference",
                "base_model": str(base_model),
                "checkpoint": str(checkpoint),
                "device": str(backend._DEVICE),
                "dtype": "bfloat16",
                "attention": attention,
                "block_size": server.BLOCK_SIZE,
                "vocab_size": len(tokenizer),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def prepare_qwen3_request(messages):
    from infer.dlm.qwen3_inputs import prepare_qwen3_inputs
    backend = server.backend
    normalized = backend._normalize_messages(messages)
    inputs = prepare_qwen3_inputs(backend._PROCESSOR, normalized)
    return {key: value.to(device=backend._DEVICE, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def main() -> None:
    server.backend.load_model = load_qwen3_model
    server.backend._prepare_inputs = prepare_qwen3_request
    # The reference decoder is serialized by the backend model lock.  Keep
    # request execution on one OS thread as well: the PPU TorchInductor build
    # stores CUDA-graph tree state in thread-local storage, so a compiled graph
    # cannot be replayed safely by ThreadingHTTPServer's next worker thread.
    # This changes no model/decode semantics and adds no serialization beyond
    # the lock that already guards every inference call.
    server.backend.Handler.protocol_version = "HTTP/1.0"
    server.backend.ThreadingHTTPServer = HTTPServer
    server.main()


if __name__ == "__main__":
    main()
