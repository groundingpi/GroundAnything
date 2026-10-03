"""Fail-closed ABI bridge for Fast-dLLM SGLang on the H800 Torch image.

The immutable H800 image uses NVIDIA Torch 2.8.  ``sgl-kernel`` 0.2.9 is the
loadable binary for that ABI, while Fast-dLLM's newer SGLang Python tree also
imports symbols added for optional GPTQ/Marlin/GGUF quantization, MoE reduction,
and fused KV writes.  RLV3 uses BF16, no quantization, Qwen3 dense attention,
and never calls any of those paths.  Supplying raising placeholders makes those
optional imports explicit without silently substituting any numerical kernel.
"""

from __future__ import annotations

import sys
import types


def _unsupported_optional_kernel(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "RLV3 attempted an unsupported optional SGLang kernel path; "
        "BF16 GAM dense causal rollout must not use quantization, MoE, or "
        "fused-set-KV"
    )


class _UnsupportedFusedSetKVBufferArg:
    def __init__(self, *args, **kwargs):
        _unsupported_optional_kernel(*args, **kwargs)


def install_sgl_kernel_compat() -> dict[str, object]:
    import sgl_kernel

    installed: list[str] = []
    if not hasattr(sgl_kernel, "gptq_gemm"):
        sgl_kernel.gptq_gemm = _unsupported_optional_kernel
        installed.append("gptq_gemm")
    if not hasattr(sgl_kernel, "gptq_shuffle"):
        sgl_kernel.gptq_shuffle = _unsupported_optional_kernel
        installed.append("gptq_shuffle")
    if not hasattr(sgl_kernel, "gptq_marlin_gemm"):
        sgl_kernel.gptq_marlin_gemm = _unsupported_optional_kernel
        installed.append("gptq_marlin_gemm")
    if not hasattr(sgl_kernel, "FusedSetKVBufferArg"):
        sgl_kernel.FusedSetKVBufferArg = _UnsupportedFusedSetKVBufferArg
        installed.append("FusedSetKVBufferArg")
    # Fast-dLLM's SGLang quantization registry imports GGUF eagerly, and GGUF
    # imports these MoE reductions even for a dense, unquantized model.  Keep
    # module discovery possible while making accidental execution fatal.
    if not hasattr(sgl_kernel, "moe_sum"):
        sgl_kernel.moe_sum = _unsupported_optional_kernel
        installed.append("moe_sum")
    if not hasattr(sgl_kernel, "moe_sum_reduce"):
        sgl_kernel.moe_sum_reduce = _unsupported_optional_kernel
        installed.append("moe_sum_reduce")
    quantization_module_name = "sgl_kernel.quantization"
    if quantization_module_name not in sys.modules:
        quantization = types.ModuleType(quantization_module_name)
        for name in (
            "ggml_dequantize",
            "ggml_moe_a8",
            "ggml_moe_a8_vec",
            "ggml_moe_get_block_size",
            "ggml_mul_mat_a8",
            "ggml_mul_mat_vec_a8",
        ):
            setattr(quantization, name, _unsupported_optional_kernel)
        sys.modules[quantization_module_name] = quantization
        sgl_kernel.quantization = quantization
        installed.append(quantization_module_name)
    # SGLang imports every host-cache transfer signature even when HiCache is
    # disabled.  Kernel 0.2.9 provides the older subset.  Fill only missing
    # names with raising placeholders; a configuration that accidentally
    # enables host KV transfer therefore fails instead of changing numerics.
    try:
        import sgl_kernel.kvcacheio as kvcacheio
    except ImportError:
        kvcacheio = None
    if kvcacheio is not None:
        for name in (
            "transfer_kv_all_layer",
            "transfer_kv_all_layer_direct_lf_pf",
            "transfer_kv_all_layer_lf_pf",
            "transfer_kv_all_layer_lf_ph",
            "transfer_kv_all_layer_mla",
            "transfer_kv_all_layer_mla_lf_pf",
            "transfer_kv_direct",
            "transfer_kv_per_layer",
            "transfer_kv_per_layer_direct_pf_lf",
            "transfer_kv_per_layer_mla",
            "transfer_kv_per_layer_mla_pf_lf",
            "transfer_kv_per_layer_pf_lf",
            "transfer_kv_per_layer_ph_lf",
        ):
            if not hasattr(kvcacheio, name):
                setattr(kvcacheio, name, _unsupported_optional_kernel)
                installed.append(f"sgl_kernel.kvcacheio.{name}")
    return {
        "status": "PASS",
        "kernel_module": str(sgl_kernel.__file__),
        "fail_closed_optional_symbols": sorted(installed),
    }
