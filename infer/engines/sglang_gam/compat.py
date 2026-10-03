"""Small compatibility layer for the H800 image's old ``sgl-kernel``.

The Fast-dLLM fork is newer than the image-provided CUDA wheel.  The 0.2.x
wheel contains the dense Qwen3 kernels we need, but it predates the public
``FusedSetKVBufferArg`` Python dataclass and the named RoPE helper exported by
newer SGLang.  Importing the model registry consequently fails before our
external model is even considered.  This module fills only that API seam; it
does not replace a CUDA kernel or alter model weights.

The helper first uses the native torch operator when the old wheel exposes it.
The pure-PyTorch path is a fallback for unsupported accelerator capabilities and
for environments whose operator was compiled without the fused KV-write
variant.  It is deliberately fail-closed for unexpected tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import sys
import types
import os
import builtins
import importlib.metadata as _metadata
import threading
from typing import Optional


# Imports for quantization/speculative/MoE implementations are eagerly walked
# by SGLang's model registry even when the selected model is dense Qwen3.  The
# old 0.2.x wheel legitimately does not export those optional symbols.  They
# must be importable so the registry can reach Qwen3, but they must never be
# selected silently by this route.  The placeholders raise if an unsupported
# optional implementation is actually called.
_OPTIONAL_SYMBOLS = {
    "apply_shuffle_mul_sum", "apply_token_bitmask_inplace_cuda",
    "awq_dequantize", "awq_marlin_moe_repack", "awq_marlin_repack",
    "bmm_fp8", "build_tree_kernel_efficient", "causal_conv1d_fwd",
    "causal_conv1d_update", "concat_mla_absorb_q", "concat_mla_k",
    "cutlass_fp4_group_mm", "cutlass_mla_decode", "cutlass_mla_get_workspace_size",
    "cutlass_scaled_fp4_mm", "cutlass_w4a8_moe_mm", "dsv3_fused_a_gemm",
    "dsv3_router_gemm", "es_fp8_blockwise_scaled_grouped_mm", "fast_topk",
    "fast_topk_transform_fused", "fast_topk_transform_ragged_fused", "fast_topk_v2",
    "fp8_blockwise_scaled_grouped_mm", "fp8_blockwise_scaled_mm", "fp8_scaled_mm",
    "fused_qk_norm_rope", "gptq_gemm", "gptq_marlin_gemm", "gptq_shuffle",
    "hadamard_transform", "int8_scaled_mm", "kimi_k2_moe_fused_gate",
    "merge_state_v2", "min_p_sampling_from_probs", "moe_align_block_size",
    "moe_fused_gate", "moe_sum", "moe_sum_reduce", "prepare_moe_input",
    "qserve_w4a8_per_chn_gemm", "qserve_w4a8_per_group_gemm",
    "scaled_fp4_experts_quant", "scaled_fp4_quant", "segment_packbits",
    "sgl_per_tensor_quant_fp8", "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_fp8", "sgl_per_token_group_quant_int8",
    "sgl_per_token_quant_fp8", "shuffle_rows", "silu_and_mul",
    "top_k_renorm_prob", "top_k_top_p_sampling_from_probs", "top_p_renorm_prob",
    "topk_sigmoid", "topk_softmax", "tree_speculative_sampling_target_only",
    "verify_tree_greedy", "weak_ref_tensor",
    "ggml_dequantize", "ggml_moe_a8", "ggml_moe_a8_vec",
    "ggml_moe_get_block_size", "ggml_mul_mat_a8", "ggml_mul_mat_vec_a8",
}

_GAM_TORCH_LIB = None
_GAM_METADATA_PATCHED = False
_GAM_FP8_SCALE_PATCHED = False
_GAM_FLASHINFER_MASK_PATCHED = False
_GAM_TRITON_DLLM_GRAPH_PATCHED = False
_GAM_FAST_EAGER_LOCK = threading.RLock()


def install_triton_dllm_graph_patch() -> bool:
    """Add an opt-in, route-local Triton CUDA-Graph path for dLLM B32.

    The Fast-dLLM SGLang fork supports ``DLLM_EXTEND`` graphs only in its
    FlashInfer backend.  Its Triton backend already has all kernels needed by
    GAM, but the graph metadata capture/replay dispatch rejects this forward
    mode.  This patch fills that narrow gap without modifying the vendored
    reference tree or any ordinary SGLang route.

    The captured graph is the *bidirectional denoise* forward: the temporary
    B32 rows do not write KV.  The explicit final causal commit remains eager
    and follows the native Triton path.  GAMSpeculativeBlock consequently uses
    the same graph for its bidirectional draft and retains its eager causal
    verifier.  The switch is fail-closed and disabled unless
    ``GAM_SGLANG_ALLOW_TRITON_DLM_GRAPH=1`` is present in the dedicated GAM
    server process.
    """

    global _GAM_TRITON_DLLM_GRAPH_PATCHED
    if _GAM_TRITON_DLLM_GRAPH_PATCHED:
        return True
    if os.environ.get("GAM_SGLANG_ALLOW_TRITON_DLM_GRAPH", "0") != "1":
        return False

    try:
        import torch
        from sglang.srt.layers.attention.triton_backend import (
            ForwardMetadata,
            TritonAttnBackend,
            create_flashinfer_kv_indices_triton,
        )
        from sglang.srt.layers.radix_attention import AttentionType
    except Exception:
        return False

    marker = "_gam_triton_dllm_graph_patch"
    if getattr(TritonAttnBackend, marker, False):
        _GAM_TRITON_DLLM_GRAPH_PATCHED = True
        return True

    block_size = int(os.environ.get("GAM_SGLANG_BLOCK_SIZE", "32"))
    if block_size not in (8, 32):
        raise RuntimeError(
            f"unsupported GAM Triton dLLM graph block size: {block_size}"
        )

    original_capture = TritonAttnBackend.init_forward_metadata_capture_cuda_graph
    original_replay = TritonAttnBackend.init_forward_metadata_replay_cuda_graph
    original_forward_extend = TritonAttnBackend.forward_extend

    def _prefix_metadata(self, bs, req_pool_indices, seq_lens):
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            raise RuntimeError(
                "GAM Triton dLLM CUDA Graph does not support sliding-window attention"
            )
        total_lens = seq_lens[:bs]
        prefix_lens = (total_lens - block_size).clamp_min(0)
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[0] = 0
        kv_indptr[1 : bs + 1] = torch.cumsum(prefix_lens, dim=0)
        kv_indices = self.cuda_graph_kv_indices
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices[:bs],
            prefix_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )
        return prefix_lens, kv_indptr, kv_indices

    def capture(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        if not forward_mode.is_dllm_extend():
            return original_capture(
                self,
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        if encoder_lens is not None or spec_info is not None:
            raise RuntimeError(
                "unexpected encoder/spec metadata in GAM Triton dLLM graph capture"
            )
        if int(num_tokens) != int(bs) * block_size:
            raise RuntimeError(
                "GAM Triton dLLM graph requires fixed B rows: "
                f"tokens={num_tokens}, bs={bs}, block={block_size}"
            )
        _, kv_indptr, kv_indices = _prefix_metadata(
            self, bs, req_pool_indices, seq_lens
        )
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[:] = torch.arange(
            0,
            (bs + 1) * block_size,
            step=block_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.forward_metadata = ForwardMetadata(
            None,
            None,
            block_size,
            None,
            kv_indptr,
            kv_indices,
            qo_indptr,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def replay(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        if not forward_mode.is_dllm_extend():
            return original_replay(
                self,
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )
        del seq_lens_sum, seq_lens_cpu
        if encoder_lens is not None or spec_info is not None:
            raise RuntimeError(
                "unexpected encoder/spec metadata in GAM Triton dLLM graph replay"
            )
        _prefix_metadata(self, bs, req_pool_indices, seq_lens)
        if os.environ.get("GAM_SGLANG_TRITON_GRAPH_METADATA_V2", "0") == "1":
            # Eager prefill/verifier uses this same allocation and overwrites
            # it with its own query lengths. Restore ALL graph query offsets
            # before replay, including zero; restoring only KV indices leaves
            # a prompt-length query span pointing past the captured B32 rows.
            self.qo_indptr[: bs + 1].copy_(torch.arange(
                0, (bs + 1) * block_size, block_size,
                dtype=torch.int32, device=self.device,
            ))

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        sinks=None,
    ):
        # CudaGraphRunner constructs a DLLM_EXTEND ForwardBatch without the
        # ordinary extend fields.  Mark it on layer zero and keep the marker for
        # every subsequent layer/warmup execution of that captured batch.
        is_graph_capture = bool(
            forward_batch.forward_mode.is_dllm_extend()
            and (
                getattr(forward_batch, "_gam_triton_dllm_graph_capture", False)
                or getattr(forward_batch, "extend_seq_lens", None) is None
            )
        )
        if not is_graph_capture:
            return original_forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                sinks=sinks,
            )

        forward_batch._gam_triton_dllm_graph_capture = True
        bs = int(forward_batch.batch_size)
        if getattr(forward_batch, "extend_seq_lens", None) is None:
            forward_batch.extend_seq_lens = torch.full(
                (bs,), block_size, dtype=torch.int32, device=self.device
            )
            forward_batch.extend_prefix_lens = (
                forward_batch.seq_lens[:bs] - block_size
            ).clamp_min(0)
            forward_batch.extend_start_loc = torch.arange(
                0,
                bs * block_size,
                step=block_size,
                dtype=torch.int32,
                device=self.device,
            )
            forward_batch.extend_seq_lens_cpu = [block_size] * bs
            forward_batch.extend_prefix_lens_cpu = [
                int(value)
                for value in forward_batch.extend_prefix_lens.detach().cpu().tolist()
            ]
            forward_batch.extend_num_tokens = bs * block_size

        # The default graph is the bidirectional draft. The opt-in verifier
        # owns a separate backend and captures causal attention plus KV writes.
        # Restore the layer attribute after each call for ordinary eager work.
        causal_graph = bool(getattr(self, "_gam_triton_causal_graph", False))
        original_type = layer.attn_type
        layer.attn_type = AttentionType.DECODER if causal_graph else AttentionType.ENCODER_ONLY
        try:
            return original_forward_extend(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                # Unified deterministic attention reads current K/V back
                # from the pool. Its graph must capture those writes too.
                save_kv_cache=bool(
                    causal_graph or (
                        os.environ.get("GAM_SGLANG_TRITON_GRAPH_METADATA_V2", "0") == "1"
                        and self.enable_deterministic
                    )
                ),
                sinks=sinks,
            )
        finally:
            layer.attn_type = original_type

    TritonAttnBackend.init_forward_metadata_capture_cuda_graph = capture
    TritonAttnBackend.init_forward_metadata_replay_cuda_graph = replay
    TritonAttnBackend.forward_extend = forward_extend
    setattr(TritonAttnBackend, marker, True)
    _GAM_TRITON_DLLM_GRAPH_PATCHED = True
    return True


def install_dllm_packed_mask_replay_patch() -> bool:
    """Keep the dLLM CUDA-graph mask packed exactly once.

    FlashInfer's dLLM graph buffer is a packed ``uint8`` mask.  The vendored
    SGLang replay path historically passed that same buffer back to
    ``begin_forward(custom_mask=...)``; FlashInfer consequently interpreted
    each packed byte as eight raw mask entries and packed it a second time.
    This produced very fast, but semantically invalid, graph decoding.

    The patch is deliberately route-local and opt-in.  It wraps only the
    ``FlashInferIndicesUpdaterPrefill.call_begin_forward`` instance call and
    changes ``custom_mask`` to ``packed_custom_mask`` when (and only when) the
    object is the GAM dLLM graph buffer.  All ordinary SGLang custom masks and
    all non-GAM routes retain their native path.
    """

    global _GAM_FLASHINFER_MASK_PATCHED
    if _GAM_FLASHINFER_MASK_PATCHED:
        return True
    if os.environ.get("GAM_SGLANG_FLASHINFER_PACKED_MASK", "0") != "1":
        return False
    try:
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferIndicesUpdaterPrefill,
        )
    except Exception:
        return False

    marker = "_gam_packed_mask_replay_patch"
    if getattr(FlashInferIndicesUpdaterPrefill, marker, False):
        _GAM_FLASHINFER_MASK_PATCHED = True
        return True
    original = FlashInferIndicesUpdaterPrefill.call_begin_forward

    def wrapped(self, *args, **kwargs):
        # ``call_begin_forward`` has a stable keyword in the Fast-dLLM fork;
        # use kwargs rather than positional indexing so this remains safe
        # across the fork's optional fixed-split arguments.
        ragged_mask = kwargs.get("ragged_custom_mask")
        if ragged_mask is None:
            return original(self, *args, **kwargs)
        backend = getattr(self, "attn_backend", None)
        graph_mask = getattr(backend, "dllm_ragged_custom_mask", None)
        packed_graph_mask = getattr(
            backend, "dllm_ragged_packed_custom_mask", None
        )
        if graph_mask is None or ragged_mask is not graph_mask:
            return original(self, *args, **kwargs)

        # The original method selects the dLLM wrapper internally.  Temporarily
        # adapt only that wrapper's plan alias so the existing scheduling and
        # buffer setup stay unchanged.  Restore it immediately after the call;
        # no class or shared SGLang method is mutated for other routes.
        wrapper = getattr(backend, "dllm_spec_wrapper_ragged", None)
        if wrapper is None:
            return original(self, *args, **kwargs)
        begin = getattr(wrapper, "begin_forward", None)
        if begin is None:
            return original(self, *args, **kwargs)

        def packed_begin(*begin_args, **begin_kwargs):
            custom = begin_kwargs.pop("custom_mask", None)
            if custom is graph_mask:
                # Newer route-local backend builds expose a distinct packed
                # destination.  Fall back to the historical alias only for
                # an older vendored backend; never pass the raw BxB storage as
                # a packed mask when the separate buffer exists.
                begin_kwargs["packed_custom_mask"] = (
                    packed_graph_mask if packed_graph_mask is not None else custom
                )
            elif custom is not None:
                begin_kwargs["custom_mask"] = custom
            return begin(*begin_args, **begin_kwargs)

        wrapper.begin_forward = packed_begin
        try:
            return original(self, *args, **kwargs)
        finally:
            wrapper.begin_forward = begin

    FlashInferIndicesUpdaterPrefill.call_begin_forward = wrapped
    setattr(FlashInferIndicesUpdaterPrefill, marker, True)
    _GAM_FLASHINFER_MASK_PATCHED = True
    return True


@contextmanager
def fast_eager_context(enabled: bool | None = None):
    """Suppress eager dLLM diagnostics only while a GAM forward is running.

    The vendored ``ModelRunner._forward_raw`` synchronizes CUDA before and
    after every non-graph dLLM extend and prints a timing line.  A previous
    implementation replaced ``ModelRunner._forward_raw`` at import time.  In
    SGLang's worker-spawn path that class-level monkey-patch could make the
    worker exit before readiness, so it was not a safe optimization boundary.

    This context is deliberately request/forward scoped: no class is modified,
    the original symbols are restored in ``finally``, and the caller wraps only
    the GAM dLLM forward.  It therefore preserves startup and non-GAM routes.
    ``max_running_requests=1`` is required by the launcher for this mode; the
    lock also makes nested/serialized calls deterministic.
    """

    if enabled is None:
        enabled = os.environ.get("GAM_SGLANG_FAST_EAGER", "0") == "1"
    if not enabled:
        yield
        return

    import torch

    with _GAM_FAST_EAGER_LOCK:
        old_sync = torch.cuda.synchronize
        old_print = builtins.print

        def _quiet_print(*values, **print_kwargs):
            if values and isinstance(values[0], str) and values[0].startswith(
                "[PREFILL]"
            ):
                return None
            return old_print(*values, **print_kwargs)

        torch.cuda.synchronize = lambda *args, **kwargs: None
        builtins.print = _quiet_print
        try:
            yield
        finally:
            builtins.print = old_print
            torch.cuda.synchronize = old_sync


def install_sgl_kernel_compat() -> bool:
    try:
        import torch
        import sgl_kernel
    except Exception:
        return False

    # The H800 image's torchao is older than this SGLang fork.  Its model
    # runner imports torchao helpers even when ``--torchao-config`` is empty
    # (the unquantized GAM route).  Supply guarded names so the empty-config
    # early return is reachable; a non-empty quantization request fails
    # explicitly instead of silently changing weights.
    try:
        import torchao.quantization as torchao_q

        def _unsupported_torchao(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("torchao quantization is disabled for GAM SGLang route")

        for name in (
            "float8_dynamic_activation_float8_weight",
            "float8_weight_only",
            "int4_weight_only",
            "int8_dynamic_activation_int8_weight",
            "int8_weight_only",
            "quantize_",
        ):
            if not hasattr(torchao_q, name):
                setattr(torchao_q, name, _unsupported_torchao)
        try:
            import torchao.quantization.observer as observer
            for name in ("PerRow", "PerTensor"):
                if not hasattr(observer, name):
                    setattr(observer, name, _unsupported_torchao)
        except Exception:
            pass
    except Exception:
        pass

    # 0.2.x exposes GGUF kernels at the package root but has no
    # ``sgl_kernel.quantization`` submodule.  The registry imports GGUF
    # unconditionally; install a route-local module containing guarded
    # placeholders so the unquantized path can be discovered.
    if "sgl_kernel.quantization" not in sys.modules:
        quant_mod = types.ModuleType("sgl_kernel.quantization")
        for name in (
            "ggml_dequantize", "ggml_moe_a8", "ggml_moe_a8_vec",
            "ggml_moe_get_block_size", "ggml_mul_mat_a8", "ggml_mul_mat_vec_a8",
        ):
            value = getattr(sgl_kernel, name, None)
            if value is None:
                def _missing(*args, _name=name, **kwargs):
                    del args, kwargs
                    raise RuntimeError(f"optional sgl-kernel symbol {_name} is unavailable")
                value = _missing
            setattr(quant_mod, name, value)
        sys.modules["sgl_kernel.quantization"] = quant_mod

    # Host KV-cache transfer is not used by the default single-device route, but
    # the vendored memory-pool module imports the newer symbol set eagerly.
    # Add guarded attributes to the old Python kvcacheio module instead of
    # pretending that a missing DMA kernel exists.
    try:
        import sgl_kernel.kvcacheio as kvcacheio

        for name in (
            "transfer_kv_all_layer_direct_lf_pf",
            "transfer_kv_all_layer_lf_pf",
            "transfer_kv_all_layer_lf_ph",
            "transfer_kv_all_layer_mla_lf_pf",
            "transfer_kv_direct",
            "transfer_kv_per_layer_direct_pf_lf",
            "transfer_kv_per_layer_mla_pf_lf",
            "transfer_kv_per_layer_pf_lf",
            "transfer_kv_per_layer_ph_lf",
        ):
            if not hasattr(kvcacheio, name):
                def _missing(*args, _name=name, **kwargs):
                    del args, kwargs
                    raise RuntimeError(f"optional KV-cache transfer {_name} is unavailable")
                setattr(kvcacheio, name, _missing)
    except Exception:
        pass

    # SGLang's fp8 helper registers fake implementations at import time.  The
    # ABI-compatible 0.2.x wheel predates those four operator registrations;
    # without schemas the *unrelated* quantization registry aborts before the
    # dense Qwen3 module can load.  Define inert schemas only when absent.  No
    # GAM route selects fp8/quantized weights, and any accidental runtime use
    # still fails at the explicit unsupported-symbol guard below.
    try:
        global _GAM_TORCH_LIB, _GAM_METADATA_PATCHED
        lib = torch.library.Library("sgl_kernel", "FRAGMENT")
        # A torch.library.Library unregisters its operators when garbage
        # collected.  Keep the handle alive for the whole interpreter.
        _GAM_TORCH_LIB = lib
        schemas = (
            "sgl_per_token_group_quant_8bit(Tensor input, Tensor output_q, Tensor output_s, int group_size, float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()",
            "sgl_per_token_group_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, int group_size, float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()",
            "sgl_per_token_quant_fp8(Tensor input, Tensor output_q, Tensor output_s) -> ()",
            "sgl_per_tensor_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, bool is_static) -> ()",
        )
        for schema in schemas:
            try:
                lib.define(schema)
            except Exception:
                pass
    except Exception:
        pass

    # FlashInfer's dLLM wrapper owns one CUDA-graph mask allocation.  Its
    # ``custom_mask_buf`` is the *packed* destination used by ``plan``; the
    # GAM route now writes packed bytes directly in ``GAMSpeculativeBlock``.
    # Earlier versions monkey-patched the FlashInfer class at import time to
    # split raw/packed buffers.  That crossed SGLang's worker-spawn boundary
    # and made the worker exit before readiness (exit 120).  Keep this
    # compatibility layer import-only and route-local; no class monkey-patch
    # is needed for the native packed-buffer contract.

    # The 0.2.9 wheel is the only H800/Torch-2.8 ABI-compatible binary, while
    # this vendored SGLang fork's startup guard asks for >=0.3.20.  Report the
    # audited compatibility level only to that guard; the actual module and
    # binary remain 0.2.9.  No other package name is intercepted.
    if not _GAM_METADATA_PATCHED:
        original_version = _metadata.version

        def _version(name):
            if name.replace("_", "-").lower() == "sgl-kernel":
                return "0.3.20"
            return original_version(name)

        _metadata.version = _version
        _GAM_METADATA_PATCHED = True

    if not hasattr(sgl_kernel, "FusedSetKVBufferArg"):

        @dataclass
        class FusedSetKVBufferArg:
            value: torch.Tensor
            k_buffer: torch.Tensor
            v_buffer: torch.Tensor
            k_scale: Optional[float]
            v_scale: Optional[float]
            cache_loc: torch.Tensor

        sgl_kernel.FusedSetKVBufferArg = FusedSetKVBufferArg

    if not hasattr(sgl_kernel, "apply_rope_with_cos_sin_cache_inplace"):

        def _fallback_rope(
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            head_size: int,
            cos_sin_cache: torch.Tensor,
            is_neox: bool = True,
            fused_set_kv_buffer_arg=None,
            enable_pdl=None,
        ) -> None:
            del enable_pdl
            if cos_sin_cache.dtype != torch.float32:
                raise ValueError("GAM compatibility RoPE expects float32 cache")
            positions = positions.reshape(-1).to(device=cos_sin_cache.device, dtype=torch.long)
            cos_sin = cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)

            def rotate(x: torch.Tensor) -> torch.Tensor:
                shape = x.shape
                if shape[-1] % head_size != 0:
                    raise ValueError(f"unexpected RoPE tensor shape {shape}, head_size={head_size}")
                y = x.reshape(shape[0], -1, head_size)
                dim = cos.shape[-1]
                xr = y[..., :dim].float()
                if is_neox:
                    half = dim // 2
                    a, b = xr[..., :half], xr[..., half:dim]
                    rotated = torch.cat((a * cos[:, None, :] - b * sin[:, None, :],
                                         b * cos[:, None, :] + a * sin[:, None, :]), dim=-1)
                else:
                    a, b = xr[..., ::2], xr[..., 1::2]
                    rotated = torch.stack((a * cos[:, None, :] - b * sin[:, None, :],
                                           b * cos[:, None, :] + a * sin[:, None, :]), dim=-1).flatten(-2)
                out = torch.cat((rotated.to(dtype=x.dtype), y[..., dim:]), dim=-1)
                return out.reshape(shape)

            query.copy_(rotate(query))
            key.copy_(rotate(key))
            arg = fused_set_kv_buffer_arg
            if arg is not None:
                if arg.k_scale is not None or arg.v_scale is not None:
                    raise ValueError("GAM compatibility RoPE does not support KV scales")
                loc = arg.cache_loc.to(device=arg.k_buffer.device, dtype=torch.long)
                arg.k_buffer.index_copy_(0, loc, key.reshape(key.shape[0], -1).to(arg.k_buffer.dtype))
                arg.v_buffer.index_copy_(0, loc, arg.value.reshape(arg.value.shape[0], -1).to(arg.v_buffer.dtype))

        sgl_kernel.apply_rope_with_cos_sin_cache_inplace = _fallback_rope

    previous_getattr = getattr(sgl_kernel, "__getattr__", None)

    def _getattr(name):
        if name in _OPTIONAL_SYMBOLS:
            def _unsupported(*args, **kwargs):
                del args, kwargs
                raise RuntimeError(
                    f"optional sgl-kernel symbol {name} is unavailable in the "
                    "H800 image's ABI-compatible wheel; dense GAM Qwen3 must "
                    "not select this path"
                )

            _unsupported.__name__ = name
            return _unsupported
        if previous_getattr is not None:
            return previous_getattr(name)
        raise AttributeError(name)

    # Python's ``from sgl_kernel import symbol`` consults module __getattr__.
    # Keep this limited to optional symbols; a typo in the dense path remains a
    # normal AttributeError and is caught by the capability gate.
    sgl_kernel.__getattr__ = _getattr

    # The vendored online W8A8 loader stores per-output-channel weight scales
    # as [1, N], while its short-row Triton fallback validates [N, 1].  Both
    # layouts are the same contiguous N scalars and the CUTLASS path accepts
    # the former, but B32 diffusion requests deliberately select Triton
    # because CUTLASS TMA requires at least 64 rows.  Normalize only this
    # unambiguous [1, N] case at the route boundary; weights, scale values and
    # quantization math are unchanged.
    try:
        global _GAM_FP8_SCALE_PATCHED
        from sglang.srt.layers.quantization import fp8_utils

        if not _GAM_FP8_SCALE_PATCHED:
            original_triton_scaled_mm = fp8_utils.triton_scaled_mm

            def _triton_scaled_mm_scale_compat(
                input,
                weight,
                scale_a,
                scale_b,
                *args,
                **kwargs,
            ):
                if (
                    getattr(scale_b, "dim", lambda: 0)() == 2
                    and scale_b.shape[0] == 1
                    and scale_b.shape[1] == weight.shape[1]
                ):
                    scale_b = scale_b.transpose(0, 1).contiguous()
                return original_triton_scaled_mm(
                    input, weight, scale_a, scale_b, *args, **kwargs
                )

            fp8_utils.triton_scaled_mm = _triton_scaled_mm_scale_compat
            _GAM_FP8_SCALE_PATCHED = True
    except Exception:
        # Unquantized capability probes need not import the FP8 registry.  A
        # real W8A8 request remains fail-closed and will surface its original
        # kernel error if this optional module cannot be loaded.
        pass

    return hasattr(sgl_kernel, "FusedSetKVBufferArg") and hasattr(
        sgl_kernel, "apply_rope_with_cos_sin_cache_inplace"
    )
