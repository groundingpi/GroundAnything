#!/usr/bin/env python3
"""Install the Qwen3.5 GatedDeltaNet convolution fast path explicitly.

Transformers 5.2 only discovers the standalone ``causal-conv1d`` package,
while flash-linear-attention 0.5 already ships a Triton implementation.  The
formal GAM runtime selects the FLA implementation through an explicit
environment contract and fails before model construction if it is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F


BACKEND_ENV = "GAM_QWEN35_FASTPATH_BACKEND"
FLA_TRITON_BACKEND = "fla_triton"
STABLE_L2NORM_ENV = "GAM_QWEN35_STABLE_L2NORM"
STABLE_CAUSAL_CONV_ENV = "GAM_QWEN35_STABLE_CAUSAL_CONV"
# Exact ``inspect.getsource`` digest from the immutable Transformers 5.2.0
# runtime shared by the formal PPU/H800 launchers.  The optimization below
# deliberately fails closed if the upstream method changes.  Do not reuse the
# old 5.2.0.dev0 image digest here: the launchers pin the released 5.2.0 wheel.
VISION_ROT_POS_EMB_SOURCE_SHA256 = (
    "c029defdbb1e283ac5cce391d82d8d01cac20d730d2171c7b99557468b39424a"
)
GATED_DELTA_FORWARD_SOURCE_SHA256 = (
    "a72326938da48c9d198e4babdb75f6f8be8f89153ce098bcf73054f1b9b01d13"
)


class Qwen35FastPathError(RuntimeError):
    """Raised when the requested fast path cannot be installed exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Qwen35FastPathError(message)


def qwen35_stable_l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mathematically equivalent fallback for the PPU-unstable FLA kernel."""

    _require(x.ndim >= 1 and x.is_floating_point(), "L2Norm 输入必须是浮点 tensor")
    x_float = x.float()
    rstd = torch.rsqrt(torch.sum(x_float * x_float, dim=-1) + eps)
    normalized = x_float * rstd.unsqueeze(-1)
    return normalized.to(dtype=output_dtype or x.dtype), rstd


def qwen35_stable_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    **kwargs,
):
    """FLA-compatible eager prefill convolution for unstable PPU shapes.

    Transformers supplies ``x`` as [B,C,T], while FLA defines the depthwise
    kernel over [B,T,C].  FLA accumulates in FP32 and uses state slot zero only
    as history padding; this implementation preserves those exact semantics
    without launching FLA's shape-specialized Triton prefill kernel.
    """

    _require(seq_idx is None, "稳定 causal_conv1d 不支持非空 seq_idx")
    _require(not kwargs, f"稳定 causal_conv1d 收到未知参数: {sorted(kwargs)}")
    _require(x.ndim == 3 and x.is_floating_point(),
             "稳定 causal_conv1d 输入必须为浮点 [B,C,T]")
    _require(weight.ndim == 2 and weight.is_floating_point(),
             "稳定 causal_conv1d weight 必须为浮点 [C,W]")
    batch_size, channels, seq_len = x.shape
    _require(weight.shape[0] == channels and weight.shape[1] > 0,
             "稳定 causal_conv1d weight shape 与输入不匹配")
    kernel_size = weight.shape[1]
    if bias is not None:
        _require(bias.shape == (channels,), "稳定 causal_conv1d bias shape 不匹配")
    if initial_state is None:
        state = x.new_zeros((batch_size, channels, kernel_size))
    else:
        _require(
            initial_state.shape == (batch_size, channels, kernel_size),
            "稳定 causal_conv1d initial_state shape 不匹配",
        )
        _require(initial_state.device == x.device and initial_state.dtype == x.dtype,
                 "稳定 causal_conv1d initial_state device/dtype 不匹配")
        state = initial_state

    history = torch.cat((state, x), dim=-1)
    output = torch.zeros((batch_size, channels, seq_len), device=x.device, dtype=torch.float32)
    weight_float = weight.float()
    for offset in range(kernel_size):
        values = history[..., offset + 1:offset + 1 + seq_len].float()
        output.add_(values * weight_float[:, offset].view(1, channels, 1))
    if bias is not None:
        output.add_(bias.float().view(1, channels, 1))
    if activation in ("silu", "swish"):
        output = F.silu(output)
    else:
        _require(activation is None, f"稳定 causal_conv1d 不支持 activation={activation!r}")

    output = output.to(dtype=x.dtype)
    if output_final_state:
        return output, history[..., -kernel_size:].contiguous()
    return output


def _install_stable_l2norm() -> dict[str, Any]:
    enabled = os.environ.get(STABLE_L2NORM_ENV, "0") == "1"
    if not enabled:
        return {"gated_delta_l2norm": "fla_triton"}
    chunk_module = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    original = getattr(chunk_module, "l2norm_fwd", None)
    _require(callable(original), "FLA gated-delta chunk 缺少 l2norm_fwd")
    chunk_module.l2norm_fwd = qwen35_stable_l2norm_fwd
    return {
        "gated_delta_l2norm": qwen35_stable_l2norm_fwd.__name__,
        "gated_delta_l2norm_original_module": getattr(original, "__module__", None),
    }


def _install_fla_triton(
    modeling: Any,
    fla_causal_conv1d: Callable[..., Any],
    fla_causal_conv1d_update: Callable[..., Any],
) -> dict[str, Any]:
    """Bind FLA's signatures to the interface expected by Transformers."""

    _require(callable(fla_causal_conv1d), "FLA causal_conv1d 不可调用")
    _require(callable(fla_causal_conv1d_update), "FLA causal_conv1d_update 不可调用")

    def qwen35_fla_causal_conv1d_fn(
        x,
        weight,
        bias=None,
        activation=None,
        seq_idx=None,
        initial_state=None,
        output_final_state=False,
        **kwargs,
    ):
        _require(seq_idx is None, "FLA Triton training fast path 不支持非空 seq_idx")
        _require(not kwargs, f"Qwen3.5 causal_conv1d 收到未知参数: {sorted(kwargs)}")
        _require(getattr(x, "ndim", None) == 3,
                 "Transformers Qwen3.5 causal_conv1d 输入必须为 [B,C,T]")
        # Transformers/causal-conv1d uses [B,C,T]; FLA Triton uses [B,T,D].
        result = fla_causal_conv1d(
            x=x.transpose(1, 2),
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            backend="triton",
        )
        if isinstance(result, tuple):
            _require(len(result) == 2,
                     "FLA causal_conv1d tuple 返回值必须为 (output, final_state)")
            output, final_state = result
            if output_final_state:
                _require(isinstance(final_state, torch.Tensor),
                         "请求 output_final_state 时 FLA 必须返回 convolution state")
            else:
                _require(final_state is None,
                         "未请求 output_final_state 时 FLA 不应返回 convolution state")
            result = output
        elif output_final_state:
            raise Qwen35FastPathError("请求 output_final_state 时 FLA 必须返回 tuple")
        _require(getattr(result, "ndim", None) == 3,
                 "FLA causal_conv1d 输出必须为 [B,T,D]")
        output = result.transpose(1, 2)
        if output_final_state:
            return output, final_state
        return output

    def qwen35_fla_causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
    ):
        _require(getattr(x, "ndim", None) == 3,
                 "Transformers Qwen3.5 causal_conv1d_update 输入必须为 [B,C,T]")
        result = fla_causal_conv1d_update(
            x=x.transpose(1, 2),
            cache=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        if isinstance(result, tuple):
            _require(len(result) == 2,
                     "FLA causal_conv1d_update tuple 返回值必须为 (output, cache)")
            output, returned_cache = result
            _require(isinstance(returned_cache, torch.Tensor),
                     "FLA causal_conv1d_update 返回 cache 必须是 tensor")
            _require(returned_cache.shape == conv_state.shape,
                     "FLA causal_conv1d_update 返回 cache shape 漂移")
            _require(returned_cache.dtype == conv_state.dtype,
                     "FLA causal_conv1d_update 返回 cache dtype 漂移")
            _require(returned_cache.device == conv_state.device,
                     "FLA causal_conv1d_update 返回 cache device 漂移")
            # FLA's input_guard may make a contiguous cache tensor and return
            # that updated copy.  Commit it explicitly into Transformers'
            # persistent Qwen3.5 cache instead of relying on Python identity.
            if returned_cache.data_ptr() != conv_state.data_ptr():
                conv_state.copy_(returned_cache)
            result = output
        _require(getattr(result, "ndim", None) == 3,
                 "FLA causal_conv1d_update 输出必须为 [B,T,D]")
        return result.transpose(1, 2)

    stable_causal_conv = os.environ.get(STABLE_CAUSAL_CONV_ENV, "0") == "1"
    modeling.causal_conv1d_fn = (
        qwen35_stable_causal_conv1d_fn if stable_causal_conv
        else qwen35_fla_causal_conv1d_fn
    )
    modeling.causal_conv1d_update = qwen35_fla_causal_conv1d_update
    modeling.is_fast_path_available = all(
        (
            modeling.causal_conv1d_fn,
            modeling.causal_conv1d_update,
            modeling.chunk_gated_delta_rule,
            modeling.fused_recurrent_gated_delta_rule,
        )
    )
    _require(modeling.is_fast_path_available is True,
             "Qwen3.5 GatedDeltaNet fast path 未完整启用")
    return {
        "backend": FLA_TRITON_BACKEND,
        "fast_path_available": True,
        "causal_conv1d_fn": modeling.causal_conv1d_fn.__name__,
        "causal_conv1d_update": qwen35_fla_causal_conv1d_update.__name__,
        "fla_conv_module": getattr(fla_causal_conv1d, "__module__", None),
        "fla_update_module": getattr(fla_causal_conv1d_update, "__module__", None),
    }


def _install_cached_chunk_extension(modeling: Any) -> dict[str, Any]:
    """Enable exact cached multi-token extension for Qwen3.5 GatedDeltaNet.

    The pinned Transformers implementation uses the previous recurrent state
    only when ``seq_len == 1``. Speculative block verification needs a short
    causal chunk to extend the same hybrid cache. This guarded patch preserves
    the native path for prefill and single-token decode and adds only the
    missing ``cache + seq_len > 1`` branch.
    """

    gated_delta_cls = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    _require(gated_delta_cls is not None, "Transformers 缺少 Qwen3_5GatedDeltaNet")
    original = getattr(gated_delta_cls, "forward", None)
    _require(callable(original), "Qwen3_5GatedDeltaNet.forward 不可调用")
    source = inspect.getsource(original)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    _require(
        source_sha256 == GATED_DELTA_FORWARD_SOURCE_SHA256,
        "Qwen3_5GatedDeltaNet.forward 实现漂移: "
        f"{source_sha256} != {GATED_DELTA_FORWARD_SOURCE_SHA256}",
    )

    def qwen35_cached_chunk_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if (
            cache_params is None
            or not cache_params.has_previous_state
            or hidden_states.shape[1] <= 1
        ):
            return original(
                self,
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        hidden_states = modeling.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]
        _require(conv_state is not None and recurrent_state is not None,
                 "Qwen3.5 cached chunk 缺少 linear-attention state")

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv, final_conv_state = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            initial_state=conv_state,
            output_final_state=True,
        )
        cache_params.conv_states[self.layer_idx] = final_conv_state
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    gated_delta_cls.forward = qwen35_cached_chunk_forward
    return {
        "gated_delta_cached_chunk": qwen35_cached_chunk_forward.__name__,
        "gated_delta_forward_original_sha256": source_sha256,
    }


def _install_vision_rot_pos_emb_patch(modeling: Any) -> dict[str, Any]:
    """Avoid shape-specialized Jiterator reductions in ViT RoPE setup.

    The pinned PPU PyTorch compiles ``torch.prod(grid_thw, dim=1)`` through
    ptxas.  Its cache key includes the small ``grid_thw`` shape, so a large
    distributed job repeatedly stalls on previously unseen image-count shapes.
    Moving this tiny integer shape tensor to Python replaces two synchronizing
    GPU reductions with one small device-to-host copy.  Position IDs and
    embeddings remain bit-identical.
    """

    vision_model_cls = getattr(modeling, "Qwen3_5VisionModel", None)
    _require(vision_model_cls is not None, "Transformers 缺少 Qwen3_5VisionModel")
    original = getattr(vision_model_cls, "rot_pos_emb", None)
    _require(callable(original), "Qwen3_5VisionModel.rot_pos_emb 不可调用")
    source = inspect.getsource(original)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    _require(
        source_sha256 == VISION_ROT_POS_EMB_SOURCE_SHA256,
        "Qwen3_5VisionModel.rot_pos_emb 实现漂移: "
        f"{source_sha256} != {VISION_ROT_POS_EMB_SOURCE_SHA256}",
    )

    def qwen35_vision_rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        # ``grid_thw`` contains only integer image/video grid metadata.  The
        # original implementation immediately reduces it to Python scalars via
        # two .item() calls; one tolist() preserves those semantics and avoids
        # dynamic PPU reduction kernels entirely.
        grid_shape = grid_thw.detach().cpu().tolist()
        max_hw = max(max(int(height), int(width)) for _, height, width in grid_shape)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(
            int(num_frames) * int(height) * int(width)
            for num_frames, height, width in grid_shape
        )
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_shape:
            num_frames = int(num_frames)
            height = int(height)
            width = int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    vision_model_cls.rot_pos_emb = qwen35_vision_rot_pos_emb
    return {
        "vision_rot_pos_emb": qwen35_vision_rot_pos_emb.__name__,
        "vision_rot_pos_emb_original_sha256": source_sha256,
        "vision_grid_reduction": "python_int",
    }


def install_qwen35_fastpath() -> dict[str, Any]:
    backend = os.environ.get(BACKEND_ENV)
    _require(backend == FLA_TRITON_BACKEND,
             f"{BACKEND_ENV} 必须显式为 {FLA_TRITON_BACKEND!r}，实际为 {backend!r}")

    try:
        from fla.modules.convolution import (
            causal_conv1d as fla_causal_conv1d,
            causal_conv1d_update as fla_causal_conv1d_update,
        )
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    except Exception as exc:  # pragma: no cover
        raise Qwen35FastPathError(f"无法导入 Qwen3.5/FLA fast path: {exc}") from exc

    result = _install_fla_triton(
        modeling,
        fla_causal_conv1d,
        fla_causal_conv1d_update,
    )
    result.update(_install_stable_l2norm())
    result.update(_install_cached_chunk_extension(modeling))
    result.update(_install_vision_rot_pos_emb_patch(modeling))
    result.update({
        "transformers_modeling": inspect.getsourcefile(modeling),
        "fla_version": __import__("fla").__version__,
    })
    print("[GAM Qwen3.5 fast path] " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="install in this process and report provenance")
    args = parser.parse_args()
    _require(args.check, "仅支持 --check")
    install_qwen35_fastpath()


if __name__ == "__main__":
    main()
