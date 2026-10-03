"""SGLang model implementation for the GAM GroundAnythingVLM/Qwen3 DLM route.

This is an inference-only adapter.  The language backbone is SGLang's native
Qwen3 implementation; ``HierarchyBlock`` from the Fast-dLLM fork switches its
RadixAttention between causal prefill/KV writes and encoder-only denoising.
The Kimi-K3 MoonViT and GroundAnythingVLM projector are loaded from the immutable model
code shipped with the checkpoint, so their parameter names and numerical
operations remain identical to the training route.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import MethodType
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from .execution_ops import cached_extend_positions

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


def _load_k3_module():
    """Load the checkpoint's vision reference module without global mutation."""

    root = os.environ.get("GAM_SGLANG_MODEL_CODE_PATH")
    if not root:
        raise RuntimeError(
            "GAM_SGLANG_MODEL_CODE_PATH is required; refusing to substitute a "
            "different vision encoder"
        )
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # The original K3 file conditionally imports FlashAttention at module
    # import time.  A task-local Torch ABI upgrade can intentionally select
    # its exact built-in eager attention implementation while a matching FA2
    # wheel is compiled; hide only the availability probe for that import.
    # The checkpoint source and weights remain immutable, and the default FA2
    # route is unchanged.
    vision_attention = os.environ.get(
        "GAM_SGLANG_VISION_ATTN", "flash_attention_2"
    )
    restore_flash_probe = None
    if vision_attention == "eager":
        import transformers.utils as transformers_utils

        restore_flash_probe = transformers_utils.is_flash_attn_2_available
        transformers_utils.is_flash_attn_2_available = lambda: False
    try:
        return importlib.import_module("modeling_groundinganything_vision")
    except Exception as exc:  # pragma: no cover - depends on image ABI
        raise RuntimeError(
            f"cannot import immutable Kimi-K3 vision implementation from {root}: {exc}"
        ) from exc
    finally:
        if restore_flash_probe is not None:
            transformers_utils.is_flash_attn_2_available = restore_flash_probe


class GroundAnythingVLMTwoLayerProjector(nn.Module):
    """Exact K3 2x2 merger/projector used by ``modeling_groundinganything``."""

    def __init__(self, config):
        super().__init__()
        merge_area = int(config.spatial_merge_size) ** 2
        input_size = int(config.hidden_size) * merge_area
        hidden_size = int(config.projector_hidden_size)
        output_size = int(config.out_hidden_size)
        self.input_size = input_size
        self.pre_norm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.projector_ln_eps)
        )
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.post_norm = nn.RMSNorm(output_size, eps=float(config.projector_ln_eps))

    def forward(self, features):
        outputs = []
        for item in features:
            if item.ndim != 3 or item.shape[1] * item.shape[2] != self.input_size:
                raise ValueError(
                    "Kimi-K3 merged features must be [tokens, 4, 1024], got "
                    f"{tuple(item.shape)}"
                )
            item = self.pre_norm(item).reshape(item.shape[0], self.input_size)
            outputs.append(self.post_norm(self.fc2(self.act(self.fc1(item)))))
        return outputs


class GroundAnythingVLMVisionModel(nn.Module):
    """Kimi-K3 MoonViT3D plus the trained GroundAnythingVLM connector."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        # The original class is imported from the checkpoint rather than
        # copied/rewritten.  This preserves divided-fixed temporal/spatial
        # positions and the qkv_hidden_size=1536 projection exactly.
        k3 = _load_k3_module()
        config._attn_implementation = os.environ.get(
            "GAM_SGLANG_VISION_ATTN", "flash_attention_2"
        )
        self.vision_tower = k3.MoonViT3dPretrainedModel(config)
        self.projector = GroundAnythingVLMTwoLayerProjector(config)

    @property
    def dtype(self):
        return self.vision_tower.patch_embed.proj.weight.dtype

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        del kwargs
        if grid_thw is None:
            raise ValueError("image_grid_thw is required for Kimi-K3 MoonViT3D")
        features = self.vision_tower(pixel_values.to(self.dtype), grid_thw)
        return torch.cat(self.projector(features), dim=0)


class FastDVLMForConditionalGeneration(nn.Module):
    """Fast-dVLM-compatible SGLang wrapper for GAM Qwen3 DLM."""

    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    default_bitsandbytes_target_modules = [
        ".gate_proj.", ".down_proj.", ".up_proj.",
        ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
    ]

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if os.environ.get("GAM_SGLANG_TRITON_VERIFY_GRAPH", "0") == "1":
            from .triton_verify_graph import install_triton_verify_graph_patch
            install_triton_verify_graph_patch()
        self.pp_group = get_pp_group()
        self.config = config
        text_config = config.text_config
        # Qwen3 checkpoints produced by current Transformers store RoPE under
        # ``rope_parameters``.  The vendored Fast-dLLM Qwen3 module predates
        # that schema and only reads the legacy top-level fields.  If we leave
        # the fields absent it silently falls back to theta=1e6, while the
        # Stage-I Kimi/GroundAnythingVLM checkpoint was trained with theta=5e6.  That
        # changes causal logits before DLM scheduling even starts.  Normalize
        # the schema at this route boundary; it is inference-only and does not
        # mutate the checkpoint or the training implementation.
        rope_parameters = getattr(text_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            if getattr(text_config, "rope_theta", None) is None:
                rope_theta = rope_parameters.get("rope_theta")
                if rope_theta is not None:
                    text_config.rope_theta = float(rope_theta)
            if getattr(text_config, "rope_scaling", None) is None:
                rope_type = rope_parameters.get("rope_type")
                if rope_type not in (None, "default"):
                    text_config.rope_scaling = {
                        key: value
                        for key, value in rope_parameters.items()
                        if key != "rope_theta"
                    }
        # The wrapper checkpoint contains one appended atomic mask row.  The
        # row must exist in the input embedding table, but it is never a
        # candidate output token.  Keep the two contracts separate: SGLang's
        # CUDA-graph logits buffer must use the original training vocabulary
        # (152670), while Qwen3's embedding/lm-head must still accept the mask
        # id (152670) in the 152671-row wrapper.  Previously both values were
        # overwritten with 152671; graph capture then copied a 152671-wide
        # logits tensor into a 152670-wide buffer and failed before inference.
        vocab_size = int(os.environ.get("GAM_SGLANG_VOCAB_SIZE", text_config.vocab_size))
        logits_vocab_size = int(
            os.environ.get("GAM_SGLANG_LOGITS_VOCAB_SIZE", text_config.vocab_size)
        )
        if logits_vocab_size <= 0 or logits_vocab_size > vocab_size:
            raise ValueError(
                f"invalid vocab split: embedding={vocab_size}, logits={logits_vocab_size}"
            )
        text_config.vocab_size = vocab_size
        config.vocab_size = logits_vocab_size
        text_config.tie_word_embeddings = False

        self.model = Qwen3Model(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self._install_optional_dlm_qkv_dump()
        # Keep the vendor's normal ``save_kv_cache=True`` behavior during
        # denoising.  Triton/torch-native EXTEND backends read the current
        # block's K/V back from the cache after writing it; forcing
        # ``save_kv_cache=False`` therefore makes them attend to uninitialized
        # or stale slots, rather than providing a read-only view.  The
        # Fast-dLLM reference follows the write-on-every-denoise contract.
        # The old no-KV wrapper remains an explicit opt-in diagnostic only.
        if os.environ.get("GAM_SGLANG_ENABLE_DLM_KV_POLICY") == "1":
            self._install_dllm_kv_policy()
        # The image's ABI-compatible sgl-kernel 0.2.9 fused RoPE is correct
        # for ordinary EXTEND/ DLLM_EXTEND batches, but its in-place kernel
        # assumes the contiguous single-stream position layout.  SGLang's
        # multimodal dLLM prompt prefill carries a different ForwardBatch
        # layout (even at batch size one), and the old kernel can report an
        # illegal access.  Save the normal dispatch target so the multimodal
        # path can temporarily use the numerically equivalent PyTorch RoPE
        # while retaining the fused kernel for all denoising/text tokens.
        self._gam_rope_dispatch = []
        for layer in self.model.layers:
            rope = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
            if rope is not None and hasattr(rope, "_forward_method"):
                self._gam_rope_dispatch.append((rope, rope._forward_method))
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.visual = GroundAnythingVLMVisionModel(config.vision_config)
        # Qwen3 uses ordinary 1-D RoPE.  The processor still returns the
        # standard multimodal item contract, but no 3-D mRoPE is applied.
        self.is_mrope_enabled = False
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def _install_dllm_kv_policy(self) -> None:
        """Prevent encoder-only denoise forwards from committing KV state.

        SGLang's ``RadixAttention.forward`` exposes ``save_kv_cache`` as a
        keyword with a default of ``True``.  Fast-dLLM's algorithm calls the
        model repeatedly while changing token ids inside one block; writing
        each intermediate k/v tensor makes the next iteration attend to stale
        versions of the same positions on backends whose dLLM-specialized
        ragged path is unavailable.  We therefore force the flag off only
        when both conditions hold:

        * the scheduler is in ``DLLM_EXTEND``; and
        * the layer is in ``ENCODER_ONLY`` denoising mode.

        The causal prefill and the explicit final ``forward_extend`` remain
        unchanged and continue to populate the KV cache.  Set
        ``GAM_SGLANG_DISABLE_DLM_KV_POLICY=1`` only for an intentional
        diagnostic A/B run.
        """
        if os.environ.get("GAM_SGLANG_DISABLE_DLM_KV_POLICY") == "1":
            return
        for layer in self.model.layers:
            attention = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attention is None or getattr(attention, "_gam_kv_policy", False):
                continue
            original_forward = attention.forward

            def _forward(
                attn_self,
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=True,
                _original=original_forward,
                **kwargs,
            ):
                mode = getattr(forward_batch, "forward_mode", None)
                is_dllm = bool(mode is not None and mode.is_dllm_extend())
                if is_dllm and attn_self.attn_type == AttentionType.ENCODER_ONLY:
                    save_kv_cache = False
                return _original(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )

            attention.forward = MethodType(_forward, attention)
            attention._gam_kv_policy = True

    def _install_optional_dlm_qkv_dump(self) -> None:
        """Capture layer-0 DLM Q/K/V once for an offline semantic audit.

        This is deliberately installed only when an explicit path is given;
        normal serving has no wrapper or host copy overhead.  Comparing the
        pre-attention tensors against the Transformers oracle distinguishes a
        weight/position issue from an attention-backend/cache issue.
        """
        path = os.environ.get("GAM_SGLANG_DUMP_DLM_QKV_PATH")
        if not path or not self.model.layers:
            return
        attention = self.model.layers[0].self_attn
        original = attention.forward_prepare_native

        def _dumping_prepare(attn_self, positions, hidden_states, _original=original):
            q, k, v = _original(positions, hidden_states)
            batch = getattr(attn_self, "_gam_qkv_forward_batch", None)
            is_dllm = bool(batch is not None and batch.forward_mode.is_dllm_extend())
            # SGLang executes a tiny internal dLLM dummy request while the
            # worker is initialized even with ``--skip-server-warmup``.  Its
            # one-token prefix is not the user's prompt and used to consume
            # this one-shot diagnostic, producing a misleading parity file.
            # Gate on an explicit minimum cached prefix so the artifact comes
            # from the real request.  This path is diagnostic-only.
            min_prefix = int(os.environ.get("GAM_SGLANG_DUMP_MIN_PREFIX", "2"))
            prefix_lens = getattr(batch, "extend_prefix_lens_cpu", None)
            prefix_len = int(prefix_lens[0]) if prefix_lens else 0
            if (
                is_dllm
                and prefix_len >= min_prefix
                and not getattr(attn_self, "_gam_qkv_dumped", False)
            ):
                import torch as _torch

                # Recompute only layer-0's pre-RoPE Q/K for the audit.  The
                # normal path above remains the sole producer of returned
                # tensors; this duplicate is gated by an explicit diagnostic
                # path and is never installed in ordinary serving.
                qkv, _ = attn_self.qkv_proj(hidden_states)
                q_size = attn_self.q_size
                kv_size = attn_self.kv_size
                q_pre, k_pre, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_pre, k_pre = attn_self._apply_qk_norm(q_pre, k_pre)

                _torch.save(
                    {
                        "positions": positions.detach().cpu(),
                        "input_ids": batch.input_ids.detach().cpu(),
                        "extend_prefix_len": prefix_len,
                        "hidden_states": hidden_states.detach().float().cpu(),
                        "q_pre": q_pre.detach().float().cpu(),
                        "k_pre": k_pre.detach().float().cpu(),
                        "q": q.detach().float().cpu(),
                        "k": k.detach().float().cpu(),
                        "v": v.detach().float().cpu(),
                    },
                    path,
                )
                attn_self._gam_qkv_dumped = True
            return q, k, v

        attention.forward_prepare_native = MethodType(_dumping_prepare, attention)
        original_forward = attention.forward

        def _remember_batch(attn_self, positions, hidden_states, forward_batch, _original=original_forward):
            attn_self._gam_qkv_forward_batch = forward_batch
            return _original(positions, hidden_states, forward_batch)

        attention.forward = MethodType(_remember_batch, attention)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_input_embedding(self, input_ids: torch.Tensor):
        return self.model.get_input_embedding(input_ids)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        features = [item.feature for item in items]
        if not features:
            raise ValueError("image item has no feature")
        pixel_values = torch.cat(features, dim=0)
        # A precomputed feature is already in the language dimension.  This
        # branch is useful for SGLang's multimodal feature cache.
        if pixel_values.ndim == 2 and pixel_values.shape[-1] == self.config.hidden_size:
            return pixel_values
        grid = torch.cat([item.image_grid_thw for item in items], dim=0)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid.tolist(), rope_type="rope_3d"
            )
        return self.visual(pixel_values, grid)

    @property
    def use_data_parallel(self) -> bool:
        args = get_global_server_args()
        return bool(args and getattr(args, "mm_enable_dp_encoder", False))

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        raise RuntimeError("GAM Kimi-K3 SGLang route currently supports images only")

    def post_process(self, inputs_embeds, modalities, embeddings, indices, forward_batch):
        return [e for e, i in zip(embeddings, indices) if e is not None and i is not None], forward_batch

    def _set_language_attention(self, attn_type: AttentionType) -> None:
        """Set every Qwen3 layer's mutable RadixAttention mode.

        Fast-dLLM changes this field during block denoising.  A later
        multimodal EXTEND request must always enter with causal attention:
        its image embeddings are being written into a causal KV cache, not
        into an encoder-only block.  Relying solely on the algorithm's
        end-of-block cleanup is unsafe because a warmup/aborted request can
        leave the process-wide module state in ENCODER_ONLY mode.
        """
        for layer in self.model.layers:
            attention = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attention is not None:
                attention.attn_type = attn_type

    def _set_mm_rope_native(self, enabled: bool) -> None:
        """Use native RoPE only for the old-kernel-sensitive MM prefill."""
        for rope, dispatch in self._gam_rope_dispatch:
            rope._forward_method = rope.forward_native if enabled else dispatch

    def _set_dlm_rope_native(self, enabled: bool) -> None:
        """Select the numerically auditable RoPE path for DLM blocks.

        The H800 image's fused ``sgl-kernel`` RoPE is fast, but it is not
        automatically equivalent to the Transformers Qwen3 reference for
        every ragged ``DLLM_EXTEND`` layout.  Keep the fused dispatch as the
        default and make the native PyTorch implementation an explicit,
        fail-closed semantic-A/B switch.  This is inference-only; it never
        changes training tensors or checkpoint weights.
        """
        for rope, dispatch in self._gam_rope_dispatch:
            rope._forward_method = rope.forward_native if enabled else dispatch

    @staticmethod
    def _align_extend_positions(
        positions: torch.Tensor, forward_batch: ForwardBatch, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Remove dLLM's fixed-B32 padding from prompt-prefill positions.

        The vendored Fast-dLLM ``ForwardBatch`` overwrites positions with a
        block-sized vector whenever a dLLM config is present, including the
        first multimodal prompt prefill.  In fact, for a 374-token image
        prompt it can provide only the first 32 positions.  Reconstruct the
        ordinary EXTEND positions from the authoritative prefix/extend lens;
        this also handles a later chunk whose prefix is already cached.
        """
        # The route-local Triton graph capture uses a synthetic ForwardBatch.
        # Its position tensor is already the stable graph input buffer and is
        # overwritten with absolute positions before every replay.  Calling
        # ``.tolist()`` on its CUDA prefix tensors while the stream is being
        # captured is illegal (and unnecessary), so keep this dummy path
        # entirely device-side.  Real eager requests never carry this marker.
        if getattr(forward_batch, "_gam_triton_dllm_graph_capture", False):
            target = int(input_ids.numel())
            flat = positions.reshape(-1)
            if flat.numel() < target:
                raise RuntimeError(
                    "GAM Triton graph capture has fewer positions than tokens: "
                    f"positions={flat.numel()} tokens={target}"
                )
            return flat[:target].contiguous()

        if os.environ.get("GAM_SGLANG_POSITION_CACHE", "0") == "1":
            cached = cached_extend_positions(positions, forward_batch, input_ids)
            if cached is not None:
                return cached

        prefixes = getattr(forward_batch, "extend_prefix_lens", None)
        lengths = getattr(forward_batch, "extend_seq_lens", None)
        if prefixes is not None and lengths is not None:
            prefixes = prefixes.reshape(-1).tolist()
            lengths = lengths.reshape(-1).tolist()
            if len(prefixes) == len(lengths) and prefixes:
                pieces = [
                    torch.arange(
                        int(prefix),
                        int(prefix) + int(length),
                        device=positions.device,
                        dtype=positions.dtype,
                    )
                    for prefix, length in zip(prefixes, lengths)
                ]
                rebuilt = torch.cat(pieces) if pieces else positions[:0]
                target = getattr(forward_batch, "num_token_non_padded_cpu", None)
                if target is None or rebuilt.numel() == int(target):
                    return rebuilt.contiguous()

        target = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if target is None:
            target = int(input_ids.numel())
        target = int(target)
        flat = positions.reshape(-1)
        if flat.numel() == target:
            return flat
        if flat.numel() > target:
            return flat[:target].contiguous()
        raise RuntimeError(
            "GAM Qwen3 received fewer RoPE positions than non-padded tokens: "
            f"positions={flat.numel()} tokens={target}"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if forward_batch.forward_mode.is_dllm_extend():
            # Image embeddings are already in the prefill KV cache.  Denoising
            # blocks therefore run through the pure Qwen3 language model.
            #
            # The vendored dLLM scheduler exposes a local B32 position vector
            # (0..31) even when the block follows a much longer cached prompt.
            # Qwen3 uses ordinary absolute 1-D RoPE, so feeding that local
            # vector silently rotates every DLM block at the wrong phase.  The
            # authoritative prefix/extend lengths are available on the
            # ForwardBatch; rebuild absolute positions for *all* dLLM blocks,
            # including the first [AR]+31-mask block.  This is inference-side
            # bookkeeping only and leaves the training route untouched.
            positions = self._align_extend_positions(positions, forward_batch, input_ids)
            # Keep the batch metadata consistent with the tensor passed into
            # Qwen3 as well.  The vendored scheduler's dLLM override leaves a
            # local 0..B32-1 vector here; the hierarchy algorithm uses this
            # field for request-boundary/inheritance bookkeeping and a few
            # attention backends consume it for draft metadata.  A mixed
            # absolute/local pair can therefore re-enter the next block with
            # a stale cache view even though RoPE itself received the right
            # positions.  This mutation is scoped to the inference request's
            # ForwardBatch and never touches checkpoint or training tensors.
            forward_batch.positions = positions.to(
                device=forward_batch.positions.device,
                dtype=forward_batch.positions.dtype,
            )
            self._set_dlm_rope_native(
                os.environ.get("GAM_SGLANG_DLM_ROPE_NATIVE") == "1"
            )
            hidden_states = self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            # Request-boundary invariant: prompt/image prefill and ordinary
            # decode are always causal, even if the previous DLM block was
            # interrupted during denoising or server warmup.  The DLM
            # algorithm switches to ENCODER_ONLY only inside its own run().
            self._set_language_attention(AttentionType.DECODER)
            positions = self._align_extend_positions(positions, forward_batch, input_ids)
            is_mm_prefill = bool(
                getattr(forward_batch, "contains_mm_inputs", lambda: False)()
            )
            # Keep the workaround scoped to the actual image embedding pass;
            # text-only DLM blocks continue to use the fused H800 kernel.
            self._set_mm_rope_native(is_mm_prefill)
            try:
                hidden_states = general_mm_embed_routine(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    language_model=self.model,
                    multimodal_model=self,
                    positions=positions,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            finally:
                self._set_mm_rope_native(False)
        if not self.pp_group.is_last_rank:
            return hidden_states
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        output = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        # Optional causal-prefill parity artifact.  SGLang normally returns
        # only the last-token logits for EXTEND; retaining one CPU snapshot
        # lets us distinguish a prompt/cache mismatch from a DLM mask mismatch
        # without changing the production route.
        dump_path = os.environ.get("GAM_SGLANG_DUMP_PREFILL_PATH")
        if (
            dump_path
            and not os.path.exists(dump_path)
            and not forward_batch.forward_mode.is_dllm_extend()
            and input_ids.numel() >= 16
            and getattr(output, "next_token_logits", None) is not None
        ):
            torch.save(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "positions": positions.detach().cpu(),
                    "forward_batch_positions": getattr(forward_batch, "positions", torch.empty(0)).detach().cpu(),
                    "extend_prefix_lens": getattr(forward_batch, "extend_prefix_lens", torch.empty(0)).detach().cpu(),
                    "extend_seq_lens": getattr(forward_batch, "extend_seq_lens", torch.empty(0)).detach().cpu(),
                    "next_token_logits": output.next_token_logits.detach().float().cpu(),
                },
                dump_path,
            )
        return output

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map the wrapper checkpoint into SGLang's fused QKV/MLP layout."""

        params = dict(self.named_parameters(remove_duplicate=False))
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        loaded: set[str] = set()
        weight_prefix = os.environ.get("GAM_SGLANG_WEIGHT_PREFIX", "base_model.")
        for raw_name, tensor in weights:
            if weight_prefix and not raw_name.startswith(weight_prefix):
                continue
            name = raw_name[len(weight_prefix):]
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
            elif name.startswith("model.visual."):
                name = "visual." + name[len("model.visual.") :]
            # lm_head is already at the wrapper root after stripping.
            layer_id = get_layer_id(name)
            if layer_id is not None and not (
                self.model.start_layer <= layer_id < self.model.end_layer
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            handled = False
            for target, source, shard in stacked:
                if source not in name:
                    continue
                candidate = name.replace(source, target)
                if candidate.endswith(".bias") and candidate not in params:
                    handled = True
                    break
                if candidate not in params:
                    handled = True
                    break
                param = params[candidate]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor, shard)
                loaded.add(candidate)
                handled = True
                break
            if handled:
                continue
            if name.endswith(".bias") and name not in params:
                continue
            if name not in params:
                # The checkpoint may contain auxiliary HF-only buffers.  Do
                # not guess a mapping for them; fail only if core weights are
                # absent after the complete iterator.
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(name)

        core = {
            n for n, p in params.items()
            if p.requires_grad and "rotary_emb" not in n
        }
        missing = sorted(core - loaded)
        if missing:
            raise RuntimeError(
                "GAM SGLang checkpoint mapping left core parameters unloaded: "
                + ", ".join(missing[:12])
            )
        # Optional one-row checkpoint audit.  The appended atomic mask row is
        # the only embedding that has no causal-prefill counterpart; a bad
        # loader/mapping there produces plausible prompt logits but completely
        # different first denoise logits.  Keep this strictly opt-in and
        # write only a tiny CPU tensor, never a production artifact.
        dump_path = os.environ.get("GAM_SGLANG_DUMP_MASK_ROW_PATH")
        if dump_path:
            mask_id = int(os.environ.get("GAM_SGLANG_MASK_ID", "-1"))
            if 0 <= mask_id < self.model.embed_tokens.weight.shape[0]:
                import torch as _torch

                _torch.save(
                    {
                        "mask_id": mask_id,
                        "embed": self.model.embed_tokens.weight[mask_id].detach().float().cpu(),
                        "lm_head": self.lm_head.weight[mask_id].detach().float().cpu(),
                    },
                    dump_path,
                )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


FastDVLMForConditionalGeneration.__name__ = "Fast_dVLMForConditionalGeneration"
EntryClass = FastDVLMForConditionalGeneration
