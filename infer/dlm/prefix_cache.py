"""Fail-closed contracts for exact Qwen3.5 Hierarchy prefix caching.

Hierarchy decoding keeps the tokens before ``block_start`` fixed while it
denoises the current block.  An exact cache therefore has two independent
parts:

* GatedDeltaNet convolution/recurrent states for the text-only noisy prefix;
* full-attention K/V tensors for the multimodal clean prefix.

The tensors in :class:`HybridBlockPrefixCache` are borrowed read-only.  A
denoising forward must clone/fork the linear states before passing them to a
Transformers cache because Qwen3.5 updates those tensors in-place.  Clean K/V
is never forked: cached draft attention only reads it.

This module deliberately contains no fallback to a full-prefix forward.  A
runtime layout that is not understood raises :class:`PrefixCacheContractError`
so an evaluation cannot silently claim prefix-cache acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Iterable, Sequence

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


_INFERENCE_TENSOR_VERSION = -1

# Short current-block queries can make FlexAttention select its decoding
# kernel, whose D=256 autotune candidate exceeds H800 shared memory.  Keep this
# boundary independent from the full-prefix oracle and pin the regular kernel
# to a conservative shape supported by both target runtimes.
_PREFIX_FLEX_KERNEL_OPTIONS = {
    "FORCE_USE_FLEX_ATTENTION": True,
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "ROWS_GUARANTEED_SAFE": True,
    "BLOCKS_ARE_CONTIGUOUS": True,
}


def _prefix_flex_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=_PREFIX_FLEX_KERNEL_OPTIONS,
    )


_compiled_prefix_flex_attention = torch.compile(
    _prefix_flex_attention_kernel,
    dynamic=True,
    fullgraph=True,
)


class PrefixCacheContractError(RuntimeError):
    """Raised when exact prefix-cache semantics cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrefixCacheContractError(message)


def _tensor_version(tensor: torch.Tensor) -> int:
    """Return a mutation version without materializing a large cache clone.

    PyTorch inference tensors intentionally do not own version counters, and
    accessing ``_version`` raises.  Identity, storage address and shape remain
    stamped for those tensors; ordinary tensors retain exact version-based
    in-place mutation detection.
    """

    if torch.is_inference(tensor):
        return _INFERENCE_TENSOR_VERSION
    return int(tensor._version)


def _state_zero(value: Any, *, name: str) -> torch.Tensor:
    """Return state slot zero from old and new Transformers cache layouts."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        value = value.get(0)
    elif isinstance(value, (list, tuple)):
        value = value[0] if value else None
    _require(isinstance(value, torch.Tensor), f"{name} 缺少 tensor state[0]")
    return value


def _layer_kind(layer: Any, layer_index: int) -> str:
    kind = getattr(layer, "layer_type", None)
    if kind is None:
        kind = getattr(layer, "block_type", None)
    _require(
        kind in {"linear_attention", "full_attention"},
        f"layer {layer_index} 类型不受支持: {kind!r}",
    )
    return str(kind)


def _linear_state_from_cache(cache: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one Qwen3.5 linear layer without mutating its runtime cache."""

    if hasattr(cache, "layers"):
        layers = cache.layers
        _require(layer_index < len(layers), f"linear cache 缺少 layer {layer_index}")
        layer_cache = layers[layer_index]
        conv_states = getattr(layer_cache, "conv_states", None)
        recurrent_states = getattr(layer_cache, "recurrent_states", None)
    else:
        # Pinned Transformers 5.2.0.dev0 layout used by the PPU image.
        conv_states = getattr(cache, "conv_states", None)
        recurrent_states = getattr(cache, "recurrent_states", None)
        _require(
            isinstance(conv_states, (dict, list, tuple))
            and isinstance(recurrent_states, (dict, list, tuple)),
            "linear cache 既无 layers，也无 conv_states/recurrent_states",
        )
        try:
            conv_states = conv_states[layer_index]
            recurrent_states = recurrent_states[layer_index]
        except (IndexError, KeyError) as error:
            raise PrefixCacheContractError(
                f"linear cache 缺少 layer {layer_index}"
            ) from error
    return (
        _state_zero(conv_states, name=f"layer {layer_index} conv"),
        _state_zero(recurrent_states, name=f"layer {layer_index} recurrent"),
    )


def _attention_state_from_cache(
    cache: Any, layer_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one full-attention K/V pair from known cache layouts."""

    if hasattr(cache, "layers"):
        layers = cache.layers
        _require(layer_index < len(layers), f"clean cache 缺少 layer {layer_index}")
        layer_cache = layers[layer_index]
        key = getattr(layer_cache, "keys", None)
        value = getattr(layer_cache, "values", None)
    else:
        keys = getattr(cache, "key_cache", None)
        values = getattr(cache, "value_cache", None)
        _require(
            isinstance(keys, (dict, list, tuple))
            and isinstance(values, (dict, list, tuple)),
            "clean cache 既无 layers，也无 key_cache/value_cache",
        )
        try:
            key, value = keys[layer_index], values[layer_index]
        except (IndexError, KeyError) as error:
            raise PrefixCacheContractError(
                f"clean cache 缺少 layer {layer_index}"
            ) from error
    _require(
        isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor),
        f"clean cache layer {layer_index} K/V 未初始化",
    )
    return key, value


@dataclass(frozen=True)
class LinearPrefixState:
    """Borrowed, immutable initial state for one GatedDeltaNet layer."""

    layer_index: int
    conv: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True)
class CleanAttentionPrefixState:
    """Borrowed, immutable clean-prefix K/V for one full-attention layer."""

    layer_index: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PrefixCacheMutationStamp:
    """Tensor identity/version stamp used to detect accidental cache writes.

    The version field is ``-1`` for PyTorch inference tensors, which have no
    version counter.  This avoids cloning the large immutable prefix solely
    for bookkeeping.
    """

    entries: tuple[tuple[int, int, int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ForkedLinearPrefixState:
    """Private mutable state owned by exactly one denoising forward."""

    layer_index: int
    conv: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True)
class HybridBlockPrefixCache:
    """Read-only prefix state reused within one Hierarchy diffusion block."""

    block_start: int
    block_size: int
    batch_size: int
    prefix_token_count: int
    noisy_text_token_count: int
    position_axes: int
    layer_count: int
    linear: tuple[LinearPrefixState, ...]
    attention: tuple[CleanAttentionPrefixState, ...]

    @classmethod
    def capture(
        cls,
        *,
        layers: Sequence[Any],
        noisy_cache: Any,
        clean_cache: Any,
        block_start: int,
        block_size: int,
        batch_size: int,
        noisy_text_token_count: int,
        position_axes: int,
        require_qwen35_fastpath: bool = True,
    ) -> "HybridBlockPrefixCache":
        """Capture borrowed tensors from validated Qwen3.5 runtime caches.

        ``clean_cache`` must have been committed through the token immediately
        before ``block_start``. ``noisy_cache`` is a separate cache produced by
        a bidirectional text-only noisy-prefix prefill.
        """

        _require(len(layers) > 0, "language model 没有 decoder layers")
        _require(block_start > 0, "block_start 必须为正")
        _require(block_size >= 2, "block_size 必须至少为 2")
        _require(batch_size >= 1, "batch_size 必须为正")
        _require(0 < noisy_text_token_count <= block_start, "noisy text prefix 长度非法")
        _require(position_axes >= 1, "position_axes 必须为正")

        get_seq_length = getattr(clean_cache, "get_seq_length", None)
        _require(callable(get_seq_length), "clean cache 缺少 get_seq_length()")
        clean_length = int(get_seq_length())
        _require(
            clean_length == block_start,
            f"clean cache 长度 {clean_length} != block_start {block_start}",
        )

        linear: list[LinearPrefixState] = []
        attention: list[CleanAttentionPrefixState] = []
        for layer_index, layer in enumerate(layers):
            kind = _layer_kind(layer, layer_index)
            if kind == "linear_attention":
                linear_attn = getattr(layer, "linear_attn", None)
                _require(linear_attn is not None, f"layer {layer_index} 缺少 linear_attn")
                if require_qwen35_fastpath:
                    forward = getattr(type(linear_attn), "forward", None)
                    _require(
                        getattr(forward, "__name__", "") == "qwen35_cached_chunk_forward",
                        "Qwen3.5 cached multi-token fast path 未安装；拒绝静默全前缀 fallback",
                    )
                conv, recurrent = _linear_state_from_cache(noisy_cache, layer_index)
                expected_conv = (
                    batch_size,
                    int(getattr(linear_attn, "conv_dim", -1)),
                    int(getattr(linear_attn, "conv_kernel_size", -1)),
                )
                expected_recurrent = (
                    batch_size,
                    int(getattr(linear_attn, "num_v_heads", -1)),
                    int(getattr(linear_attn, "head_k_dim", -1)),
                    int(getattr(linear_attn, "head_v_dim", -1)),
                )
                _require(
                    tuple(conv.shape) == expected_conv,
                    f"layer {layer_index} conv shape {tuple(conv.shape)} != {expected_conv}",
                )
                _require(
                    tuple(recurrent.shape) == expected_recurrent,
                    "layer "
                    f"{layer_index} recurrent shape {tuple(recurrent.shape)} != {expected_recurrent}",
                )
                linear.append(LinearPrefixState(layer_index, conv, recurrent))
            else:
                self_attn = getattr(layer, "self_attn", None)
                _require(self_attn is not None, f"layer {layer_index} 缺少 self_attn")
                key, value = _attention_state_from_cache(clean_cache, layer_index)
                _require(
                    key.ndim == 4 and value.ndim == 4,
                    f"layer {layer_index} K/V 必须为 [B,H,S,D]",
                )
                _require(
                    key.shape[0] == value.shape[0] == batch_size,
                    f"layer {layer_index} K/V batch 不匹配",
                )
                _require(
                    key.shape[-2] == value.shape[-2] == block_start,
                    f"layer {layer_index} K/V prefix length 不匹配",
                )
                _require(
                    key.device == value.device and key.dtype == value.dtype,
                    f"layer {layer_index} K/V device 或 dtype 不匹配",
                )
                attention.append(CleanAttentionPrefixState(layer_index, key, value))

        _require(linear, "未发现 linear-attention layer")
        _require(attention, "未发现 full-attention layer")
        cache = cls(
            block_start=block_start,
            block_size=block_size,
            batch_size=batch_size,
            prefix_token_count=block_start,
            noisy_text_token_count=noisy_text_token_count,
            position_axes=position_axes,
            layer_count=len(layers),
            linear=tuple(linear),
            attention=tuple(attention),
        )
        cache.validate()
        return cache

    def validate(self) -> None:
        """Validate layer coverage and immutable tensor properties."""

        indices = [state.layer_index for state in self.linear]
        indices.extend(state.layer_index for state in self.attention)
        _require(
            sorted(indices) == list(range(self.layer_count)),
            "prefix cache layer coverage 必须完整且无重复",
        )
        tensors = self._borrowed_tensors()
        _require(all(not tensor.requires_grad for tensor in tensors), "prefix cache tensor 不应带梯度")
        devices = {tensor.device for tensor in tensors}
        _require(len(devices) == 1, "prefix cache tensor 必须位于同一 device")

    def validate_current(
        self,
        current_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
    ) -> int:
        """Validate the inherited-token + visible-subblock draft shape."""

        _require(current_ids.ndim == 2, "current_ids 必须为 [B,L]")
        _require(current_ids.shape[0] == self.batch_size, "current_ids batch 不匹配")
        width = int(current_ids.shape[1])
        _require(2 <= width <= self.block_size, f"current block width {width} 非法")
        _require(
            current_position_ids.shape == (self.position_axes, self.batch_size, width),
            "current_position_ids shape 不匹配",
        )
        cache_device = self._borrowed_tensors()[0].device
        _require(
            current_ids.device == current_position_ids.device == cache_device,
            "current tensors 与 prefix cache device 不一致",
        )
        return width

    def fork_linear(self) -> tuple[ForkedLinearPrefixState, ...]:
        """Clone mutable linear states for one denoising NFE."""

        return tuple(
            ForkedLinearPrefixState(
                layer_index=state.layer_index,
                conv=state.conv.clone(),
                recurrent=state.recurrent.clone(),
            )
            for state in self.linear
        )

    def mutation_stamp(self) -> PrefixCacheMutationStamp:
        entries = tuple(
            (
                id(tensor),
                tensor.data_ptr(),
                _tensor_version(tensor),
                tuple(tensor.shape),
            )
            for tensor in self._borrowed_tensors()
        )
        return PrefixCacheMutationStamp(entries)

    def assert_unchanged(self, stamp: PrefixCacheMutationStamp) -> None:
        """Fail if a denoising forward mutated any borrowed state tensor."""

        _require(self.mutation_stamp() == stamp, "denoise forward 污染了只读 prefix cache")

    @property
    def borrowed_bytes(self) -> int:
        """Number of bytes referenced by the cache, without double counting."""

        seen: set[tuple[torch.device, int]] = set()
        total = 0
        for tensor in self._borrowed_tensors():
            identity = (tensor.device, tensor.data_ptr())
            if identity in seen:
                continue
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
        return total

    def _borrowed_tensors(self) -> tuple[torch.Tensor, ...]:
        result: list[torch.Tensor] = []
        for state in self.linear:
            result.extend((state.conv, state.recurrent))
        for state in self.attention:
            result.extend((state.key, state.value))
        return tuple(result)


def linear_state_map(
    states: Iterable[ForkedLinearPrefixState],
) -> dict[int, ForkedLinearPrefixState]:
    """Index a fork while rejecting duplicate layer states."""

    result: dict[int, ForkedLinearPrefixState] = {}
    for state in states:
        _require(state.layer_index not in result, f"重复 linear state: layer {state.layer_index}")
        result[state.layer_index] = state
    return result


class Qwen35PrefixDraftRunner:
    """Current-block-only draft forward for the pinned Qwen3.5 runtime.

    Hierarchy uses this runner only through the explicit opt-in configuration.
    Deployments must establish logits/token parity against
    ``model.draft_block_logits`` on each real target runtime first.
    """

    _CACHE_CLASS = "Qwen3_5DynamicCache"
    _CACHE_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
    _ATTENTION_FORWARD_SHA256 = "4978ddd72dcaeb7505f0cc927392f066bbe39aaaecb38369e3e6b496b49f908a"
    _APPLY_ROPE_SHA256 = "ab8feb64f19555713efc4c559ce306b35205b1197f8431784585999d9a5b6662"
    _CACHE_INIT_SHA256 = "02a3d63a14f80e926a05d95aa08abec8acfbba42c906e39bd42ae1725bda5f42"

    def __init__(self, model: Any, block_size: int):
        self.model = model
        self.language_model = model.language_model
        self.layers = tuple(self.language_model.layers)
        self.block_size = int(block_size)
        self._visibility_masks: dict[tuple[str, int, int, int], Any] = {}
        self._guard_qwen_sources()
        _require(self.block_size >= 2, "prefix draft block_size 必须至少为 2")
        _require(self.layers, "Qwen3.5 language model 没有 layers")
        for layer_index, layer in enumerate(self.layers):
            kind = _layer_kind(layer, layer_index)
            if kind == "linear_attention":
                attention = getattr(layer, "linear_attn", None)
                _require(attention is not None, f"layer {layer_index} 缺少 linear_attn")
                forward = getattr(type(attention), "forward", None)
                _require(
                    getattr(forward, "__name__", "") == "qwen35_cached_chunk_forward",
                    "Qwen3.5 cached multi-token fast path 未安装",
                )
            else:
                attention = getattr(layer, "self_attn", None)
                _require(attention is not None, f"layer {layer_index} 缺少 self_attn")
                for name in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "q_norm",
                    "k_norm",
                    "o_proj",
                    "head_dim",
                ):
                    _require(
                        hasattr(attention, name),
                        f"Qwen3.5 full attention 缺少受保护字段 {name}",
                    )

    @torch.inference_mode()
    def build(
        self,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        clean_cache: Any,
    ) -> HybridBlockPrefixCache:
        """Run one text-only noisy-prefix prefill and borrow clean K/V."""

        _require(prefix_input_ids.ndim == 2, "prefix_input_ids 必须为 [B,L]")
        batch_size, prefix_length = prefix_input_ids.shape
        _require(int(batch_size) == 1, "prefix cache runner 仅支持 batch size 1")
        _require(prefix_length > 0, "prefix 不能为空")
        _require(
            prefix_position_ids.ndim == 3
            and prefix_position_ids.shape[1:] == prefix_input_ids.shape,
            "prefix_position_ids shape 不匹配",
        )
        self._guard_cache(clean_cache)

        cfg = self.model.config
        text = torch.ones_like(prefix_input_ids[0], dtype=torch.bool)
        for token_id in (
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        ):
            text &= prefix_input_ids[0].ne(token_id)
        noisy_ids = prefix_input_ids[:, text]
        _require(noisy_ids.shape[1] > 0, "text-only noisy prefix 不能为空")
        noisy_positions = prefix_position_ids[:, :, text]
        noisy = self.language_model.embed_tokens(noisy_ids)
        rotary_positions = self._rotary_positions(noisy_positions)
        position_embeddings = self.language_model.rotary_emb(noisy, rotary_positions)
        noisy_valid = torch.ones_like(noisy_ids, dtype=torch.bool)
        block_mask = self._all_visible_block_mask(
            query_length=int(noisy.shape[1]),
            key_length=int(noisy.shape[1]),
            num_heads=int(self.model.config.text_config.num_attention_heads),
            device=noisy.device,
        )

        noisy_cache = type(clean_cache)(self.language_model.config)
        self._guard_cache(noisy_cache)
        for layer_index, layer in enumerate(self.layers):
            if _layer_kind(layer, layer_index) == "linear_attention":
                noisy = layer(
                    noisy,
                    position_embeddings=position_embeddings,
                    attention_mask=noisy_valid,
                    past_key_values=noisy_cache,
                )
            else:
                noisy = self._full_layer(
                    layer,
                    noisy,
                    position_embeddings,
                    clean_prefix=None,
                    block_mask=block_mask,
                )
            noisy = noisy * noisy_valid.unsqueeze(-1).to(noisy.dtype)

        return HybridBlockPrefixCache.capture(
            layers=self.layers,
            noisy_cache=noisy_cache,
            clean_cache=clean_cache,
            block_start=int(prefix_length),
            block_size=self.block_size,
            batch_size=int(batch_size),
            noisy_text_token_count=int(noisy_ids.shape[1]),
            position_axes=int(prefix_position_ids.shape[0]),
        )

    @torch.inference_mode()
    def draft_logits(
        self,
        cache: HybridBlockPrefixCache,
        *,
        current_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits after the inherited first token using only current block."""

        cache.validate_current(current_ids, current_position_ids)
        stamp = cache.mutation_stamp()
        current = self.language_model.embed_tokens(current_ids)
        rotary_positions = self._rotary_positions(current_position_ids)
        position_embeddings = self.language_model.rotary_emb(current, rotary_positions)
        current_valid = torch.ones_like(current_ids, dtype=torch.bool)
        runtime_cache = self._fork_runtime_cache(cache)
        attention = {state.layer_index: state for state in cache.attention}
        block_mask = self._all_visible_block_mask(
            query_length=int(current.shape[1]),
            key_length=int(current.shape[1]) + cache.prefix_token_count,
            num_heads=int(self.model.config.text_config.num_attention_heads),
            device=current.device,
        )

        for layer_index, layer in enumerate(self.layers):
            if _layer_kind(layer, layer_index) == "linear_attention":
                current = layer(
                    current,
                    position_embeddings=position_embeddings,
                    attention_mask=current_valid,
                    past_key_values=runtime_cache,
                )
            else:
                _require(
                    layer_index in attention,
                    f"prefix cache 缺少 full-attention layer {layer_index}",
                )
                current = self._full_layer(
                    layer,
                    current,
                    position_embeddings,
                    clean_prefix=attention[layer_index],
                    block_mask=block_mask,
                )
            current = current * current_valid.unsqueeze(-1).to(current.dtype)

        current = self.language_model.norm(current)
        logits = self.model.lm_head(current[0, :-1])
        cache.assert_unchanged(stamp)
        return logits

    def _fork_runtime_cache(self, cache: HybridBlockPrefixCache) -> Any:
        runtime_cache = self._cache_type()(self.language_model.config)
        self._guard_cache(runtime_cache)
        _require(
            isinstance(runtime_cache.conv_states, list)
            and isinstance(runtime_cache.recurrent_states, list),
            "当前 exact runner 只支持 pinned Qwen3_5DynamicCache list layout",
        )
        for state in cache.fork_linear():
            runtime_cache.conv_states[state.layer_index] = state.conv
            runtime_cache.recurrent_states[state.layer_index] = state.recurrent
        return runtime_cache

    def _cache_type(self) -> type:
        # Import only on an actual Qwen3.5 worker.  Local CPU unit-test images
        # intentionally need not install the pinned Transformers version.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        return Qwen3_5DynamicCache

    def _guard_qwen_sources(self) -> None:
        provenance = self.qwen_source_provenance()
        for name, item in provenance.items():
            _require(
                item["matches"],
                f"{name} source 漂移: {item['actual_sha256']} != {item['expected_sha256']}",
            )

    @classmethod
    def qwen_source_provenance(cls) -> dict[str, dict[str, str | bool]]:
        """Return exact pinned-source hashes before constructing a runner."""

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Attention,
            Qwen3_5DynamicCache,
            apply_rotary_pos_emb,
        )

        guarded = (
            (Qwen3_5Attention.forward, cls._ATTENTION_FORWARD_SHA256, "Qwen3_5Attention.forward"),
            (apply_rotary_pos_emb, cls._APPLY_ROPE_SHA256, "apply_rotary_pos_emb"),
            (Qwen3_5DynamicCache.__init__, cls._CACHE_INIT_SHA256, "Qwen3_5DynamicCache.__init__"),
        )
        result: dict[str, dict[str, str | bool]] = {}
        for function, expected, name in guarded:
            actual = hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
            result[name] = {
                "module": function.__module__,
                "actual_sha256": actual,
                "expected_sha256": expected,
                "matches": actual == expected,
            }
        return result

    def _guard_cache(self, cache: Any) -> None:
        cache_type = type(cache)
        _require(
            cache_type.__name__ == self._CACHE_CLASS
            and cache_type.__module__ == self._CACHE_MODULE,
            "prefix runner 仅支持 pinned Transformers Qwen3_5DynamicCache",
        )
        for name in (
            "key_cache",
            "value_cache",
            "conv_states",
            "recurrent_states",
            "get_seq_length",
        ):
            _require(hasattr(cache, name), f"Qwen3_5DynamicCache 缺少字段 {name}")
        for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            values = getattr(cache, name)
            _require(
                isinstance(values, list) and len(values) == len(self.layers),
                f"Qwen3_5DynamicCache.{name} layer layout 漂移",
            )

    @staticmethod
    def _rotary_positions(position_ids: torch.Tensor) -> torch.Tensor:
        _require(position_ids.shape[0] in (3, 4), "Qwen3.5 position axes 必须为 3 或 4")
        return position_ids[1:] if position_ids.shape[0] == 4 else position_ids

    def _all_visible_block_mask(
        self,
        *,
        query_length: int,
        key_length: int,
        num_heads: int,
        device: torch.device,
    ) -> Any:
        _require(query_length > 0 and key_length >= query_length, "prefix attention 长度非法")
        cache_key = (str(device), query_length, key_length, num_heads)
        cached = self._visibility_masks.get(cache_key)
        if cached is not None:
            return cached

        def all_visible(
            b: torch.Tensor,
            h: torch.Tensor,
            q: torch.Tensor,
            kv: torch.Tensor,
        ) -> torch.Tensor:
            del b, h, kv
            return q >= 0

        block_mask = create_block_mask(
            all_visible,
            B=1,
            H=num_heads,
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=str(device),
            _compile=False,
        )
        self._visibility_masks[cache_key] = block_mask
        return block_mask

    @staticmethod
    def _full_layer(
        layer: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_prefix: CleanAttentionPrefixState | None,
        block_mask: Any,
    ) -> torch.Tensor:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        hidden = Qwen35PrefixDraftRunner._full_attention(
            layer.self_attn,
            hidden,
            position_embeddings,
            clean_prefix,
            block_mask,
        )
        hidden = residual + hidden
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    @staticmethod
    def _full_attention(
        attention: Any,
        states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_prefix: CleanAttentionPrefixState | None,
        block_mask: Any,
    ) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        input_shape = states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query, gate = torch.chunk(
            attention.q_proj(states).view(*input_shape, -1, attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(states).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if clean_prefix is not None:
            _require(
                clean_prefix.key.shape[:2] == key.shape[:2]
                and clean_prefix.key.shape[-1] == key.shape[-1],
                f"layer {clean_prefix.layer_index} clean/current key shape 不兼容",
            )
            _require(
                clean_prefix.value.shape[:2] == value.shape[:2]
                and clean_prefix.value.shape[-1] == value.shape[-1],
                f"layer {clean_prefix.layer_index} clean/current value shape 不兼容",
            )
            key = torch.cat((key, clean_prefix.key), dim=2)
            value = torch.cat((value, clean_prefix.value), dim=2)
        output = _compiled_prefix_flex_attention(query, key, value, block_mask)
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return attention.o_proj(output)
