"""Qwen3.5 Direct-Conversion model for GAM.

Qwen3.5 interleaves causal GatedDeltaNet and full-attention layers.  A
Fast-dVLM block mask cannot be applied to the recurrent linear layers.  This
adapter therefore keeps noisy and clean states isolated through those layers
and applies the exact Direct-Conversion N2N/N2C/C2C mask in every full-
attention layer.  Vision is encoded once and appears only in the clean stream.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from transformers.utils import ModelOutput


MASK_TOKEN = "|<MASK>|"
STABLE_BLOCK_MASK_ENV = "GAM_DLM_STABLE_BLOCK_MASK"
FLEX_CUDAGRAPHS_ENV = "GAM_DLM_FLEX_CUDAGRAPHS"
ATTENTION_PROFILE_ENV = "GAM_DLM_ATTENTION_PROFILE"


@contextmanager
def _attention_profile_region(name: str):
    if os.environ.get(ATTENTION_PROFILE_ENV) == "1":
        with torch.profiler.record_function(name):
            yield
    else:
        yield


def _flex_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    return flex_attention(query, key, value, block_mask=block_mask, enable_gqa=True)


# Calling flex_attention from eager mode deliberately uses its dense reference
# backend.  Compile this boundary so Inductor lowers the HOP to the block-sparse
# Triton kernel.  Dynamic shapes avoid recompiling for every variable-length
# multimodal batch.
_flex_cudagraphs = os.environ.get(FLEX_CUDAGRAPHS_ENV, "1") == "1"
_compiled_flex_attention = torch.compile(
    _flex_attention_kernel,
    dynamic=True,
    fullgraph=True,
    options={
        # Packed multimodal lengths vary on every optimizer step.  Retaining
        # one CUDA Graph pool per concrete shape causes live HBM to grow until
        # OOM even though the dynamic Inductor/Triton kernel itself is stable.
        # Keep the historical default for other routes; PPU formal training
        # opts out explicitly through the environment contract.
        "triton.cudagraphs": _flex_cudagraphs,
        "triton.cudagraph_trees": _flex_cudagraphs,
    },
)


def compute_response_blocks(labels: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Match Fast-dVLM block/turn boundaries for one unpadded sample."""

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    response = labels.ne(-100)
    block = torch.full_like(labels, -1)
    current_block = 0
    in_response = False
    response_position = 0
    for index in range(labels.numel()):
        if bool(response[index]):
            if not in_response:
                in_response = True
                response_position = 0
            block[index] = current_block + response_position // block_size
            response_position += 1
        elif in_response:
            current_block += (response_position + block_size - 1) // block_size
            in_response = False
    if in_response:
        current_block += (response_position + block_size - 1) // block_size

    turn = torch.zeros_like(labels)
    for index in range(1, labels.numel()):
        turn[index] = turn[index - 1] + block[index].ne(block[index - 1]).to(turn.dtype)
    return block, turn, current_block


def initialize_mask_token(tokenizer: Any, model: nn.Module) -> int:
    """Add one token and mean-initialize both untied vocabulary rows."""

    encoded_before = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)
    if len(encoded_before) == 1 and tokenizer.convert_ids_to_tokens(encoded_before[0]) == MASK_TOKEN:
        mask_id = int(encoded_before[0])
    else:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
        if added != 1:
            raise RuntimeError(f"expected to add exactly one mask token, got {added}")
        fallback_output = None
        if model.get_output_embeddings() is None and isinstance(getattr(model, "lm_head", None), nn.Linear):
            fallback_output = model.lm_head
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        if fallback_output is not None and fallback_output.out_features != len(tokenizer):
            resized_output = nn.Linear(
                fallback_output.in_features,
                len(tokenizer),
                bias=fallback_output.bias is not None,
                device=fallback_output.weight.device,
                dtype=fallback_output.weight.dtype,
            )
            with torch.no_grad():
                resized_output.weight[: fallback_output.out_features].copy_(fallback_output.weight)
                if fallback_output.bias is not None:
                    resized_output.bias[: fallback_output.out_features].copy_(fallback_output.bias)
            model.lm_head = resized_output
        mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))

        seed_tokens = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
        seed_ids = [
            int(tokenizer.convert_tokens_to_ids(token))
            for token in seed_tokens
            if int(tokenizer.convert_tokens_to_ids(token)) >= 0
        ]
        if not seed_ids:
            raise RuntimeError("cannot initialize mask row: no seed special tokens")
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight
            embeddings[mask_id].copy_(embeddings[seed_ids].float().mean(dim=0).to(embeddings.dtype))
            output = model.get_output_embeddings()
            if output is None:
                output = getattr(model, "lm_head", None)
            if output is not None and output.weight.data_ptr() != embeddings.data_ptr():
                output.weight[mask_id].copy_(output.weight[seed_ids].float().mean(dim=0).to(output.weight.dtype))

    check = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)
    if check != [mask_id]:
        raise RuntimeError(f"mask token is not atomic: {check}")
    return mask_id


@dataclass
class DLMOutput(ModelOutput):
    loss: torch.Tensor
    logits: torch.Tensor | None = None
    mdm_loss: torch.Tensor | None = None
    causal_loss: torch.Tensor | None = None


def _pad_tensor_list(values: list[torch.Tensor], padding_value: float | int) -> torch.Tensor:
    return nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding_value)


class GAMQwen35DLM(nn.Module):
    """Direct Conversion wrapper around a Qwen3.5 VLM.

    Training uses complementary noisy views.  Evaluation uses the same hybrid
    noisy/clean attention contract through :meth:`draft_block`; causal
    verification is deliberately delegated to the unchanged base VLM.
    """

    def _native_position_ids(self, positions):
        return positions

    def __init__(
        self,
        base_model: nn.Module,
        mask_token_id: int,
        im_end_token_id: int,
        block_size: int = 32,
        minimum_noise_level: float = 0.0,
        allow_nondefault_block_size: bool = False,
    ) -> None:
        super().__init__()
        if block_size not in (8, 32):
            raise ValueError("GAM Direct Conversion only validates fixed block_size 8 or 32")
        if block_size != 32 and not allow_nondefault_block_size:
            raise ValueError(
                "non-default DLM block size requires the isolated fixed_block_size_experiment route"
            )
        if not 0.0 <= minimum_noise_level < 1.0:
            raise ValueError("minimum_noise_level must be in [0, 1)")
        self.base_model = base_model
        self.mask_token_id = int(mask_token_id)
        self.im_end_token_id = int(im_end_token_id)
        self.block_size = block_size
        self.minimum_noise_level = minimum_noise_level
        # Backward-compatible default: historical DLM runs optimize the
        # equally weighted MDM and causal objectives. New experiments must
        # opt out explicitly through ``casuallossenable``.
        self.casuallossenable = True
        self.mdm_loss_weight = 0.5
        self.causal_loss_weight = 0.5
        self.config = base_model.config
        self.gradient_checkpointing = False
        self.language_checkpoint_stride = 0
        # Padding-free DLM uses a separate, opt-in checkpoint cadence.  A
        # stride of one preserves the established route (every language
        # layer); larger values checkpoint every Nth layer uniformly for all
        # samples.  The legacy non-packed stride above remains untouched.
        self.packing_language_checkpoint_stride = 1
        self.padding_free_packing = False
        self.packing_linear_kernel = "chunk"
        self.packing_activation_cpu_offload = False
        self.packing_offload_threshold_bytes = 8 << 20
        self.packing_activation_offload_min_tokens = 0
        self.packing_activation_offload_layer_stride = 1
        self.packing_activation_offload_pin_memory = True
        self.packing_lm_head_loss_backend = "fused"
        self.packing_lm_head_loss_chunk_tokens = 128
        self._last_packing_activation_cpu_offload = False
        self._last_packing_offloaded_tensor_count = 0
        self._last_packing_offloaded_bytes = 0
        self._gradient_checkpointing_kwargs: dict[str, Any] = {"use_reentrant": False}

    def set_loss_contract(
        self,
        *,
        casuallossenable: bool,
        mdm_loss_weight: float,
        causal_loss_weight: float,
    ) -> None:
        """Configure the isolated training-loss branch without changing inference."""

        mdm_loss_weight = float(mdm_loss_weight)
        causal_loss_weight = float(causal_loss_weight)
        if casuallossenable:
            if (mdm_loss_weight, causal_loss_weight) != (0.5, 0.5):
                raise ValueError("enabled causal loss requires the legacy 0.5/0.5 contract")
        elif (mdm_loss_weight, causal_loss_weight) != (1.0, 0.0):
            raise ValueError("disabled causal loss requires the MDM-only 1.0/0.0 contract")
        self.casuallossenable = bool(casuallossenable)
        self.mdm_loss_weight = mdm_loss_weight
        self.causal_loss_weight = causal_loss_weight

    @property
    def multimodal_model(self) -> nn.Module:
        return self.base_model.model

    @property
    def language_model(self) -> nn.Module:
        return self.multimodal_model.language_model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.base_model.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        """Checkpoint the custom hybrid LM loop and the native vision tower."""

        self.gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            **(gradient_checkpointing_kwargs or {}),
        }
        # Padding-free DLM checkpoints the custom packed language-layer
        # boundary below. Vision checkpointing is configured independently by
        # train_dlm.py; enabling the native whole-model path here would also
        # checkpoint a frozen ViT in General and bypass the packed contract.
        if not self.padding_free_packing and hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self._gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def set_language_checkpoint_stride(self, stride: int) -> None:
        """Checkpoint every Nth LM layer without checkpointing the ViT."""

        if stride < 0:
            raise ValueError("language checkpoint stride must be non-negative")
        self.language_checkpoint_stride = int(stride)

    def set_padding_free_packing(self, enabled: bool) -> None:
        """Pack logical samples with explicit varlen state-reset boundaries."""

        self.padding_free_packing = bool(enabled)
        if self.padding_free_packing and self.language_checkpoint_stride:
            raise ValueError("DLM packing forbids legacy language_checkpoint_stride")

    def set_packing_language_checkpoint_stride(self, stride: int) -> None:
        """Set a sample-independent checkpoint cadence for packed LM layers."""

        if stride < 1:
            raise ValueError("packed language checkpoint stride must be positive")
        self.packing_language_checkpoint_stride = int(stride)

    def _checkpoint_packed_language_layer(self, layer_index: int) -> bool:
        if not self.training or not self.gradient_checkpointing:
            return False
        return (layer_index + 1) % self.packing_language_checkpoint_stride == 0

    def set_packing_linear_kernel(self, kernel: str) -> None:
        """Select the DLM-only varlen GatedDeltaNet implementation."""

        if kernel != "chunk":
            raise ValueError(f"unsupported DLM packing linear kernel: {kernel}")
        self.packing_linear_kernel = kernel

    def set_packing_activation_cpu_offload(
        self,
        enabled: bool,
        threshold_mib: int = 8,
        min_tokens: int = 0,
        layer_stride: int = 1,
        pin_memory: bool = True,
    ) -> None:
        """Offload saved activations with one fixed, sample-independent policy.

        ``layer_stride=1`` applies the hook to every language layer.  Larger
        values apply it to every Nth layer for every batch; unlike the legacy
        token gate, this never changes strategy for long versus short samples.
        """

        if threshold_mib < 1:
            raise ValueError("packing activation offload threshold must be positive")
        if min_tokens < 0:
            raise ValueError("packing activation offload token threshold must be non-negative")
        if layer_stride < 1:
            raise ValueError("packing activation offload layer stride must be positive")
        self.packing_activation_cpu_offload = bool(enabled)
        self.packing_offload_threshold_bytes = int(threshold_mib) << 20
        self.packing_activation_offload_min_tokens = int(min_tokens)
        self.packing_activation_offload_layer_stride = int(layer_stride)
        self.packing_activation_offload_pin_memory = bool(pin_memory)

    def set_packing_lm_head_loss(self, backend: str, chunk_tokens: int = 128) -> None:
        """Select the packed LM-head loss implementation.

        The historical FLA fused implementation remains the default.  The
        checkpointed implementation bounds logits memory and avoids returning
        a freshly materialized full-vocabulary ``dw`` from a custom autograd
        forward on every variable-shape optimizer step.
        """

        if backend not in {"fused", "checkpointed_chunk"}:
            raise ValueError(f"unsupported packed LM-head loss backend: {backend}")
        if chunk_tokens < 1:
            raise ValueError("packed LM-head loss chunk_tokens must be positive")
        self.packing_lm_head_loss_backend = backend
        self.packing_lm_head_loss_chunk_tokens = int(chunk_tokens)

    @contextmanager
    def _packing_saved_tensor_offload(self, enabled: bool):
        if not enabled or not torch.is_grad_enabled():
            yield
            return
        parameter_storages = {
            parameter.untyped_storage().data_ptr()
            for parameter in self.parameters()
            if parameter.device.type == "cuda"
        }
        threshold = self.packing_offload_threshold_bytes

        def pack(tensor: torch.Tensor):
            storage_ptr = tensor.untyped_storage().data_ptr()
            should_offload = (
                tensor.device.type == "cuda"
                and tensor.numel() * tensor.element_size() >= threshold
                and storage_ptr not in parameter_storages
            )
            if not should_offload:
                return None, tensor
            # Preserve strides: causal-conv1d's varlen backward requires the
            # saved BxCxT input to retain its channel-last (C-stride=1) layout.
            packed = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
                pin_memory=self.packing_activation_offload_pin_memory and not tensor.is_sparse,
            )
            packed.copy_(tensor)
            self._last_packing_offloaded_tensor_count += 1
            self._last_packing_offloaded_bytes += tensor.numel() * tensor.element_size()
            return tensor.device, packed

        def unpack(packed):
            device, tensor = packed
            if device is None:
                return tensor
            return tensor.to(device, non_blocking=True)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def _checkpoint_language_layer(self, layer_index: int) -> bool:
        if not self.training:
            return False
        if self.gradient_checkpointing:
            return True
        stride = self.language_checkpoint_stride
        return stride > 0 and (layer_index + 1) % stride == 0

    def _checkpoint_linear_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        def custom_forward(
            states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            valid: torch.Tensor,
        ) -> torch.Tensor:
            return layer(
                states,
                position_embeddings=(cos, sin),
                attention_mask=valid,
                past_key_values=None,
                use_cache=False,
            )

        return checkpoint(
            custom_forward,
            hidden,
            position_embeddings[0],
            position_embeddings[1],
            attention_mask,
            **self._gradient_checkpointing_kwargs,
        )

    def _checkpoint_full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        clean_causal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.utils.checkpoint import checkpoint

        def custom_forward(
            noisy_states: torch.Tensor,
            clean_states: torch.Tensor,
            noisy_cos: torch.Tensor,
            noisy_sin: torch.Tensor,
            clean_cos: torch.Tensor,
            clean_sin: torch.Tensor,
            noisy_states_valid: torch.Tensor,
            clean_states_valid: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._full_layer(
                layer,
                noisy_states,
                clean_states,
                (noisy_cos, noisy_sin),
                (clean_cos, clean_sin),
                block_mask,
                noisy_states_valid,
                clean_states_valid,
                clean_causal_mask,
            )

        return checkpoint(
            custom_forward,
            noisy,
            clean,
            noisy_position_embeddings[0],
            noisy_position_embeddings[1],
            clean_position_embeddings[0],
            clean_position_embeddings[1],
            noisy_valid,
            clean_valid,
            **self._gradient_checkpointing_kwargs,
        )

    def _checkpoint_packed_full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
        clean_cu_seqlens: torch.Tensor,
        clean_max_seqlen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute one packed Qwen layer without saved-tensor CPU hooks."""

        from torch.utils.checkpoint import checkpoint

        def custom_forward(
            noisy_states: torch.Tensor,
            clean_states: torch.Tensor,
            noisy_cos: torch.Tensor,
            noisy_sin: torch.Tensor,
            clean_cos: torch.Tensor,
            clean_sin: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._packed_full_layer(
                layer,
                noisy_states,
                clean_states,
                (noisy_cos, noisy_sin),
                (clean_cos, clean_sin),
                block_mask,
                clean_cu_seqlens,
                clean_max_seqlen,
            )

        return checkpoint(
            custom_forward,
            noisy,
            clean,
            noisy_position_embeddings[0],
            noisy_position_embeddings[1],
            clean_position_embeddings[0],
            clean_position_embeddings[1],
            **self._gradient_checkpointing_kwargs,
        )

    def _embed_clean(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        patch_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del patch_positions
        model = self.multimodal_model
        embeds = model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(outputs.pooler_output, dim=0).to(embeds.device, embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(input_ids, embeds, image_features=image_embeds)
            embeds = embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            outputs = model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(outputs.pooler_output, dim=0).to(embeds.device, embeds.dtype)
            _, video_mask = model.get_placeholder_mask(input_ids, embeds, video_features=video_embeds)
            embeds = embeds.masked_scatter(video_mask, video_embeds)

        position_kwargs = dict(
            input_ids=input_ids,
            inputs_embeds=embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        # Transformers 5.5 added mm_token_type_ids; the pinned 5.2 API derives
        # image/video regions directly from input_ids and grid metadata.
        if "mm_token_type_ids" in inspect.signature(model.compute_3d_position_ids).parameters:
            position_kwargs["mm_token_type_ids"] = mm_token_type_ids
        position_ids = model.compute_3d_position_ids(**position_kwargs)
        if position_ids is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            position_ids = positions.view(1, 1, -1).expand(4, input_ids.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, -1, -1)
        return embeds, position_ids

    def _sample_noise(self, labels: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block, turn, n_blocks = compute_response_blocks(labels, self.block_size)
        response = labels.ne(-100)
        selected = torch.zeros_like(response)
        if n_blocks:
            t = torch.rand(n_blocks, device=labels.device)
            probabilities = self.minimum_noise_level + (1.0 - self.minimum_noise_level) * t
            for block_index in range(n_blocks):
                positions = block.eq(block_index)
                selected[positions] = torch.rand(int(positions.sum()), device=labels.device).lt(
                    probabilities[block_index]
                )
        selected |= response & input_ids.eq(self.im_end_token_id)
        return selected, turn

    def _build_streams(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        clean_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        vision_ids = {
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        }
        base: list[dict[str, torch.Tensor]] = []
        for batch_index in range(input_ids.shape[0]):
            valid = attention_mask[batch_index].bool()
            ids = input_ids[batch_index, valid]
            target = labels[batch_index, valid]
            embeds = clean_embeds[batch_index, valid]
            positions = position_ids[:, batch_index, valid]
            selected, turn = self._sample_noise(target, ids)
            text = torch.ones_like(ids, dtype=torch.bool)
            for token_id in vision_ids:
                text &= ids.ne(token_id)
            base.append(
                {
                    "ids": ids,
                    "labels": target,
                    "embeds": embeds,
                    "positions": positions,
                    "selected": selected,
                    "turn": turn,
                    "text": text,
                }
            )

        noisy_embeds: list[torch.Tensor] = []
        noisy_labels: list[torch.Tensor] = []
        noisy_turn: list[torch.Tensor] = []
        noisy_positions: list[torch.Tensor] = []
        # The clean stream is independent of the noisy stream (C2C only), so
        # complementary noisy views share one clean forward.  Duplicating it
        # per view is mathematically redundant and adds 25% to the language
        # token workload before activation checkpoint recomputation.
        clean_streams = [item["embeds"] for item in base]
        clean_positions = [item["positions"].transpose(0, 1) for item in base]
        clean_turn = [item["turn"] for item in base]
        clean_labels = [item["labels"] for item in base]
        for view in range(2):
            for item in base:
                response = item["labels"].ne(-100)
                mask = item["selected"] if view == 0 else response & ~item["selected"]
                mask |= response & item["ids"].eq(self.im_end_token_id)
                noisy_ids = item["ids"].clone()
                noisy_ids[mask] = self.mask_token_id
                noisy = self.language_model.embed_tokens(noisy_ids)[item["text"]]
                target = item["labels"].clone()
                target[~mask] = -100
                noisy_embeds.append(noisy)
                noisy_labels.append(target[item["text"]])
                noisy_turn.append(item["turn"][item["text"]])
                noisy_positions.append(item["positions"][:, item["text"]].transpose(0, 1))

        return {
            "noisy": _pad_tensor_list(noisy_embeds, 0.0),
            "noisy_valid": _pad_tensor_list(
                [torch.ones(value.shape[0], dtype=torch.bool, device=value.device) for value in noisy_embeds], False
            ),
            "noisy_labels": _pad_tensor_list(noisy_labels, -100),
            "noisy_turn": _pad_tensor_list(noisy_turn, -1),
            "noisy_positions": _pad_tensor_list(noisy_positions, 0).permute(2, 0, 1),
            "clean": _pad_tensor_list(clean_streams, 0.0),
            "clean_valid": _pad_tensor_list(
                [torch.ones(value.shape[0], dtype=torch.bool, device=value.device) for value in clean_streams], False
            ),
            "clean_turn": _pad_tensor_list(clean_turn, -1),
            "clean_positions": _pad_tensor_list(clean_positions, 0).permute(2, 0, 1),
            "clean_labels": _pad_tensor_list(clean_labels, -100),
        }

    @staticmethod
    def _pack_streams(streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove padding while retaining independent logical sequence IDs."""

        noisy_valid = streams["noisy_valid"]
        clean_valid = streams["clean_valid"]
        clean_batch = clean_valid.shape[0]
        if noisy_valid.shape[0] != 2 * clean_batch:
            raise ValueError("packed complementary batch must contain two noisy views")

        def pack_2d(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return torch.cat([value[index, valid[index]] for index in range(value.shape[0])], dim=0).unsqueeze(0)

        def pack_3d(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return torch.cat([value[index, valid[index]] for index in range(value.shape[0])], dim=0).unsqueeze(0)

        def pack_positions(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [value[:, index, valid[index]] for index in range(value.shape[1])],
                dim=1,
            ).unsqueeze(1)

        noisy_lengths = noisy_valid.sum(dim=1, dtype=torch.long)
        clean_lengths = clean_valid.sum(dim=1, dtype=torch.long)
        noisy_cu = F.pad(noisy_lengths.cumsum(0), (1, 0))
        clean_cu = F.pad(clean_lengths.cumsum(0), (1, 0))
        noisy_sequence_id = torch.repeat_interleave(
            torch.arange(noisy_valid.shape[0], device=noisy_valid.device),
            noisy_lengths,
        )
        clean_sample_id = torch.repeat_interleave(
            torch.arange(clean_batch, device=clean_valid.device),
            clean_lengths,
        )
        noisy_sample_id = noisy_sequence_id.remainder(clean_batch)
        noisy_local_position = torch.cat(
            [torch.arange(int(length), device=noisy_valid.device) for length in noisy_lengths.tolist()]
        )
        clean_local_position = torch.cat(
            [torch.arange(int(length), device=clean_valid.device) for length in clean_lengths.tolist()]
        )
        return {
            "noisy": pack_3d(streams["noisy"], noisy_valid),
            "noisy_valid": torch.ones(
                (1, int(noisy_lengths.sum())), dtype=torch.bool, device=noisy_valid.device
            ),
            "noisy_labels": pack_2d(streams["noisy_labels"], noisy_valid),
            "noisy_turn": pack_2d(streams["noisy_turn"], noisy_valid),
            "noisy_positions": pack_positions(streams["noisy_positions"], noisy_valid),
            "noisy_cu_seqlens": noisy_cu,
            "noisy_cu_seqlens_cpu": noisy_cu.cpu(),
            "noisy_sequence_id": noisy_sequence_id,
            "noisy_sample_id": noisy_sample_id,
            "noisy_local_position": noisy_local_position,
            "noisy_max_seqlen": noisy_lengths.max(),
            "clean": pack_3d(streams["clean"], clean_valid),
            "clean_valid": torch.ones(
                (1, int(clean_lengths.sum())), dtype=torch.bool, device=clean_valid.device
            ),
            "clean_turn": pack_2d(streams["clean_turn"], clean_valid),
            "clean_positions": pack_positions(streams["clean_positions"], clean_valid),
            "clean_labels": pack_2d(streams["clean_labels"], clean_valid),
            "clean_cu_seqlens": clean_cu,
            "clean_cu_seqlens_cpu": clean_cu.cpu(),
            "clean_sample_id": clean_sample_id,
            "clean_local_position": clean_local_position,
            "clean_max_seqlen": clean_lengths.max(),
        }

    @staticmethod
    def _block_mask(
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        noisy_turn: torch.Tensor,
        clean_turn: torch.Tensor,
        num_heads: int,
    ) -> Any:
        from torch.nn.attention.flex_attention import create_block_mask

        noisy_length = noisy_valid.shape[1]
        total_length = noisy_length + clean_valid.shape[1]
        clean_batch_size = clean_valid.shape[0]
        noisy_batch_size = noisy_valid.shape[0]
        if noisy_batch_size not in (clean_batch_size, 2 * clean_batch_size):
            raise ValueError(
                "noisy batch must contain one inference view or two complementary training views per clean sample"
            )

        def mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
            del h
            kv_clean = kv >= noisy_length
            kv_pos = torch.where(kv_clean, kv - noisy_length, kv)
            clean_b = b % clean_batch_size
            q_noisy_valid = noisy_valid[b, torch.clamp(q, max=noisy_length - 1)]
            kv_noisy_valid = noisy_valid[b, torch.clamp(kv_pos, max=noisy_length - 1)]
            kv_clean_valid = clean_valid[clean_b, torch.clamp(kv_pos, max=clean_valid.shape[1] - 1)]
            q_valid = q_noisy_valid
            kv_valid = torch.where(kv_clean, kv_clean_valid, kv_noisy_valid)
            q_turn = noisy_turn[b, torch.clamp(q, max=noisy_turn.shape[1] - 1)]
            kv_turn = torch.where(
                kv_clean,
                clean_turn[clean_b, torch.clamp(kv_pos, max=clean_turn.shape[1] - 1)],
                noisy_turn[b, torch.clamp(kv_pos, max=noisy_turn.shape[1] - 1)],
            )
            n2n = ~kv_clean & q_turn.eq(kv_turn)
            n2c = kv_clean & q_turn.gt(kv_turn)
            allowed = q_valid & kv_valid & (n2n | n2c)
            # Flex attention requires at least one key per query. Padding rows
            # attend only to their noisy-stream diagonal and are zeroed after
            # each layer.
            return allowed | (~q_valid & q.eq(kv))

        return create_block_mask(
            mask_mod,
            B=noisy_batch_size,
            H=num_heads,
            Q_LEN=noisy_length,
            KV_LEN=total_length,
            device=str(noisy_valid.device),
            _compile=os.environ.get(STABLE_BLOCK_MASK_ENV, "0") != "1",
        )

    @staticmethod
    def _packed_block_mask(streams: dict[str, torch.Tensor], num_heads: int) -> Any:
        """Build the exact N2N/N2C mask for varlen packed noisy attention."""

        from torch.nn.attention.flex_attention import BlockMask, create_block_mask

        noisy_length = streams["noisy"].shape[1]
        clean_length = streams["clean"].shape[1]
        noisy_sequence_id = streams["noisy_sequence_id"]
        noisy_sample_id = streams["noisy_sample_id"]
        noisy_turn = streams["noisy_turn"][0]
        clean_sample_id = streams["clean_sample_id"]
        clean_turn = streams["clean_turn"][0]

        def noisy_mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
            del b, h
            kv_clean = kv >= noisy_length
            kv_pos = torch.where(kv_clean, kv - noisy_length, kv)
            safe_noisy_kv = torch.clamp(kv_pos, max=noisy_length - 1)
            safe_clean_kv = torch.clamp(kv_pos, max=clean_length - 1)
            n2n = (
                ~kv_clean
                & noisy_sequence_id[q].eq(noisy_sequence_id[safe_noisy_kv])
                & noisy_turn[q].eq(noisy_turn[safe_noisy_kv])
            )
            n2c = (
                kv_clean
                & noisy_sample_id[q].eq(clean_sample_id[safe_clean_kv])
                & noisy_turn[q].gt(clean_turn[safe_clean_kv])
            )
            return n2n | n2c

        if os.environ.get(STABLE_BLOCK_MASK_ENV, "0") != "1":
            return create_block_mask(
                noisy_mask_mod,
                B=1,
                H=num_heads,
                Q_LEN=noisy_length,
                KV_LEN=noisy_length + clean_length,
                device=str(streams["noisy"].device),
                _compile=True,
            )

        # torch 2.9's PPU Inductor backend can emit an invalid-address kernel
        # for create_block_mask on a valid dynamic packing shape.  Its eager
        # fallback materializes the token-level dense mask and is prohibitively
        # slow.  Classify the 128x128 sparse tiles deterministically on CPU,
        # then transfer only the small BlockMask index tensors.  Partial tiles
        # still use the exact mask_mod above inside compiled FlexAttention.
        block_size = 128
        noisy_sequence_cpu = noisy_sequence_id.detach().to("cpu").tolist()
        noisy_sample_cpu = noisy_sample_id.detach().to("cpu").tolist()
        noisy_turn_cpu = noisy_turn.detach().to("cpu").tolist()
        clean_sample_cpu = clean_sample_id.detach().to("cpu").tolist()
        clean_turn_cpu = clean_turn.detach().to("cpu").tolist()
        if not (
            len(noisy_sequence_cpu) == noisy_length
            and len(noisy_sample_cpu) == noisy_length
            and len(noisy_turn_cpu) == noisy_length
            and len(clean_sample_cpu) == clean_length
            and len(clean_turn_cpu) == clean_length
        ):
            raise RuntimeError("packed block-mask metadata length mismatch")

        q_blocks = (noisy_length + block_size - 1) // block_size
        kv_length = noisy_length + clean_length
        kv_blocks = (kv_length + block_size - 1) // block_size
        partial = torch.zeros((q_blocks, kv_blocks), dtype=torch.bool)
        full = torch.zeros_like(partial)

        for q_block in range(q_blocks):
            q_start = q_block * block_size
            q_end = min(q_start + block_size, noisy_length)
            q_sequence_turn = set(
                zip(
                    noisy_sequence_cpu[q_start:q_end],
                    noisy_turn_cpu[q_start:q_end],
                )
            )
            q_sample_turns: dict[int, list[int]] = {}
            for sample, turn in zip(
                noisy_sample_cpu[q_start:q_end],
                noisy_turn_cpu[q_start:q_end],
            ):
                q_sample_turns.setdefault(int(sample), []).append(int(turn))
            q_is_complete = q_end - q_start == block_size

            for kv_block in range(kv_blocks):
                kv_start = kv_block * block_size
                kv_end = min(kv_start + block_size, kv_length)
                any_allowed = False
                all_allowed = q_is_complete and kv_end - kv_start == block_size

                noisy_k_start = kv_start
                noisy_k_end = min(kv_end, noisy_length)
                if noisy_k_start < noisy_k_end:
                    k_sequence_turn = set(
                        zip(
                            noisy_sequence_cpu[noisy_k_start:noisy_k_end],
                            noisy_turn_cpu[noisy_k_start:noisy_k_end],
                        )
                    )
                    any_allowed |= bool(q_sequence_turn.intersection(k_sequence_turn))
                    all_allowed &= (
                        len(q_sequence_turn) == 1
                        and q_sequence_turn == k_sequence_turn
                    )

                clean_k_start = max(kv_start - noisy_length, 0)
                clean_k_end = max(kv_end - noisy_length, 0)
                if clean_k_start < clean_k_end:
                    k_sample_turns: dict[int, list[int]] = {}
                    for sample, turn in zip(
                        clean_sample_cpu[clean_k_start:clean_k_end],
                        clean_turn_cpu[clean_k_start:clean_k_end],
                    ):
                        k_sample_turns.setdefault(int(sample), []).append(int(turn))
                    any_allowed |= any(
                        sample in k_sample_turns
                        and max(q_turns) > min(k_sample_turns[sample])
                        for sample, q_turns in q_sample_turns.items()
                    )
                    same_sample = (
                        len(q_sample_turns) == 1
                        and len(k_sample_turns) == 1
                        and next(iter(q_sample_turns)) == next(iter(k_sample_turns))
                    )
                    all_allowed &= same_sample and (
                        min(next(iter(q_sample_turns.values())))
                        > max(next(iter(k_sample_turns.values())))
                    )

                full[q_block, kv_block] = any_allowed and all_allowed
                partial[q_block, kv_block] = any_allowed and not all_allowed

        partial = partial.view(1, 1, q_blocks, kv_blocks).expand(
            1, num_heads, q_blocks, kv_blocks
        ).contiguous()
        full = full.view(1, 1, q_blocks, kv_blocks).expand(
            1, num_heads, q_blocks, kv_blocks
        ).contiguous()
        def dense_to_ordered(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            integer_mask = mask.to(torch.int32)
            counts = integer_mask.sum(dim=-1).to(torch.int32).contiguous()
            indices = torch.argsort(
                integer_mask,
                dim=-1,
                descending=True,
                stable=True,
            ).to(torch.int32).contiguous()
            return counts, indices

        partial_counts, partial_indices = dense_to_ordered(partial)
        full_counts, full_indices = dense_to_ordered(full)
        block_mask_kwargs: dict[str, Any] = {
            "BLOCK_SIZE": (block_size, block_size),
            "mask_mod": noisy_mask_mod,
        }
        if "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters:
            block_mask_kwargs["seq_lengths"] = (noisy_length, kv_length)
        block_mask = BlockMask.from_kv_blocks(
            partial_counts,
            partial_indices,
            full_counts,
            full_indices,
            **block_mask_kwargs,
        )
        return block_mask.to(streams["noisy"].device)

    @staticmethod
    def _packed_gated_delta_attention(
        attention: nn.Module,
        hidden_states: torch.Tensor,
        sequence_id: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        kernel: str,
    ) -> torch.Tensor:
        """Qwen3.5 GatedDeltaNet with explicit conv and recurrent resets."""

        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("FLA varlen packing requires a flattened batch of one")
        mixed_qkv = attention.in_proj_qkv(hidden_states)
        z = attention.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, attention.head_v_dim)
        b = attention.in_proj_b(hidden_states)
        a = attention.in_proj_a(hidden_states)
        conv_weight = attention.conv1d.weight.squeeze(1)
        if attention.causal_conv1d_fn is not None:
            # Standalone causal-conv1d consumes [B,C,T] and resets its state
            # whenever the deterministic logical sequence id changes.
            mixed_qkv = attention.causal_conv1d_fn(
                x=mixed_qkv.transpose(1, 2),
                weight=conv_weight,
                bias=attention.conv1d.bias,
                activation=attention.activation,
                seq_idx=sequence_id.view(1, -1).to(torch.int32),
            ).transpose(1, 2)
        else:
            # The immutable PPU image intentionally ships FLA 0.5 without the
            # standalone causal-conv1d wheel.  FLA's native varlen convolution
            # has the same boundary-reset contract through cu_seqlens and
            # avoids a slow PyTorch loop/fallback.  It consumes [B,T,D].
            from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

            mixed_qkv = fla_causal_conv1d(
                x=mixed_qkv,
                weight=conv_weight,
                bias=attention.conv1d.bias,
                activation=attention.activation,
                backend="triton",
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                output_final_state=False,
            )
            if isinstance(mixed_qkv, tuple):
                mixed_qkv, final_state = mixed_qkv
                if final_state is not None:
                    raise RuntimeError("FLA packed causal convolution returned an unexpected state")
        query, key, value = torch.split(
            mixed_qkv,
            [attention.key_dim, attention.key_dim, attention.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, attention.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, attention.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, attention.head_v_dim)
        beta = b.sigmoid()
        g = -attention.A_log.float().exp() * F.softplus(a.float() + attention.dt_bias)
        if attention.num_v_heads // attention.num_k_heads > 1:
            repeats = attention.num_v_heads // attention.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        common = {
            "g": g,
            "beta": beta,
            "initial_state": None,
            "output_final_state": False,
            "use_qk_l2norm_in_kernel": True,
            "cu_seqlens": cu_seqlens,
        }
        if kernel == "chunk":
            core_attn_out, _ = attention.chunk_gated_delta_rule(
                query,
                key,
                value,
                cu_seqlens_cpu=cu_seqlens_cpu,
                **common,
            )
        else:
            raise ValueError(f"unsupported DLM packing linear kernel: {kernel}")
        core_attn_out = core_attn_out.reshape(-1, attention.head_v_dim)
        z = z.reshape(-1, attention.head_v_dim)
        core_attn_out = attention.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
        return attention.out_proj(core_attn_out)

    @classmethod
    def _packed_linear_layer(
        cls,
        layer: nn.Module,
        hidden: torch.Tensor,
        sequence_id: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        kernel: str,
    ) -> torch.Tensor:
        residual = hidden
        hidden = cls._packed_gated_delta_attention(
            layer.linear_attn,
            layer.input_layernorm(hidden),
            sequence_id,
            cu_seqlens,
            cu_seqlens_cpu,
            kernel,
        )
        hidden = residual + hidden
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    @staticmethod
    def _flex_qwen_attention(
        attention: nn.Module,
        noisy_states: torch.Tensor,
        clean_states: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
    ) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        input_shape = noisy_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query, gate = torch.chunk(
            attention.q_proj(noisy_states).view(*input_shape, -1, attention.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        noisy_key = attention.k_norm(attention.k_proj(noisy_states).view(hidden_shape)).transpose(1, 2)
        noisy_value = attention.v_proj(noisy_states).view(hidden_shape).transpose(1, 2)
        query, noisy_key = apply_rotary_pos_emb(query, noisy_key, *noisy_position_embeddings)

        clean_input_shape = clean_states.shape[:-1]
        clean_hidden_shape = (*clean_input_shape, -1, attention.head_dim)
        clean_key = attention.k_norm(attention.k_proj(clean_states).view(clean_hidden_shape)).transpose(1, 2)
        clean_value = attention.v_proj(clean_states).view(clean_hidden_shape).transpose(1, 2)
        # Only the clean key needs RoPE for noisy-to-clean attention.  Reusing
        # it as the dummy query avoids another clean q_proj.
        _, clean_key = apply_rotary_pos_emb(
            clean_key,
            clean_key,
            *clean_position_embeddings,
        )
        repeats = noisy_states.shape[0] // clean_states.shape[0]
        clean_key = clean_key.repeat((repeats, 1, 1, 1))
        clean_value = clean_value.repeat((repeats, 1, 1, 1))
        key = torch.cat([noisy_key, clean_key], dim=2)
        value = torch.cat([noisy_value, clean_value], dim=2)
        with _attention_profile_region("gam_dlm.noisy.flex_attention"):
            output = _compiled_flex_attention(query, key, value, block_mask)
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return attention.o_proj(output)

    @staticmethod
    def _flash_qwen_varlen_self_attention(
        attention: nn.Module,
        states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Qwen clean C2C attention through native varlen FlashAttention-2."""

        from flash_attn import flash_attn_varlen_func
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
        query = query.transpose(1, 2).reshape(-1, query.shape[1], attention.head_dim)
        key = key.transpose(1, 2).reshape(-1, key.shape[1], attention.head_dim)
        value = value.transpose(1, 2).reshape(-1, value.shape[1], attention.head_dim)
        cu = cu_seqlens.to(dtype=torch.int32)
        maximum = int(max_seqlen)
        with _attention_profile_region("gam_dlm.clean.flash_attention_2_varlen"):
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=maximum,
                max_seqlen_k=maximum,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )
        output = output.reshape(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate)
        return attention.o_proj(output)

    def _packed_full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        noisy_block_mask: Any,
        clean_cu_seqlens: torch.Tensor,
        clean_max_seqlen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_residual, clean_residual = noisy, clean
        noisy = self._flex_qwen_attention(
            layer.self_attn,
            layer.input_layernorm(noisy),
            layer.input_layernorm(clean),
            noisy_position_embeddings,
            clean_position_embeddings,
            noisy_block_mask,
        )
        clean = self._flash_qwen_varlen_self_attention(
            layer.self_attn,
            layer.input_layernorm(clean),
            clean_position_embeddings,
            clean_cu_seqlens,
            clean_max_seqlen,
        )
        noisy = noisy_residual + noisy
        clean = clean_residual + clean
        noisy = noisy + layer.mlp(layer.post_attention_layernorm(noisy))
        clean = clean + layer.mlp(layer.post_attention_layernorm(clean))
        return noisy, clean

    def _full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        clean_causal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_residual, clean_residual = noisy, clean
        noisy_norm = layer.input_layernorm(noisy)
        clean_norm = layer.input_layernorm(clean)
        noisy = self._flex_qwen_attention(
            layer.self_attn,
            noisy_norm,
            clean_norm,
            noisy_position_embeddings,
            clean_position_embeddings,
            block_mask,
        )
        clean, _ = layer.self_attn(
            hidden_states=clean_norm,
            position_embeddings=clean_position_embeddings,
            attention_mask=clean_causal_mask,
            past_key_values=None,
        )
        noisy = noisy_residual + noisy
        clean = clean_residual + clean
        noisy_residual, clean_residual = noisy, clean
        noisy = noisy_residual + layer.mlp(layer.post_attention_layernorm(noisy))
        clean = clean_residual + layer.mlp(layer.post_attention_layernorm(clean))
        noisy = noisy * noisy_valid.unsqueeze(-1).to(noisy.dtype)
        clean = clean * clean_valid.unsqueeze(-1).to(clean.dtype)
        return noisy, clean

    def _packed_hybrid_language_forward(
        self,
        streams: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.language_checkpoint_stride:
            raise RuntimeError("packed DLM forbids legacy language_checkpoint_stride")
        if self.gradient_checkpointing and self.packing_activation_cpu_offload:
            raise RuntimeError("packed checkpointing and saved-tensor CPU offload are mutually exclusive")
        noisy, clean = streams["noisy"], streams["clean"]
        offload_this_pack = (
            self.packing_activation_cpu_offload
            and clean.shape[1] >= self.packing_activation_offload_min_tokens
        )
        self._last_packing_activation_cpu_offload = offload_this_pack
        noisy_positions = streams["noisy_positions"]
        clean_positions = streams["clean_positions"]
        if noisy_positions.shape[0] == 4:
            noisy_rotary_positions = noisy_positions[1:]
            clean_rotary_positions = clean_positions[1:]
        else:
            noisy_rotary_positions = noisy_positions
            clean_rotary_positions = clean_positions
        noisy_position_embeddings = self.language_model.rotary_emb(noisy, noisy_rotary_positions)
        clean_position_embeddings = self.language_model.rotary_emb(clean, clean_rotary_positions)
        noisy_block_mask = self._packed_block_mask(
            streams,
            int(self.config.text_config.num_attention_heads),
        )
        for layer_index, layer in enumerate(self.language_model.layers):
            with self._packing_saved_tensor_offload(offload_this_pack):
                if layer.layer_type == "linear_attention":
                    noisy = self._packed_linear_layer(
                        layer,
                        noisy,
                        streams["noisy_sequence_id"],
                        streams["noisy_cu_seqlens"],
                        streams["noisy_cu_seqlens_cpu"],
                        self.packing_linear_kernel,
                    )
                    clean = self._packed_linear_layer(
                        layer,
                        clean,
                        streams["clean_sample_id"],
                        streams["clean_cu_seqlens"],
                        streams["clean_cu_seqlens_cpu"],
                        self.packing_linear_kernel,
                    )
                else:
                    packed_full = (
                        self._checkpoint_packed_full_layer
                        if self._checkpoint_packed_language_layer(layer_index)
                        else self._packed_full_layer
                    )
                    noisy, clean = packed_full(
                        layer,
                        noisy,
                        clean,
                        noisy_position_embeddings,
                        clean_position_embeddings,
                        noisy_block_mask,
                        streams["clean_cu_seqlens"],
                        streams["clean_max_seqlen"],
                    )
        return self.language_model.norm(noisy), self.language_model.norm(clean)

    def _hybrid_language_forward(self, streams: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        noisy, clean = streams["noisy"], streams["clean"]
        noisy_valid, clean_valid = streams["noisy_valid"], streams["clean_valid"]
        noisy_positions = streams["noisy_positions"]
        clean_positions = streams["clean_positions"]
        if noisy_positions.shape[0] == 4:
            noisy_rotary_positions = noisy_positions[1:]
            clean_rotary_positions = clean_positions[1:]
            clean_text_positions = clean_positions[0]
        else:
            noisy_rotary_positions = noisy_positions
            clean_rotary_positions = clean_positions
            clean_text_positions = clean_positions[0]
        noisy_position_embeddings = self.language_model.rotary_emb(noisy, noisy_rotary_positions)
        clean_position_embeddings = self.language_model.rotary_emb(clean, clean_rotary_positions)

        from transformers.masking_utils import create_causal_mask

        clean_cache_position = torch.arange(clean.shape[1], device=clean.device)
        clean_causal_mask = create_causal_mask(
            config=self.language_model.config,
            inputs_embeds=clean,
            attention_mask=clean_valid,
            cache_position=clean_cache_position,
            past_key_values=None,
            position_ids=clean_text_positions,
        )
        block_mask = self._block_mask(
            noisy_valid,
            clean_valid,
            streams["noisy_turn"],
            streams["clean_turn"],
            int(self.config.text_config.num_attention_heads),
        )

        for layer_index, layer in enumerate(self.language_model.layers):
            checkpoint_layer = self._checkpoint_language_layer(layer_index)
            if layer.layer_type == "linear_attention":
                noisy_positions = (
                    noisy_position_embeddings[0],
                    noisy_position_embeddings[1],
                )
                clean_positions = (
                    clean_position_embeddings[0],
                    clean_position_embeddings[1],
                )
                if checkpoint_layer:
                    noisy = self._checkpoint_linear_layer(layer, noisy, noisy_positions, noisy_valid)
                    clean = self._checkpoint_linear_layer(layer, clean, clean_positions, clean_valid)
                else:
                    noisy = layer(
                        noisy,
                        position_embeddings=noisy_positions,
                        attention_mask=noisy_valid,
                        past_key_values=None,
                        use_cache=False,
                    )
                    clean = layer(
                        clean,
                        position_embeddings=clean_positions,
                        attention_mask=clean_valid,
                        past_key_values=None,
                        use_cache=False,
                    )
                noisy = noisy * noisy_valid.unsqueeze(-1).to(noisy.dtype)
                clean = clean * clean_valid.unsqueeze(-1).to(clean.dtype)
            else:
                if checkpoint_layer:
                    noisy, clean = self._checkpoint_full_layer(
                        layer,
                        noisy,
                        clean,
                        noisy_position_embeddings,
                        clean_position_embeddings,
                        block_mask,
                        noisy_valid,
                        clean_valid,
                        clean_causal_mask,
                    )
                else:
                    noisy, clean = self._full_layer(
                        layer,
                        noisy,
                        clean,
                        noisy_position_embeddings,
                        clean_position_embeddings,
                        block_mask,
                        noisy_valid,
                        clean_valid,
                        clean_causal_mask,
                    )
        return self.language_model.norm(noisy), self.language_model.norm(clean)

    def _selective_causal_ce(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels[:, 1:]
        selected = targets.ne(-100)
        if not bool(selected.any()):
            return hidden.sum() * 0.0
        prediction_hidden = hidden[:, :-1][selected]
        target_ids = targets[selected]
        if self.padding_free_packing:
            if self.packing_lm_head_loss_backend == "checkpointed_chunk":
                from torch.utils.checkpoint import checkpoint

                total = prediction_hidden.new_zeros((), dtype=torch.float32)
                chunk_tokens = self.packing_lm_head_loss_chunk_tokens
                weight = self.lm_head.weight
                bias = self.lm_head.bias

                def chunk_loss(
                    chunk_hidden: torch.Tensor,
                    chunk_targets: torch.Tensor,
                    output_weight: torch.Tensor,
                ) -> torch.Tensor:
                    logits = F.linear(chunk_hidden, output_weight, bias)
                    return F.cross_entropy(logits.float(), chunk_targets, reduction="sum")

                for start in range(0, target_ids.numel(), chunk_tokens):
                    end = min(start + chunk_tokens, target_ids.numel())
                    total = total + checkpoint(
                        chunk_loss,
                        prediction_hidden[start:end],
                        target_ids[start:end],
                        weight,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                return total / target_ids.numel()
            from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss

            return fused_linear_cross_entropy_loss(
                prediction_hidden,
                target_ids,
                weight=self.lm_head.weight,
                bias=self.lm_head.bias,
                ignore_index=-100,
                num_chunks=8,
                reduction="mean",
                accumulate_grad_in_fp32=False,
            )
        logits = self.lm_head(prediction_hidden)
        return F.cross_entropy(logits.float(), target_ids, reduction="mean")

    def _build_inference_streams(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clean_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        block_start: int,
    ) -> dict[str, torch.Tensor]:
        """Build one Direct-Conversion draft view for batch-size-one inference.

        ``block_start`` is the known first token of the speculative block.  All
        tokens from there onward form the current noisy turn.  Earlier text is
        retained because Qwen3.5 linear-attention layers carry recurrent state;
        vision placeholders remain clean-stream-only, exactly as in training.
        """

        if input_ids.shape[0] != 1:
            raise ValueError("DLM block drafting currently requires batch size 1")
        valid = attention_mask[0].bool()
        if not bool(valid.all()):
            raise ValueError("DLM block drafting expects an unpadded sequence")
        sequence_length = input_ids.shape[1]
        if not 0 <= block_start < sequence_length:
            raise ValueError(f"invalid block_start={block_start} for length={sequence_length}")

        ids = input_ids[0]
        cfg = self.config
        vision_ids = {
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        }
        text = torch.ones_like(ids, dtype=torch.bool)
        for token_id in vision_ids:
            text &= ids.ne(token_id)

        # The current block is one new turn.  N2N is bidirectional within this
        # turn; strict N2C (q_turn > kv_turn) exposes only completed context.
        clean_turn = torch.zeros_like(ids)
        clean_turn[block_start:] = 1
        noisy_turn = clean_turn[text]
        noisy_ids = ids[text]
        noisy = self.language_model.embed_tokens(noisy_ids).unsqueeze(0)
        noisy_positions = position_ids[:, 0, text].unsqueeze(1)
        return {
            "noisy": noisy,
            "noisy_valid": torch.ones_like(noisy_ids, dtype=torch.bool).unsqueeze(0),
            "noisy_turn": noisy_turn.unsqueeze(0),
            "noisy_positions": noisy_positions,
            "clean": clean_embeds,
            "clean_valid": attention_mask.bool(),
            "clean_turn": clean_turn.unsqueeze(0),
            "clean_positions": position_ids,
        }

    @torch.inference_mode()
    def draft_block_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_start: int,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        precomputed_clean_embeds: torch.Tensor | None = None,
        precomputed_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for every position after ``block_start`` in one NFE.

        The returned tensor contains predictions only for positions strictly
        after ``block_start``.  Token shift matches the training loss: hidden
        state at position ``i-1`` predicts token ``i``.
        """

        was_training = self.training
        self.eval()
        if (precomputed_clean_embeds is None) != (precomputed_position_ids is None):
            raise ValueError("clean embeddings and position ids must be precomputed together")
        if precomputed_clean_embeds is None:
            clean_embeds, position_ids = self._embed_clean(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                mm_token_type_ids,
            )
        else:
            clean_embeds = precomputed_clean_embeds
            position_ids = precomputed_position_ids
            if clean_embeds.shape[:2] != input_ids.shape:
                raise ValueError("precomputed clean embeddings do not match input ids")
            if position_ids.shape[1:] != input_ids.shape:
                raise ValueError("precomputed position ids do not match input ids")
        streams = self._build_inference_streams(
            input_ids,
            attention_mask,
            clean_embeds,
            position_ids,
            block_start,
        )
        dump_streams_path = os.environ.get("GAM_DLM_DUMP_DLM_STREAMS_PATH")
        if dump_streams_path and not os.path.exists(dump_streams_path):
            # Tiny semantic-audit artifact: embeddings/positions before layer
            # 0.  It is opt-in and CPU-only; ordinary inference/training is
            # untouched.  The caller can compare the current block segment
            # with SGLang without serializing the full model state.
            torch.save(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "block_start": int(block_start),
                    "noisy": streams["noisy"].detach().float().cpu(),
                    "clean": streams["clean"].detach().float().cpu(),
                    "noisy_positions": streams["noisy_positions"].detach().cpu(),
                    "clean_positions": streams["clean_positions"].detach().cpu(),
                    "noisy_turn": streams["noisy_turn"].detach().cpu(),
                    "clean_turn": streams["clean_turn"].detach().cpu(),
                },
                dump_streams_path,
            )
        noisy_hidden, _ = self._hybrid_language_forward(streams)

        cfg = self.config
        text = torch.ones_like(input_ids[0], dtype=torch.bool)
        for token_id in (
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        ):
            text &= input_ids[0].ne(token_id)
        text_indices = torch.nonzero(text, as_tuple=False).flatten()
        target_positions = torch.arange(
            block_start + 1,
            input_ids.shape[1],
            device=input_ids.device,
        )
        # map original positions to compact noisy-stream positions
        compact = torch.searchsorted(text_indices, target_positions)
        if not torch.equal(text_indices[compact], target_positions):
            raise RuntimeError("speculative target unexpectedly contains a vision placeholder")
        prediction_hidden = noisy_hidden[0, compact - 1]
        logits = self.lm_head(prediction_hidden)
        if was_training:
            self.train()
        return logits

    @torch.inference_mode()
    def draft_block(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_start: int,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        precomputed_clean_embeds: torch.Tensor | None = None,
        precomputed_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict every position after ``block_start`` in one DLM NFE."""

        logits = self.draft_block_logits(
            input_ids,
            attention_mask,
            block_start,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            precomputed_clean_embeds=precomputed_clean_embeds,
            precomputed_position_ids=precomputed_position_ids,
        )
        return logits.argmax(dim=-1)

    @staticmethod
    def _extend_optional_sequence(value: torch.Tensor | None, count: int, fill: int = 0) -> torch.Tensor | None:
        if value is None:
            return None
        extension = torch.full(
            (value.shape[0], count),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, extension], dim=1)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        history: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply the Transformers greedy repetition-penalty contract."""

        if penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")
        if penalty == 1.0:
            return logits
        penalized = logits.clone()
        seen_scores = torch.gather(penalized, 1, history)
        seen_scores = torch.where(
            seen_scores < 0,
            seen_scores * penalty,
            seen_scores / penalty,
        )
        penalized.scatter_(1, history, seen_scores)
        return penalized

    @staticmethod
    def _generation_position_ids(
        cache_positions: torch.Tensor,
        axes: int,
        batch_size: int,
        rope_deltas: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build Qwen3.5 mRoPE positions for generated text tokens."""

        positions = cache_positions.view(1, 1, -1).expand(axes, batch_size, -1)
        if rope_deltas is None:
            return positions
        deltas = rope_deltas.to(device=cache_positions.device, dtype=cache_positions.dtype)
        deltas = deltas.reshape(batch_size, -1)[:, :1]
        return positions + deltas.view(1, batch_size, 1)

    @torch.inference_mode()
    def causal_generate_full_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...] = (),
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Slow, cache-free greedy oracle used by deployment parity tests."""

        if input_ids.shape[0] != 1:
            raise ValueError("causal oracle currently requires batch size 1")
        if max_new_tokens < 1:
            return input_ids[:, :0]
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")

        prompt_length = int(input_ids.shape[1])
        generated = input_ids
        generated_attention = attention_mask
        stop_ids = set(int(value) for value in stop_token_ids)
        prompt_clean_embeds, prompt_position_ids = self._embed_clean(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
        )
        rope_deltas = getattr(self.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        position_axes = int(prompt_position_ids.shape[0])

        nfe = 0
        while generated.shape[1] - prompt_length < max_new_tokens:
            suffix_ids = generated[:, prompt_length:]
            if suffix_ids.numel():
                suffix_embeds = self.language_model.embed_tokens(suffix_ids)
                clean_embeds = torch.cat([prompt_clean_embeds, suffix_embeds], dim=1)
                suffix_cache_positions = torch.arange(
                    prompt_length,
                    generated.shape[1],
                    dtype=torch.long,
                    device=generated.device,
                )
                suffix_positions = self._generation_position_ids(
                    suffix_cache_positions,
                    position_axes,
                    int(input_ids.shape[0]),
                    rope_deltas,
                )
                position_ids = torch.cat([prompt_position_ids, suffix_positions], dim=2)
            else:
                clean_embeds = prompt_clean_embeds
                position_ids = prompt_position_ids

            cache_position = torch.arange(generated.shape[1], device=generated.device)
            outputs = self.language_model(
                input_ids=None,
                inputs_embeds=clean_embeds,
                position_ids=self._native_position_ids(position_ids),
                attention_mask=generated_attention,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                return_dict=True,
            )
            logits = self.lm_head(outputs.last_hidden_state[:, -1])
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            generated_attention = self._extend_optional_sequence(generated_attention, 1, fill=1)
            nfe += 1
            if int(next_token[0, 0]) in stop_ids:
                break

        response = generated[:, prompt_length : prompt_length + max_new_tokens]
        self._last_generation_stats = {
            "decoding": "causal_full_prefix_oracle",
            "output_tokens": int(response.shape[1]),
            "nfe": nfe,
            "repetition_penalty": repetition_penalty,
        }
        return response

    @torch.inference_mode()
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        inference_block_size: int = 16,
        stop_token_ids: tuple[int, ...] = (),
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cuda_cache_cleanup_interval_tokens: int = 0,
        cuda_cache_cleanup_fraction: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Greedy self-speculative decoding with an exact Qwen3.5 verifier.

        Qwen3.5 interleaves full attention and recurrent GatedDeltaNet layers.
        A multi-token causal chunk is mathematically equivalent to incremental
        decoding, but its convolution/recurrent kernels need not produce the
        same floating-point cache state as one-token cached generation.  Even
        when every immediate argmax agrees, promoting that chunk state can
        therefore change a later near-tied token.  The verifier deliberately
        advances the real cache one token at a time and never promotes a
        speculative chunk cache.  Rejected draft tokens cannot contaminate
        either GatedDeltaNet state or full-attention KV.
        """

        if input_ids.shape[0] != 1:
            raise ValueError("DLM speculative generation currently requires batch size 1")
        if inference_block_size < 2 or inference_block_size > self.block_size:
            raise ValueError(
                f"inference_block_size must be in [2, {self.block_size}], got {inference_block_size}"
            )
        if max_new_tokens < 1:
            return input_ids[:, :0]
        if cuda_cache_cleanup_interval_tokens < 0:
            raise ValueError("cuda_cache_cleanup_interval_tokens must be non-negative")
        if not 0.0 <= cuda_cache_cleanup_fraction <= 1.0:
            raise ValueError("cuda_cache_cleanup_fraction must be in [0, 1]")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")
        if no_repeat_ngram_size == 1 or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be 0 or at least 2")

        prompt_length = input_ids.shape[1]
        generated = input_ids
        generated_attention = attention_mask
        generated_mm_types = mm_token_type_ids
        stop_ids = set(int(value) for value in stop_token_ids)
        accepted_draft_tokens = 0
        drafted_tokens = 0
        draft_nfe = 0
        causal_nfe = 0
        promoted_cache_blocks = 0
        rejection_cache_commits = 0
        sequential_fallback_blocks = 0
        sequential_verifier_blocks = 0
        cuda_cache_cleanups = 0
        last_cache_check_tokens = 0
        peak_cuda_reserved_bytes = 0
        repeated_ngram_suppressions = 0

        def suppress_repeated_ngram(
            logits: torch.Tensor, history: torch.Tensor
        ) -> None:
            nonlocal repeated_ngram_suppressions
            if no_repeat_ngram_size < 2:
                return
            # Match the standard causal no-repeat contract: the prompt is part
            # of the token history, even though only generated tokens are
            # returned.  Slicing it away permits the first repeated n-gram to
            # straddle the prompt/response boundary and made the verifier
            # safety weaker than ordinary causal decoding.
            tokens = history[0].tolist()
            if len(tokens) < no_repeat_ngram_size - 1:
                return
            suffix = tuple(tokens[-(no_repeat_ngram_size - 1) :])
            banned = {
                tokens[index + no_repeat_ngram_size - 1]
                for index in range(len(tokens) - no_repeat_ngram_size + 1)
                if tuple(tokens[index : index + no_repeat_ngram_size - 1])
                == suffix
            }
            for token in banned:
                if 0 <= token < logits.shape[-1] and torch.isfinite(logits[0, token]):
                    logits[0, token] = -torch.inf
                    repeated_ngram_suppressions += 1

        def update_progress(active: bool = True) -> None:
            self._inflight_generation_stats = {
                "active": active,
                "output_tokens": int(generated.shape[1] - prompt_length),
                "nfe": draft_nfe + causal_nfe,
                "draft_nfe": draft_nfe,
                "causal_nfe": causal_nfe,
                "promoted_cache_blocks": promoted_cache_blocks,
                "rejection_cache_commits": rejection_cache_commits,
                "sequential_fallback_blocks": sequential_fallback_blocks,
                "sequential_verifier_blocks": sequential_verifier_blocks,
                "cuda_cache_cleanups": cuda_cache_cleanups,
                "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
            }

        def maybe_release_fragmented_cuda_cache() -> None:
            """Bound allocator fragmentation from monotonically growing prefixes.

            Each draft iteration creates a new full-prefix tensor shape.
            CUDA's caching allocator otherwise retains the old
            blocks even though live allocations stay small.  Releasing only
            when reserved memory is both above a device fraction and more than
            twice live memory is numerically inert and avoids work on normal
            short structured responses.
            """

            nonlocal cuda_cache_cleanups, last_cache_check_tokens, peak_cuda_reserved_bytes
            if cuda_cache_cleanup_interval_tokens == 0 or not generated.is_cuda:
                return
            output_tokens = int(generated.shape[1] - prompt_length)
            if output_tokens - last_cache_check_tokens < cuda_cache_cleanup_interval_tokens:
                return
            last_cache_check_tokens = output_tokens
            reserved = int(torch.cuda.memory_reserved(generated.device))
            allocated = int(torch.cuda.memory_allocated(generated.device))
            peak_cuda_reserved_bytes = max(peak_cuda_reserved_bytes, reserved)
            total = int(torch.cuda.get_device_properties(generated.device).total_memory)
            if reserved < int(total * cuda_cache_cleanup_fraction) or reserved <= 2 * allocated:
                return
            torch.cuda.empty_cache()
            cuda_cache_cleanups += 1

        # Encode vision and exact multimodal positions once.  Subsequent draft
        # iterations append only ordinary text/mask embeddings and positions.
        prompt_clean_embeds, prompt_position_ids = self._embed_clean(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
        )
        rope_deltas = getattr(self.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        position_axes = int(prompt_position_ids.shape[0])
        batch_size = int(input_ids.shape[0])

        prompt_cache_positions = torch.arange(prompt_length, device=input_ids.device)
        causal_outputs = self.language_model(
            input_ids=None,
            inputs_embeds=prompt_clean_embeds,
            position_ids=self._native_position_ids(prompt_position_ids),
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            cache_position=prompt_cache_positions,
            return_dict=True,
        )
        causal_cache = causal_outputs.past_key_values
        if causal_cache is None:
            raise RuntimeError("Qwen3.5 causal prefill did not return a cache")
        first_logits = self.lm_head(causal_outputs.last_hidden_state[:, -1])
        first_logits = self._apply_repetition_penalty(first_logits, generated, repetition_penalty)
        suppress_repeated_ngram(first_logits, generated)
        next_token = first_logits.argmax(dim=-1, keepdim=True)
        causal_nfe += 1
        generated = torch.cat([generated, next_token], dim=1)
        generated_attention = self._extend_optional_sequence(generated_attention, 1, fill=1)
        generated_mm_types = self._extend_optional_sequence(generated_mm_types, 1, fill=0)
        update_progress()

        def cached_causal_logits(tokens_to_commit: torch.Tensor, cache: Any) -> torch.Tensor:
            nonlocal causal_nfe
            cache_start = int(cache.get_seq_length())
            cache_position = torch.arange(
                cache_start,
                cache_start + tokens_to_commit.shape[1],
                dtype=torch.long,
                device=tokens_to_commit.device,
            )
            position_ids = self._generation_position_ids(
                cache_position,
                position_axes,
                batch_size,
                rope_deltas,
            )
            outputs = self.language_model(
                input_ids=tokens_to_commit,
                attention_mask=None,
                position_ids=self._native_position_ids(position_ids),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )
            causal_nfe += 1
            return self.lm_head(outputs.last_hidden_state)

        def exact_verify_block(
            drafted: torch.Tensor,
            remaining: int,
        ) -> tuple[torch.Tensor, int, bool]:
            """Verify and commit against the native one-token cache contract."""

            parts: list[torch.Tensor] = []
            accepted_count = 0
            stopped = False
            commit_token = generated[:, -1:]
            for index in range(drafted.shape[1]):
                history = torch.cat([generated, drafted[:, :index]], dim=1)
                logits = cached_causal_logits(commit_token, causal_cache)[:, -1]
                logits = self._apply_repetition_penalty(logits, history, repetition_penalty)
                suppress_repeated_ngram(logits, history)
                verifier_token = logits.argmax(dim=-1, keepdim=True)
                draft_token = drafted[:, index : index + 1]
                if not torch.equal(verifier_token, draft_token):
                    parts.append(verifier_token)
                    stopped = int(verifier_token[0, 0]) in stop_ids
                    break
                parts.append(draft_token)
                accepted_count += 1
                if int(draft_token[0, 0]) in stop_ids:
                    stopped = True
                    break
                commit_token = draft_token
            else:
                if drafted.shape[1] < remaining:
                    history = torch.cat([generated, drafted], dim=1)
                    logits = cached_causal_logits(commit_token, causal_cache)[:, -1]
                    logits = self._apply_repetition_penalty(logits, history, repetition_penalty)
                    suppress_repeated_ngram(logits, history)
                    verifier_token = logits.argmax(dim=-1, keepdim=True)
                    parts.append(verifier_token)
                    stopped = int(verifier_token[0, 0]) in stop_ids
            return torch.cat(parts, dim=1), accepted_count, stopped

        while generated.shape[1] - prompt_length < max_new_tokens:
            if int(generated[0, -1]) in stop_ids:
                break
            remaining = max_new_tokens - (generated.shape[1] - prompt_length)
            draft_width = min(inference_block_size - 1, remaining)
            if draft_width <= 0:
                break
            block_start = generated.shape[1] - 1
            masks = torch.full(
                (1, draft_width),
                self.mask_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            draft_input = torch.cat([generated, masks], dim=1)
            draft_attention = self._extend_optional_sequence(generated_attention, draft_width, fill=1)
            draft_mm_types = self._extend_optional_sequence(generated_mm_types, draft_width, fill=0)
            draft_suffix_embeds = self.language_model.embed_tokens(draft_input[:, prompt_length:])
            draft_clean_embeds = torch.cat([prompt_clean_embeds, draft_suffix_embeds], dim=1)
            draft_cache_positions = torch.arange(
                prompt_length,
                draft_input.shape[1],
                dtype=torch.long,
                device=draft_input.device,
            )
            draft_suffix_positions = self._generation_position_ids(
                draft_cache_positions,
                position_axes,
                batch_size,
                rope_deltas,
            )
            draft_position_ids = torch.cat([prompt_position_ids, draft_suffix_positions], dim=2)
            drafted = self.draft_block(
                draft_input,
                draft_attention,
                block_start,
                mm_token_type_ids=draft_mm_types,
                precomputed_clean_embeds=draft_clean_embeds,
                precomputed_position_ids=draft_position_ids,
            ).unsqueeze(0)
            draft_nfe += 1
            drafted_tokens += int(drafted.numel())

            # Never promote or commit a multi-token verifier cache.  The
            # accepted prefix is replayed through the exact one-token path so
            # its recurrent/conv state stays identical to cached causal decode.
            append, accepted, terminated = exact_verify_block(drafted, remaining)
            sequential_verifier_blocks += 1
            accepted_draft_tokens += accepted

            generated = torch.cat([generated, append], dim=1)
            generated_attention = self._extend_optional_sequence(generated_attention, append.shape[1], fill=1)
            generated_mm_types = self._extend_optional_sequence(generated_mm_types, append.shape[1], fill=0)

            # Drop references to the old, differently sized prefix tensors
            # before asking the caching allocator to release inactive blocks.
            del (
                masks,
                draft_input,
                draft_attention,
                draft_mm_types,
                draft_suffix_embeds,
                draft_clean_embeds,
                draft_cache_positions,
                draft_suffix_positions,
                draft_position_ids,
                drafted,
                append,
            )
            maybe_release_fragmented_cuda_cache()
            update_progress()
            if terminated:
                break

        response = generated[:, prompt_length : prompt_length + max_new_tokens]
        nfe = draft_nfe + causal_nfe
        self._last_generation_stats = {
            "output_tokens": int(response.shape[1]),
            "nfe": nfe,
            "draft_nfe": draft_nfe,
            "causal_nfe": causal_nfe,
            "promoted_cache_blocks": promoted_cache_blocks,
            "rejection_cache_commits": rejection_cache_commits,
            "sequential_fallback_blocks": sequential_fallback_blocks,
            "sequential_verifier_blocks": sequential_verifier_blocks,
            "drafted_tokens": drafted_tokens,
            "accepted_draft_tokens": accepted_draft_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repeated_ngram_suppressions": repeated_ngram_suppressions,
            "tokens_per_nfe": float(response.shape[1]) / max(nfe, 1),
            "draft_acceptance": float(accepted_draft_tokens) / max(drafted_tokens, 1),
            "causal_cache_tokens": int(causal_cache.get_seq_length()),
            "cuda_cache_cleanups": cuda_cache_cleanups,
            "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
        }
        update_progress(active=False)
        return response

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        patch_positions: torch.Tensor | None = None,
        packing_workload_tokens: int | None = None,
        **_: Any,
    ) -> DLMOutput:
        clean_embeds, position_ids = self._embed_clean(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
            patch_positions,
        )
        streams = self._build_streams(input_ids, labels, attention_mask, clean_embeds, position_ids)
        if self.padding_free_packing:
            streams = self._pack_streams(streams)
            if packing_workload_tokens is not None:
                streams["packing_workload_tokens"] = int(packing_workload_tokens)
            noisy_hidden, clean_hidden = self._packed_hybrid_language_forward(streams)
        else:
            noisy_hidden, clean_hidden = self._hybrid_language_forward(streams)
        mdm_loss = self._selective_causal_ce(noisy_hidden, streams["noisy_labels"])
        if self.casuallossenable:
            causal_loss = self._selective_causal_ce(clean_hidden, streams["clean_labels"])
        else:
            # The clean stream remains part of hybrid attention and therefore
            # still carries gradients from the MDM objective. Only its
            # independent LM-head/CE supervision is skipped.
            causal_loss = mdm_loss.detach().new_zeros(())
        loss = self.mdm_loss_weight * mdm_loss + self.causal_loss_weight * causal_loss
        return DLMOutput(loss=loss, mdm_loss=mdm_loss.detach(), causal_loss=causal_loss.detach())
