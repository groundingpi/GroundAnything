"""MinerU-style full-block sampler for the GAM Direct-Conversion DLM.

The model/attention/KV contract remains the audited Fast-dVLM route.  Only the
reverse-process policy changes: the whole remaining B32 response block is
sampled and positions are committed using a dynamic confidence threshold plus
an exact per-step transfer quota.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from .dlm_sampling import (
    sample_tokens,
    select_transfer_positions,
    transfer_schedule,
)
from .gam_hierarchy import GAMHierarchyBlock
from .compat import fast_eager_context
from infer.decode.decode_ops import shifted_local_logits


class GAMMinerUBlock(GAMHierarchyBlock):
    """Full-B32 MinerU token sampler with deterministic progress guarantees."""

    def __init__(self, config):
        super().__init__(config)
        self.dynamic_threshold = float(
            os.environ.get(
                "GAM_MINERU_DYNAMIC_THRESHOLD",
                config.algorithm_config.get("dynamic_threshold", self.threshold),
            )
        )
        self.requested_max_denoise_steps = int(
            os.environ.get("GAM_MINERU_DENOISE_STEPS", self.max_denoise_steps)
        )
        # ``GAM_MINERU_DENOISE_STEPS`` is a ceiling, not a request to perform
        # redundant forwards after every position has been committed.  This
        # matters for a B8 checkpoint: after its inherited causal anchor, a
        # normal block contains at most seven masks.  Preserve the requested
        # value in the audit while capping the effective schedule by B.
        self.max_denoise_steps = min(
            self.requested_max_denoise_steps, self.block_size
        )
        self._sampling_audit_requests: set[int] = set()
        if not 0.0 <= self.dynamic_threshold <= 1.0:
            raise ValueError("dynamic_threshold must be in [0, 1]")
        if not 1 <= self.requested_max_denoise_steps <= 31:
            raise ValueError(
                "GAM_MINERU_DENOISE_STEPS must be within [1, 31]"
            )

    @staticmethod
    def _request_sampling(
        forward_batch: ForwardBatch,
    ) -> tuple[float, float, int, Optional[int]]:
        info = getattr(forward_batch, "sampling_info", None)
        if info is None or len(info) != 1:
            raise RuntimeError(
                "GAMMinerUBlock requires exactly one request with sampling_info"
            )
        temperature = float(info.temperatures.reshape(-1)[0].item())
        top_p = float(info.top_ps.reshape(-1)[0].item())
        top_k = int(info.top_ks.reshape(-1)[0].item())
        seeds = getattr(info, "sampling_seed", None)
        seed = int(seeds.reshape(-1)[0].item()) if seeds is not None else None
        return temperature, top_p, top_k, seed

    @staticmethod
    def _generator(
        device: torch.device,
        seed: Optional[int],
        absolute_start: int,
        denoise_step: int,
    ) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator = torch.Generator(device=device)
        # Make every response block/denoise step reproducible without sharing
        # random state across requests or depending on server request order.
        mixed = (
            int(seed)
            + 1_000_003 * int(absolute_start)
            + 97_409 * int(denoise_step)
        ) % (2**63 - 1)
        generator.manual_seed(mixed)
        return generator

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        ids = forward_batch.input_ids
        total_len = int(ids.numel())
        initial_mask = ids.eq(self.mask_id)
        num_masked = int(initial_mask.sum().item())
        if num_masked == 0:
            raise RuntimeError("GAM MinerU block contains no mask tokens")
        block_start = total_len - num_masked

        positions = getattr(forward_batch, "positions", None)
        flat_positions = (
            positions.reshape(-1) if isinstance(positions, torch.Tensor) else None
        )
        is_new_request = bool(
            flat_positions is not None
            and flat_positions.numel() > 0
            and int(flat_positions[0].item()) == 0
        )
        if is_new_request:
            self.last_inherited_token = None
            self.last_block_end_position = None
            self._trace_request += 1
            self._trace_step = 0

        if block_start == 0 and self.last_inherited_token is not None and not is_new_request:
            ids[0] = int(self.last_inherited_token)

        remaining_positions = torch.nonzero(
            ids.eq(self.mask_id), as_tuple=False
        ).flatten()
        if remaining_positions.numel() == 0:
            raise RuntimeError("GAM MinerU inheritance left no denoise targets")
        rel_start = int(remaining_positions[0].item())
        rel_end = total_len
        schedule = transfer_schedule(
            int(remaining_positions.numel()), self.max_denoise_steps
        )
        temperature, top_p, top_k, seed = self._request_sampling(forward_batch)
        absolute_start = (
            int(flat_positions[0].item())
            if flat_positions is not None and flat_positions.numel() > 0
            else 0
        )
        self._audit_sampling(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            absolute_start=absolute_start,
            mask_count=int(remaining_positions.numel()),
            schedule=schedule,
        )

        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)
        for denoise_step, minimum_count in enumerate(schedule):
            local_mask = ids[rel_start:rel_end].eq(self.mask_id)
            if not bool(local_mask.any()):
                break
            with fast_eager_context():
                out = model_runner.forward(
                    forward_batch, pp_proxy_tensors=None
                )
            logits_output = out.logits_output
            full_logits = getattr(logits_output, "full_logits", None)
            if full_logits is None:
                raise RuntimeError("GAM MinerU DLM requires full logits")
            local_logits = shifted_local_logits(
                full_logits, rel_start, rel_end, self.token_shift
            )
            if os.environ.get("GAM_SGLANG_DUMP_DLM_LOGITS_PATH"):
                full_shifted = (
                    torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                    if self.token_shift > 0
                    else full_logits
                )
                self._dump_full_logits_if_requested(
                    forward_batch=forward_batch,
                    ids=ids,
                    full_logits=full_logits,
                    shifted=full_shifted,
                    rel_start=rel_start,
                    rel_end=rel_end,
                )
            shifted = local_logits
            predicted, confidence = sample_tokens(
                local_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=self._generator(
                    local_logits.device, seed, absolute_start, denoise_step
                ),
            )
            accepted = select_transfer_positions(
                confidence,
                local_mask,
                threshold=self.dynamic_threshold,
                minimum_count=minimum_count,
            )
            self._trace(
                model_runner,
                forward_batch,
                phase="mineru_denoise",
                rel_start=rel_start,
                rel_end=rel_end,
                ids=ids,
                shifted=shifted,
                confidence=confidence,
                accepted=accepted,
            )
            ids[rel_start:rel_end] = torch.where(
                accepted, predicted, ids[rel_start:rel_end]
            )

        if bool(ids[rel_start:rel_end].eq(self.mask_id).any()):
            raise RuntimeError(
                "GAM MinerU transfer schedule ended with residual mask tokens"
            )

        self._set_attention_type(model_runner, AttentionType.DECODER)
        logits_output = model_runner.forward_extend(
            forward_batch, pp_proxy_tensors=None
        )
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        final_full_logits = getattr(logits_output, "full_logits", None)
        if final_full_logits is not None and final_full_logits.numel() > 0:
            self.last_inherited_token = int(final_full_logits[-1].argmax().item())
            if flat_positions is not None and flat_positions.numel() > 0:
                self.last_block_end_position = int(flat_positions[-1].item())
        return logits_output, ids[block_start:], False

    def _audit_sampling(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        absolute_start: int,
        mask_count: int,
        schedule: tuple[int, ...],
    ) -> None:
        """Persist one small sampler contract row per request when requested."""

        path = os.environ.get("GAM_MINERU_SAMPLING_AUDIT_PATH")
        if not path or self._trace_request in self._sampling_audit_requests:
            return
        self._sampling_audit_requests.add(self._trace_request)
        row = {
            "request": self._trace_request,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "sampling_seed": seed,
            "dynamic_threshold": self.dynamic_threshold,
            "requested_denoise_steps": self.requested_max_denoise_steps,
            "effective_denoise_step_ceiling": self.max_denoise_steps,
            "scheduled_denoise_steps": len(schedule),
            "absolute_start": absolute_start,
            "mask_count": mask_count,
            "transfer_schedule": list(schedule),
        }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


Algorithm = GAMMinerUBlock
