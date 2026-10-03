"""GAM-safe block algorithm built on Fast-dLLM's SGLang hooks.

The vendored ``HierarchyBlock`` is a useful reference, but its original loop
silently drops a tail shorter than ``sub_block_size`` and can perform more
denoise iterations than the trained B32 contract.  This adapter keeps the
same encoder-only denoise / decoder KV-commit sequence while making both
boundaries explicit and deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from .dlm_sampling import sample_tokens
from .compat import fast_eager_context
from infer.decode.decode_ops import shifted_local_logits


class GAMHierarchyBlock(DllmAlgorithm):
    """Fixed-width GAM block denoising with a bounded confidence schedule."""

    def __init__(self, config):
        super().__init__(config)
        self.threshold = float(config.algorithm_config.get("threshold", 0.90))
        self.sub_block_size = int(config.algorithm_config.get("sub_block_size", 4))
        self.max_denoise_steps = int(config.algorithm_config.get("denoise_steps", 8))
        self.token_shift = int(config.algorithm_config.get("token_shift", 1))
        # Fast-dLLM carries the causal prediction at the end of one block into
        # position zero of the next all-mask block.  Keep this state per
        # server worker; it is reset at the next request boundary.
        self.last_inherited_token: Optional[int] = None
        self.last_block_end_position: Optional[int] = None
        self._trace_step = 0
        self._trace_request = 0
        self._dumped_request = None
        # Disabled for the production hierarchy.  The isolated reliable
        # subclass opts in without changing any existing route.
        self.reliable_decoding = False
        self.reliable_request_sampling = False
        self.reliable_min_stability = 1
        self.reliable_eos_token_id: Optional[int] = None
        self.reliable_temperature: Optional[float] = None
        self.reliable_top_p: Optional[float] = None
        self.reliable_top_k: Optional[int] = None
        expected_block_size = int(os.environ.get("GAM_SGLANG_BLOCK_SIZE", "32"))
        if self.block_size != expected_block_size:
            raise ValueError(
                "GAM DLM block contract mismatch: "
                f"expected={expected_block_size}, got={self.block_size}"
            )
        if not 1 <= self.sub_block_size <= self.block_size:
            raise ValueError(
                f"sub_block_size must be in [1, {self.block_size}]"
            )
        if not 1 <= self.max_denoise_steps <= 31:
            raise ValueError("denoise_steps must be in [1, 31]")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType) -> None:
        model = model_runner.model
        language = getattr(model, "model", model)
        layers = getattr(language, "layers", None)
        if layers is None:
            raise RuntimeError("GAM SGLang model exposes no language layers")
        # The type is a layer attribute and is unchanged across all denoise
        # forwards in a block.  Cache it per runner to avoid walking every K3
        # layer at each block boundary; request boundaries still explicitly
        # set DECODER in the caller.
        if getattr(model_runner, "_gam_hierarchy_attention_type", None) == attn_type:
            return
        for layer in layers:
            attention = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attention is not None:
                attention.attn_type = attn_type
        model_runner._gam_hierarchy_attention_type = attn_type

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool]:
        ids = forward_batch.input_ids
        total_len = int(ids.numel())
        mask = ids.eq(self.mask_id)
        num_masked = int(mask.sum().item())
        if num_masked == 0:
            raise RuntimeError("GAM DLM block contains no mask tokens")
        # Compute this before inheritance.  The first scheduler block is
        # [AR] + [MASK] * 31 and returns 31 tokens; later blocks are all-mask
        # and return all 32 positions after position zero is inherited.
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

        # The reference implementation performs this substitution only for
        # an all-mask block.  Do not recompute block_start after replacing the
        # first mask: the scheduler's output accounting depends on the
        # original all-mask length.
        if block_start == 0 and self.last_inherited_token is not None and not is_new_request:
            ids[0] = int(self.last_inherited_token)

        if os.environ.get("GAM_SGLANG_DEBUG_DLM") == "1":
            print(
                "[GAMHierarchyBlock] "
                f"len={total_len} masks={num_masked} block_start={block_start} "
                f"new={is_new_request} inherited={self.last_inherited_token} "
                f"ids={ids.detach().cpu().tolist()} "
                f"positions={flat_positions.detach().cpu().tolist() if flat_positions is not None else None} "
                f"prefix={getattr(forward_batch, 'extend_prefix_lens_cpu', None)} "
                f"extend={getattr(forward_batch, 'extend_seq_lens_cpu', None)}",
                flush=True,
            )

        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)
        # Propagate the runtime's graph-eligibility bit to the scheduler.
        # Previously this adapter always returned ``False`` even when the
        # denoise forward was captured/replayable, forcing the scheduler to
        # treat every GAM block as eager.  This flag is telemetry/scheduling
        # only; it does not alter logits, masks, or token acceptance.
        can_run_cuda_graph = False
        # Position zero is the causal AR anchor (also on later blocks after
        # inheritance).  With token_shift=1 its hidden row predicts position
        # one, so the first sub-block contains mask positions [1, 1+S), not
        # [0, S).  Grouping the anchor itself used to make the first group one
        # target short and shift every later hierarchy boundary relative to
        # the Transformers GAM decoder/training loss.
        remaining_mask_positions = torch.nonzero(
            ids.eq(self.mask_id), as_tuple=False
        ).flatten()
        if remaining_mask_positions.numel() == 0:
            raise RuntimeError("GAM DLM inheritance left no denoise targets")
        denoise_start = int(remaining_mask_positions[0].item())
        for rel_start in range(denoise_start, total_len, self.sub_block_size):
            rel_end = min(rel_start + self.sub_block_size, total_len)
            steps = 0
            if self.reliable_decoding:
                previous_prediction = torch.full(
                    (rel_end - rel_start,), -1, dtype=torch.long, device=ids.device
                )
                stability = torch.zeros(
                    (rel_end - rel_start,), dtype=torch.long, device=ids.device
                )
            else:
                # Keep the established hierarchy route free of extra device
                # allocations; reliable decoding is an isolated opt-in route.
                previous_prediction = None
                stability = None
            # ``hierarchy_dynamic`` accepts at most one pass per position in
            # the audited reference implementation.  Capping a sub-block's
            # loop by its width prevents an extra denoise pass from changing
            # the token trajectory while retaining the bounded fallback for
            # low-confidence early checkpoints.
            if self.reliable_decoding:
                step_limit = min(
                    self.max_denoise_steps, 2 * (rel_end - rel_start)
                )
            else:
                step_limit = min(self.max_denoise_steps, rel_end - rel_start)
            while bool(ids[rel_start:rel_end].eq(self.mask_id).any()) and steps < step_limit:
                with fast_eager_context():
                    out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                can_run_cuda_graph = bool(getattr(out, "can_run_graph", False))
                logits_output = out.logits_output
                full_logits = getattr(logits_output, "full_logits", None)
                if full_logits is None:
                    raise RuntimeError("GAM SGLang DLM requires full logits from LogitsProcessor")
                # Slice after applying the token-shift index.  Materializing a
                # full [B, vocab] concat here costs ~20 MB per forward while
                # the hierarchy route consumes only this sub-block.
                shifted = shifted_local_logits(
                    full_logits, rel_start, rel_end, self.token_shift
                )
                if os.environ.get("GAM_SGLANG_DUMP_DLM_LOGITS_PATH"):
                    # The debug snapshot intentionally retains the complete
                    # shifted tensor; production requests never enter this
                    # branch and keep the sliced allocation above.
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
                local_logits = shifted
                if self.reliable_decoding and self.reliable_request_sampling:
                    temperature, top_p, top_k, seed = self._reliable_sampling_params(
                        forward_batch
                    )
                    generator = None
                    if seed is not None:
                        generator = torch.Generator(device=local_logits.device)
                        generator.manual_seed(
                            (
                                int(seed)
                                + 1_000_003 * int(rel_start)
                                + 97_409 * int(steps)
                                + 17 * int(self._trace_request)
                            )
                            % (2**63 - 1)
                        )
                    pred, prob = sample_tokens(
                        local_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        generator=generator,
                    )
                else:
                    pred = local_logits.argmax(dim=-1)
                    prob = local_logits.softmax(dim=-1).amax(dim=-1)
                local_mask = ids[rel_start:rel_end].eq(self.mask_id)
                confidence = torch.where(local_mask, prob, torch.zeros_like(prob))
                if self.reliable_decoding:
                    assert previous_prediction is not None and stability is not None
                    same = local_mask & pred.eq(previous_prediction)
                    stability = torch.where(
                        same,
                        stability + 1,
                        torch.where(local_mask, torch.ones_like(stability), stability),
                    )
                    previous_prediction = torch.where(
                        local_mask, pred, previous_prediction
                    )
                self._trace(
                    model_runner,
                    forward_batch,
                    phase="denoise",
                    rel_start=rel_start,
                    rel_end=rel_end,
                    ids=ids,
                    shifted=shifted,
                    confidence=confidence,
                    accepted=None,
                )
                unmask = local_mask & (confidence >= self.threshold)
                if self.reliable_decoding:
                    assert stability is not None
                    unmask &= stability.ge(self.reliable_min_stability)
                    unmask = self._gate_reliable_eos(
                        ids=ids,
                        pred=pred,
                        proposed=unmask,
                        stability=stability,
                        rel_start=rel_start,
                    )
                if not bool(unmask.any()):
                    if not self.reliable_decoding or steps + 1 >= step_limit:
                        # Reliable decoding permits an observation-only step;
                        # the bounded last-step fallback still guarantees
                        # progress for every sub-block.
                        candidate = torch.where(
                            local_mask, confidence, torch.zeros_like(confidence)
                        ).argmax()
                        unmask[candidate] = True
                ids[rel_start:rel_end] = torch.where(
                    unmask, pred, ids[rel_start:rel_end]
                )
                self._trace_last_accept(unmask)
                if os.environ.get("GAM_SGLANG_DEBUG_DLM") == "1":
                    print(
                        "[GAMHierarchyBlock.step] "
                        f"start={rel_start} end={rel_end} step={steps} "
                        f"pred={pred.detach().cpu().tolist()} "
                        f"conf={[round(float(v), 6) for v in confidence.detach().cpu().tolist()]} "
                        f"accepted={unmask.detach().cpu().tolist()} "
                        f"ids={ids[rel_start:rel_end].detach().cpu().tolist()}",
                        flush=True,
                    )
                steps += 1

            # A final deterministic argmax closes any residual masks after the
            # bounded schedule.  This is only a safety fallback, not a sample-
            # dependent strategy change.
            if bool(ids[rel_start:rel_end].eq(self.mask_id).any()):
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                can_run_cuda_graph = bool(getattr(out, "can_run_graph", False))
                full_logits = getattr(out.logits_output, "full_logits", None)
                if full_logits is None:
                    raise RuntimeError("full logits disappeared during GAM fallback")
                shifted = shifted_local_logits(
                    full_logits, rel_start, rel_end, self.token_shift
                )
                fallback_logits = shifted
                pred = fallback_logits.argmax(dim=-1)
                local_mask = ids[rel_start:rel_end].eq(self.mask_id)
                if self.reliable_decoding and self.reliable_eos_token_id is not None:
                    assert stability is not None
                    unsafe = (
                        local_mask
                        & pred.eq(self.reliable_eos_token_id)
                        & stability.lt(self.reliable_min_stability)
                    )
                    if bool(unsafe.any()):
                        adjusted = fallback_logits.clone()
                        adjusted[unsafe, self.reliable_eos_token_id] = -torch.inf
                        pred = torch.where(unsafe, adjusted.argmax(dim=-1), pred)
                self._trace(
                    model_runner,
                    forward_batch,
                    phase="fallback",
                    rel_start=rel_start,
                    rel_end=rel_end,
                    ids=ids,
                    shifted=shifted,
                    confidence=None,
                    accepted=None,
                )
                ids[rel_start:rel_end] = torch.where(local_mask, pred, ids[rel_start:rel_end])

        # Switch back to causal attention before committing the block to KV.
        # ``forward_extend`` deliberately bypasses the CUDA graph captured for
        # encoder-only denoising, exactly as in Fast-dVLM's reference path.
        self._set_attention_type(model_runner, AttentionType.DECODER)
        logits_output = model_runner.forward_extend(forward_batch, pp_proxy_tensors=None)
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        final_full_logits = getattr(logits_output, "full_logits", None)
        if final_full_logits is not None and final_full_logits.numel() > 0:
            # Must come from the causal KV-commit pass, not a denoising pass.
            self.last_inherited_token = int(final_full_logits[-1].argmax().item())
            if flat_positions is not None and flat_positions.numel() > 0:
                self.last_block_end_position = int(flat_positions[-1].item())
        # Leave the model in the normal causal state after a completed block.
        # The preceding call already set DECODER; avoid a second walk over all
        # transformer layers on every block.
        return logits_output, ids[block_start:], can_run_cuda_graph

    def _reliable_sampling_params(
        self, forward_batch: ForwardBatch
    ) -> tuple[float, float, int, Optional[int]]:
        """Read one request's sampler contract with explicit route overrides."""

        info = getattr(forward_batch, "sampling_info", None)
        if info is None or len(info) != 1:
            raise RuntimeError(
                "GAMReliableBlock requires exactly one request with sampling_info"
            )
        temperature = (
            self.reliable_temperature
            if self.reliable_temperature is not None
            else float(info.temperatures.reshape(-1)[0].item())
        )
        top_p = (
            self.reliable_top_p
            if self.reliable_top_p is not None
            else float(info.top_ps.reshape(-1)[0].item())
        )
        top_k = (
            self.reliable_top_k
            if self.reliable_top_k is not None
            else int(info.top_ks.reshape(-1)[0].item())
        )
        seeds = getattr(info, "sampling_seed", None)
        seed = int(seeds.reshape(-1)[0].item()) if seeds is not None else None
        return float(temperature), float(top_p), max(int(top_k), 1), seed

    def _gate_reliable_eos(
        self,
        *,
        ids: torch.Tensor,
        pred: torch.Tensor,
        proposed: torch.Tensor,
        stability: torch.Tensor,
        rel_start: int,
    ) -> torch.Tensor:
        """Delay EOS until it is stable and all earlier local masks can commit.

        This is not a task-format rule: it only enforces left-to-right
        completion ordering for the model's own EOS prediction.  Coordinate,
        OCR and dense tokens remain unconstrained.
        """

        eos_id = self.reliable_eos_token_id
        if eos_id is None:
            return proposed
        gated = proposed.clone()
        eos_positions = torch.nonzero(
            proposed & pred.eq(int(eos_id)), as_tuple=False
        ).flatten()
        for local_index_tensor in eos_positions:
            local_index = int(local_index_tensor.item())
            earlier = ids[rel_start : rel_start + local_index]
            earlier_ready = bool(
                (~earlier.eq(self.mask_id) | proposed[:local_index]).all()
            )
            if (
                int(stability[local_index].item()) < self.reliable_min_stability
                or not earlier_ready
            ):
                gated[local_index] = False
        return gated

    def _trace(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        *,
        phase: str,
        rel_start: int,
        rel_end: int,
        ids: torch.Tensor,
        shifted: torch.Tensor,
        confidence: Optional[torch.Tensor],
        accepted: Optional[torch.Tensor],
    ) -> None:
        """Write a bounded, opt-in token trajectory for semantic debugging."""
        path = os.environ.get("GAM_SGLANG_TRACE_PATH")
        limit = int(os.environ.get("GAM_SGLANG_TRACE_MAX_STEPS", "24"))
        if not path or self._trace_step >= limit:
            return
        self._trace_step += 1
        local = shifted[rel_start:rel_end]
        top_k = min(5, local.shape[-1])
        values, indices = torch.topk(local.float(), k=top_k, dim=-1)
        row = {
            "request": self._trace_request,
            "step": self._trace_step,
            "phase": phase,
            "rel_start": rel_start,
            "rel_end": rel_end,
            "input_ids": ids.detach().cpu().tolist(),
            "positions": getattr(forward_batch, "positions", torch.empty(0)).detach().cpu().reshape(-1).tolist(),
            "extend_prefix_lens": getattr(forward_batch, "extend_prefix_lens", torch.empty(0)).detach().cpu().reshape(-1).tolist(),
            "backend": type(getattr(forward_batch, "attn_backend", None)).__name__,
            "attention_type": str(getattr(getattr(model_runner.model.model.layers[0].self_attn, "attn", None), "attn_type", None)),
            "top_ids": indices.detach().cpu().tolist(),
            "top_logits": values.detach().cpu().tolist(),
            "confidence": confidence.detach().float().cpu().tolist() if confidence is not None else None,
            "accepted": accepted.detach().cpu().tolist() if accepted is not None else None,
        }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _trace_last_accept(self, accepted: torch.Tensor) -> None:
        """Attach acceptance to the most recent trace row without extra GPU work."""
        path = os.environ.get("GAM_SGLANG_TRACE_PATH")
        if not path:
            return
        # The trajectory is intentionally append-only; a separate tiny record
        # keeps the hot path free of read/modify/write filesystem operations.
        with open(path + ".accept", "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"request": self._trace_request, "step": self._trace_step, "accepted": accepted.detach().cpu().tolist()}) + "\n")

    def _dump_full_logits_if_requested(
        self,
        *,
        forward_batch: ForwardBatch,
        ids: torch.Tensor,
        full_logits: torch.Tensor,
        shifted: torch.Tensor,
        rel_start: int,
        rel_end: int,
    ) -> None:
        """Persist one bounded full-logit snapshot for an offline parity audit.

        The hook is opt-in and request-scoped because a 32x152k tensor is
        intentionally too large for the normal trajectory JSON.  It is an
        inference diagnostic only; no production request pays the copy unless
        ``GAM_SGLANG_DUMP_DLM_LOGITS_PATH`` is explicitly set.
        """
        path = os.environ.get("GAM_SGLANG_DUMP_DLM_LOGITS_PATH")
        if not path:
            return
        target_request = int(os.environ.get("GAM_SGLANG_DUMP_DLM_REQUEST", "2"))
        if self._trace_request != target_request or self._dumped_request == target_request:
            return
        payload = {
            "request": self._trace_request,
            "rel_start": rel_start,
            "rel_end": rel_end,
            "input_ids": ids.detach().cpu(),
            "positions": getattr(forward_batch, "positions", torch.empty(0)).detach().cpu(),
            "extend_prefix_lens": getattr(forward_batch, "extend_prefix_lens", torch.empty(0)).detach().cpu(),
            "extend_seq_lens": getattr(forward_batch, "extend_seq_lens", torch.empty(0)).detach().cpu(),
            "full_logits": full_logits.detach().float().cpu(),
            "shifted_logits": shifted.detach().float().cpu(),
        }
        import torch as _torch

        _torch.save(payload, path)
        self._dumped_request = target_request

class GAMReliableBlock(GAMHierarchyBlock):
    """Hierarchy decoding with real request sampling and temporal commit gates."""

    def __init__(self, config):
        super().__init__(config)
        self.reliable_decoding = True
        self.reliable_request_sampling = (
            os.environ.get("GAM_RELIABLE_REQUEST_SAMPLING", "1") == "1"
        )
        self.reliable_min_stability = int(
            os.environ.get("GAM_RELIABLE_MIN_STABILITY", "2")
        )
        eos = os.environ.get("GAM_SGLANG_IM_END_TOKEN_ID")
        self.reliable_eos_token_id = int(eos) if eos is not None else None
        temperature = os.environ.get("GAM_RELIABLE_TEMPERATURE")
        top_p = os.environ.get("GAM_RELIABLE_TOP_P")
        top_k = os.environ.get("GAM_RELIABLE_TOP_K")
        self.reliable_temperature = (
            float(temperature) if temperature is not None else None
        )
        self.reliable_top_p = float(top_p) if top_p is not None else None
        self.reliable_top_k = int(top_k) if top_k is not None else None
        if not 1 <= self.reliable_min_stability <= 4:
            raise ValueError("GAM_RELIABLE_MIN_STABILITY must be within [1, 4]")


Algorithm = GAMHierarchyBlock
