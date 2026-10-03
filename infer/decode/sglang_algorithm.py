"""SGLang GAM B32 decoder with configurable reliability and token choice.

The outer B32/sub-block architecture and causal KV commit remain compatible
with the trained model.  Only the block-internal commit policy is replaced.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang_gam.compat import fast_eager_context
from infer.decode.decode_ops import shifted_local_logits

from .config import DecodeConfig, request_decode_mapping
from .reliability import (
    adjusted_entropy,
    most_reliable_position,
    normal_acceptance,
    raw_reliability,
    raw_reliability_masked,
)
from .repetition import repetition_risks
from .sampling import best_non_token, sample_tokens
from .structure import pending_eos_cut
from .telemetry import BlockTelemetry, telemetry_from_environment


class GAMDecodeV2Block(DllmAlgorithm):
    """Fixed B32 decoder whose block policy is fully request-configurable."""

    def __init__(self, config):
        super().__init__(config)
        eos = os.environ.get("GAM_SGLANG_IM_END_TOKEN_ID")
        eos_token_id = int(eos) if eos else None
        self.base_config = DecodeConfig.from_mappings(
            config.algorithm_config,
            eos_token_id=eos_token_id,
        )
        expected = int(os.environ.get("GAM_SGLANG_BLOCK_SIZE", "32"))
        if self.block_size != expected or self.base_config.block_size != expected:
            raise ValueError(
                "GAMDecodeV2Block block contract mismatch: "
                f"dllm={self.block_size} yaml={self.base_config.block_size} expected={expected}"
            )
        self.last_inherited_token: int | None = None
        self.last_block_end_position: int | None = None
        self.request_index = 0
        self.block_index = 0
        self.telemetry = telemetry_from_environment()

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType) -> None:
        model = model_runner.model
        language = getattr(model, "model", model)
        layers = getattr(language, "layers", None)
        if layers is None:
            raise RuntimeError("GAM SGLang model exposes no language layers")
        # The attention type is a layer attribute consumed by the selected
        # backend.  It only changes at the draft/commit boundary; avoiding a
        # Python walk over every K3/Qwen layer on every repeated denoise
        # forward removes scheduler overhead without changing the mask.
        if getattr(model_runner, "_gam_decode_attention_type", None) == attn_type:
            return
        for layer in layers:
            attention = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attention is not None:
                attention.attn_type = attn_type
        model_runner._gam_decode_attention_type = attn_type

    @staticmethod
    def _request(forward_batch: ForwardBatch) -> Any | None:
        reqs = getattr(forward_batch, "reqs", None)
        return reqs[0] if reqs and len(reqs) == 1 else None

    def _request_config(self, forward_batch: ForwardBatch) -> DecodeConfig:
        custom = None
        info = getattr(forward_batch, "sampling_info", None)
        custom_batch = getattr(info, "custom_params", None)
        if isinstance(custom_batch, list) and len(custom_batch) == 1:
            custom = custom_batch[0]
        if custom is None:
            req = self._request(forward_batch)
            sampling_params = getattr(req, "sampling_params", None)
            custom = getattr(sampling_params, "custom_params", None)
        mapping = request_decode_mapping(custom)
        config = DecodeConfig.from_mappings(self.base_config.to_dict(), mapping)
        if config.block_size != self.block_size:
            raise RuntimeError(
                "request cannot change the loaded DLM block width: "
                f"request={config.block_size} server={self.block_size}"
            )
        return config

    @staticmethod
    def _request_id(forward_batch: ForwardBatch, fallback: int) -> str:
        req = GAMDecodeV2Block._request(forward_batch)
        return str(getattr(req, "rid", None) or getattr(req, "request_id", None) or fallback)

    @staticmethod
    def _align_dllm_positions_before_graph(
        model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> None:
        """Apply GAM Qwen3 absolute positions before CUDA Graph input copy.

        The model wrapper performs the same normalization in eager mode, but
        graph replay never re-enters that Python wrapper.  Keeping the repair
        here makes DecodeV4 eager and graph use identical RoPE positions while
        remaining a no-op for models without GAM's alignment helper.
        """

        positions = getattr(forward_batch, "positions", None)
        align = getattr(model_runner.model, "_align_extend_positions", None)
        if align is None or not isinstance(positions, torch.Tensor):
            return
        aligned = align(positions, forward_batch, forward_batch.input_ids)
        forward_batch.positions = aligned.to(
            device=positions.device,
            dtype=positions.dtype,
        )

    @staticmethod
    def _history(forward_batch: ForwardBatch, ids: torch.Tensor) -> list[int]:
        req = GAMDecodeV2Block._request(forward_batch)
        history = list(getattr(req, "output_ids", None) or [])
        # The prefill anchor is already in output_ids and is also position zero
        # of the first DLM workspace.  Do not count it twice for repetition.
        if history and ids.numel() and int(history[-1]) == int(ids[0].item()):
            history.pop()
        return [int(item) for item in history]

    @staticmethod
    def _generator(
        config: DecodeConfig,
        forward_batch: ForwardBatch,
        request_index: int,
        rel_start: int,
        step: int,
    ) -> torch.Generator | None:
        seed = config.sampling_seed
        if seed is None:
            info = getattr(forward_batch, "sampling_info", None)
            seeds = getattr(info, "sampling_seed", None)
            if isinstance(seeds, torch.Tensor) and seeds.numel() == 1:
                seed = int(seeds.reshape(-1)[0].item())
        if seed is None:
            return None
        generator = torch.Generator(device=forward_batch.input_ids.device)
        generator.manual_seed(
            (int(seed) + request_index * 1_000_003 + rel_start * 97_409 + step * 17)
            % (2**63 - 1)
        )
        return generator

    @staticmethod
    def _eos_allowed(
        *,
        config: DecodeConfig,
        candidate: torch.Tensor,
        top1: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
        eos_probability: torch.Tensor | None,
        stability: torch.Tensor | None,
    ) -> torch.Tensor:
        allowed = torch.ones_like(candidate, dtype=torch.bool)
        eos_id = config.eos_token_id
        if not config.enable_eos_early_stop or eos_id is None:
            return allowed
        is_eos = candidate.eq(int(eos_id))
        if config.eos_require_top1:
            allowed &= ~is_eos | top1.eq(int(eos_id))
        if config.eos_confidence_threshold is not None:
            assert eos_probability is not None
            allowed &= ~is_eos | eos_probability.ge(config.eos_confidence_threshold)
        if config.eos_entropy_threshold is not None:
            allowed &= ~is_eos | entropy.le(config.eos_entropy_threshold)
        if stability is None:
            raise RuntimeError("EOS gate requires stability tracking")
        allowed &= ~is_eos | stability.ge(config.eos_stability_steps)
        return allowed

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool]:
        ids = forward_batch.input_ids
        total_len = int(ids.numel())
        initial_mask = ids.eq(self.mask_id)
        num_masked = int(initial_mask.sum().item())
        if num_masked == 0:
            raise RuntimeError("GAM decode workspace contains no mask tokens")
        block_start = total_len - num_masked

        positions = getattr(forward_batch, "positions", None)
        flat_positions = positions.reshape(-1) if isinstance(positions, torch.Tensor) else None
        is_new_request = bool(
            flat_positions is not None
            and flat_positions.numel() > 0
            and int(flat_positions[0].item()) == 0
        )
        if is_new_request:
            self.last_inherited_token = None
            self.last_block_end_position = None
            self.request_index += 1
            self.block_index = 0
        else:
            self.block_index += 1

        if block_start == 0 and self.last_inherited_token is not None and not is_new_request:
            ids[0] = int(self.last_inherited_token)

        # CUDA Graph copies ``forward_batch.positions`` before replay and does
        # not execute GAM's Python model boundary, where eager mode normally
        # converts local B32 offsets into absolute Qwen3 positions.
        self._align_dllm_positions_before_graph(model_runner, forward_batch)

        config = self._request_config(forward_batch)
        request_id = self._request_id(forward_batch, self.request_index)
        block_stats = BlockTelemetry(
            request_id=request_id,
            request_index=self.request_index,
            block_index=self.block_index,
            task=config.task,
            profile=config.profile,
        )
        telemetry_enabled = block_stats.enabled
        # In the normal deterministic DecodeV4 profile EOS is disabled and
        # repetition/cardinality policies are off.  Do not ask the reliability
        # kernel for an EOS column or allocate stability buffers in that case.
        # These switches are explicit and leave calibration/EOS-enabled routes
        # on the original full-statistics path.
        # Detail/summary telemetry historically reports EOS stability too, so
        # retain that state when telemetry is enabled.  The benchmark-only
        # ``off`` mode takes the allocation-free path.
        track_stability = bool(config.enable_eos_early_stop or telemetry_enabled)
        need_eos_stats = telemetry_enabled or track_stability
        reliability_eos_id = config.eos_token_id if need_eos_stats else None
        fast_confidence = (
            os.environ.get("GAM_DLM_FAST_RELIABILITY", "0") == "1"
            and config.acceptance_policy == "confidence"
            and not need_eos_stats
        )
        history = self._history(forward_batch, ids)
        commit_end = total_len
        eos_accepted = False
        pending_eos_position: int | None = None

        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)
        # The scheduler uses this bit to select its captured DLLM graph on the
        # next dispatch. Returning a hard-coded ``False`` silently disabled
        # graph replay for every DecodeV4 block. The final causal KV commit is
        # still eager, but denoise eligibility must be propagated.
        can_run_cuda_graph = False
        remaining = torch.nonzero(ids.eq(self.mask_id), as_tuple=False).flatten()
        if remaining.numel() == 0:
            raise RuntimeError("GAM inheritance left no denoise targets")
        denoise_start = int(remaining[0].item())

        for rel_start in range(denoise_start, total_len, config.sub_block_size):
            if eos_accepted:
                break
            rel_end = min(rel_start + config.sub_block_size, total_len)
            width = rel_end - rel_start
            previous_candidate = (
                torch.full((width,), -1, dtype=torch.long, device=ids.device)
                if track_stability
                else None
            )
            stability = (
                torch.zeros((width,), dtype=torch.long, device=ids.device)
                if track_stability
                else None
            )
            step = 0
            safety_drain = False

            while bool(
                ids[
                    rel_start : (
                        min(rel_end, pending_eos_position + 1)
                        if pending_eos_position is not None
                        else rel_end
                    )
                ]
                .eq(self.mask_id)
                .any()
            ):
                if step >= config.denoise_steps:
                    if config.legacy_force_fill_all:
                        safety_drain = True
                    else:
                        # A non-default denoise_steps < sub-block width is
                        # allowed, but it drains with one commit per fresh
                        # forward rather than silently force-filling all masks.
                        safety_drain = True

                # Request-scoped eager fast path: suppresses only the vendored
                # diagnostic sync/print around this forward, without patching
                # ModelRunner globally during worker startup.
                with fast_eager_context():
                    out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                can_run_cuda_graph = bool(getattr(out, "can_run_graph", False))
                full_logits = getattr(out.logits_output, "full_logits", None)
                if full_logits is None:
                    raise RuntimeError("GAM decoder requires full logits")
                local_logits = shifted_local_logits(
                    full_logits, rel_start, rel_end, config.token_shift
                )
                local_mask = ids[rel_start:rel_end].eq(self.mask_id)
                if pending_eos_position is not None:
                    tail_start = max(pending_eos_position + 1 - rel_start, 0)
                    local_mask[tail_start:] = False
                if os.environ.get("GAM_DLM_ACTIVE_ROW_COMPACTION", "1") == "1":
                    reliability = raw_reliability_masked(
                        local_logits,
                        local_mask,
                        reliability_eos_id,
                        need_entropy=not fast_confidence,
                    )
                else:
                    reliability = raw_reliability(
                        local_logits,
                        reliability_eos_id,
                        need_entropy=not fast_confidence,
                    )
                deterministic_choice = (
                    config.temperature <= 0
                    and config.top_p >= 1.0
                    and config.top_k <= 0
                )
                if deterministic_choice:
                    # Reliability already computed the raw-logit top-1.  Reuse
                    # it instead of a second full-vocabulary argmax/cast;
                    # this is exactly the same greedy choice for the raw path.
                    candidate = reliability.top1_token
                else:
                    generator = self._generator(
                        config, forward_batch, self.request_index, rel_start, step
                    )
                    candidate = sample_tokens(
                        local_logits,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        generator=generator,
                    )
                if track_stability:
                    assert previous_candidate is not None and stability is not None
                    same = local_mask & candidate.eq(previous_candidate)
                    stability = torch.where(
                        same,
                        stability + 1,
                        torch.where(
                            local_mask, torch.ones_like(stability), stability
                        ),
                    )
                    previous_candidate = torch.where(
                        local_mask, candidate, previous_candidate
                    )

                risks = repetition_risks(
                    history=history,
                    current_ids=ids,
                    candidate_tokens=candidate,
                    mask_id=self.mask_id,
                    global_start=rel_start,
                    config=config,
                )
                adjusted = adjusted_entropy(reliability.entropy, risks, config)
                accepted = normal_acceptance(reliability, adjusted, local_mask, config)
                eos_allowed = self._eos_allowed(
                    config=config,
                    candidate=candidate,
                    top1=reliability.top1_token,
                    confidence=reliability.confidence,
                    entropy=reliability.entropy,
                    eos_probability=reliability.eos_probability,
                    stability=stability,
                )

                eos_id = reliability_eos_id
                if eos_id is not None:
                    eos_candidates = local_mask & candidate.eq(int(eos_id))
                    if telemetry_enabled:
                        block_stats.eos_top1_count += int(
                            (local_mask & reliability.top1_token.eq(int(eos_id)))
                            .sum()
                            .item()
                        )
                        block_stats.eos_candidate_count += int(
                            eos_candidates.sum().item()
                        )
                    assert reliability.eos_probability is not None
                    if telemetry_enabled:
                        block_stats.observe_eos(
                            eos_probability=reliability.eos_probability,
                            entropy=reliability.entropy,
                            local_mask=local_mask,
                            eos_candidates=eos_candidates,
                            stability=stability,
                        )

                if config.enable_eos_early_stop and eos_id is not None:
                    accepted &= eos_allowed
                    if telemetry_enabled:
                        block_stats.eos_rejected_count += int(
                            (eos_candidates & ~eos_allowed).sum().item()
                        )

                    # A reliable EOS may be committed before earlier positions
                    # in the same diffusion sub-block are resolved.  Keep it
                    # pending, denoise only its prefix, then cut immediately
                    # when that prefix becomes ordered.  Rejecting it here can
                    # make the EOS disappear on the next forward and lengthen
                    # generation compared with the native scheduler path.
                    for local_tensor in torch.nonzero(
                        accepted & candidate.eq(int(eos_id)), as_tuple=False
                    ).flatten():
                        local = int(local_tensor.item())
                        earlier_global = ids[:rel_start]
                        if bool(earlier_global.eq(self.mask_id).any()):
                            accepted[local] = False
                            if telemetry_enabled:
                                block_stats.eos_rejected_count += 1
                        else:
                            absolute = rel_start + local
                            if (
                                pending_eos_position is None
                                or absolute < pending_eos_position
                            ):
                                pending_eos_position = absolute
                            # Tokens after pending EOS are drafts outside the
                            # ordered output and must never be committed.
                            accepted[local + 1 :] = False
                            break

                if telemetry_enabled:
                    block_stats.observe_reliability(
                        reliability.confidence,
                        reliability.entropy,
                        local_mask,
                        adjusted,
                    )
                    block_stats.repetition_trigger_count += int(
                        (risks > 0).sum().item()
                    )
                    if risks.numel():
                        block_stats.max_repetition_score = max(
                            block_stats.max_repetition_score, float(risks.max().item())
                        )

                normal_count = int(accepted.sum().item())
                if safety_drain and config.legacy_force_fill_all:
                    accepted = local_mask.clone()
                    if telemetry_enabled:
                        block_stats.forced_all_tokens += int(accepted.sum().item())
                elif not bool(accepted.any()):
                    force_mask = local_mask.clone()
                    # A gated EOS must not become forced-one.  Replace that
                    # candidate by its best non-EOS alternative and preserve
                    # progress without violating the gate.
                    index = most_reliable_position(
                        reliability, adjusted, force_mask, config
                    )
                    if (
                        config.enable_eos_early_stop
                        and eos_id is not None
                        and int(candidate[index].item()) == int(eos_id)
                    ):
                        # EOS is never a progress fallback.  It must pass the
                        # normal reliability policy plus its own gate; otherwise
                        # a low-reliability forced-one could terminate a request.
                        candidate[index] = best_non_token(
                            local_logits[index : index + 1], int(eos_id)
                        )[0]
                    accepted[index] = True
                    if telemetry_enabled:
                        block_stats.forced_one_tokens += 1
                else:
                    if telemetry_enabled:
                        block_stats.normal_accept_tokens += normal_count

                ids[rel_start:rel_end] = torch.where(
                    accepted, candidate, ids[rel_start:rel_end]
                )
                if telemetry_enabled:
                    block_stats.accepted_per_step.append(int(accepted.sum().item()))

                if (
                    config.enable_eos_early_stop
                    and eos_id is not None
                    and pending_eos_position is not None
                ):
                    cut = pending_eos_cut(
                        ids_after_commit=ids,
                        pending_eos_position=pending_eos_position,
                        mask_id=self.mask_id,
                        eos_token_id=int(eos_id),
                    )
                    if cut is not None:
                        commit_end = cut
                        ids[commit_end:] = int(self.mask_id)
                        eos_accepted = True
                        if telemetry_enabled:
                            block_stats.eos_accepted_count += 1
                            block_stats.first_eos_position = commit_end - 1
                            block_stats.finish_reason = "eos_early_cut"
                step += 1
                if step > self.block_size * 2:
                    raise RuntimeError("GAM decoder safety drain failed to make progress")

        # Physical SGLang allocation remains B32.  On early EOS the discarded
        # tail contains MASK placeholders; only the ordered prefix is returned
        # to the scheduler, which releases overallocated KV immediately when
        # the native EOS stop finishes the request.
        self._set_attention_type(model_runner, AttentionType.DECODER)
        logits_output = model_runner.forward_extend(forward_batch, pp_proxy_tensors=None)
        if isinstance(logits_output, tuple):
            logits_output = logits_output[0]
        final_logits = getattr(logits_output, "full_logits", None)
        if final_logits is not None and final_logits.numel() > 0:
            anchor_row = max(min(commit_end - 1, int(final_logits.shape[0]) - 1), 0)
            self.last_inherited_token = int(final_logits[anchor_row].argmax().item())
            if flat_positions is not None and flat_positions.numel() > anchor_row:
                self.last_block_end_position = int(flat_positions[anchor_row].item())
        returned = ids[block_start:commit_end]
        if telemetry_enabled:
            block_stats.generated_tokens = int(returned.numel())
            self.telemetry.append(block_stats.row(config.to_dict()))
        return logits_output, returned, can_run_cuda_graph


Algorithm = GAMDecodeV2Block
