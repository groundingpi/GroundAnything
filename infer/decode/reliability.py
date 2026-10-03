"""Raw-logit reliability statistics and commit policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DecodeConfig


@dataclass(frozen=True)
class RawReliability:
    top1_token: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    eos_probability: torch.Tensor | None = None


def raw_reliability(
    raw_logits: torch.Tensor,
    eos_token_id: int | None = None,
    *,
    need_entropy: bool = True,
) -> RawReliability:
    """Compute calibrated statistics at T=1 before any sampling filter.

    ``need_entropy=False`` is an explicit fast path for the common greedy
    confidence policy.  Confidence is still computed from the *unmodified*
    logits (max-logit minus log-sum-exp); entropy is returned as a zero tensor
    because callers that select this path have proved that they do not read
    it.  Keeping this switch here, rather than changing the default
    implementation, preserves the calibration path bit-for-bit for existing
    evaluators and RL code.
    """

    logits = raw_logits.float()
    max_logits, top1 = logits.max(dim=-1)
    log_normalizer = torch.logsumexp(logits, dim=-1)
    confidence = (max_logits - log_normalizer).exp()
    if need_entropy:
        log_probs = logits - log_normalizer.unsqueeze(-1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
    else:
        entropy = torch.zeros_like(confidence)
    eos_probability = (
        (logits[:, int(eos_token_id)] - log_normalizer).exp()
        if eos_token_id is not None
        else None
    )
    return RawReliability(
        top1_token=top1,
        confidence=confidence,
        entropy=entropy,
        eos_probability=eos_probability,
    )


def raw_reliability_masked(
    raw_logits: torch.Tensor,
    active_mask: torch.Tensor,
    eos_token_id: int | None = None,
    *,
    need_entropy: bool = True,
) -> RawReliability:
    """Compute raw statistics only for writable rows, then scatter them back.

    During DecodeV4 denoising, committed positions remain in the fixed B32
    workspace but their reliability is never consulted.  Running a full
    vocabulary ``log_softmax`` for those rows is pure work (and becomes the
    dominant cost near the end of a sub-block).  This helper preserves the
    exact per-row operation for active positions and returns zero placeholders
    elsewhere; all callers mask those placeholders before making a decision.

    The all-active fast path avoids an index/gather allocation on the first
    forward of each sub-block.  Set ``GAM_DLM_ACTIVE_ROW_COMPACTION=0`` in the
    launcher to reproduce the original full-width calculation.
    """

    mask = active_mask.to(dtype=torch.bool).reshape(-1)
    if mask.numel() != raw_logits.shape[0]:
        raise ValueError(
            "active_mask must have one entry per logit row: "
            f"{mask.numel()} != {raw_logits.shape[0]}"
        )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == raw_logits.shape[0]:
        return raw_reliability(
            raw_logits, eos_token_id, need_entropy=need_entropy
        )
    if indices.numel() == 0:
        raise ValueError("raw_reliability_masked requires at least one active row")

    active = raw_reliability(
        raw_logits.index_select(0, indices),
        eos_token_id,
        need_entropy=need_entropy,
    )
    width = raw_logits.shape[0]
    top1 = torch.zeros(
        (width,), dtype=active.top1_token.dtype, device=raw_logits.device
    )
    confidence = torch.zeros(
        (width,), dtype=active.confidence.dtype, device=raw_logits.device
    )
    entropy = torch.zeros(
        (width,), dtype=active.entropy.dtype, device=raw_logits.device
    )
    top1.index_copy_(0, indices, active.top1_token)
    confidence.index_copy_(0, indices, active.confidence)
    entropy.index_copy_(0, indices, active.entropy)
    eos_probability = None
    if active.eos_probability is not None:
        eos_probability = torch.zeros(
            (width,),
            dtype=active.eos_probability.dtype,
            device=raw_logits.device,
        )
        eos_probability.index_copy_(0, indices, active.eos_probability)
    return RawReliability(
        top1_token=top1,
        confidence=confidence,
        entropy=entropy,
        eos_probability=eos_probability,
    )


def adjusted_entropy(
    raw_entropy: torch.Tensor,
    repetition_risk: torch.Tensor,
    config: DecodeConfig,
) -> torch.Tensor:
    # Confidence acceptance never consumes adjusted entropy.  Avoid creating
    # an arange (and a device-side add) on every denoise forward in the
    # overwhelmingly common deterministic DecodeV4 profile.
    if config.acceptance_policy == "confidence":
        return raw_entropy
    if (
        config.acceptance_policy == "entropy"
        and config.repetition_weight == 0
    ):
        return raw_entropy
    offsets = torch.arange(
        raw_entropy.numel(), device=raw_entropy.device, dtype=raw_entropy.dtype
    )
    value = raw_entropy
    if config.acceptance_policy in {
        "entropy_position",
        "entropy_position_repetition",
    }:
        value = value + float(config.position_penalty) * offsets
    if config.acceptance_policy == "entropy_position_repetition":
        value = value + float(config.repetition_weight) * repetition_risk.to(value)
    return value


def normal_acceptance(
    reliability: RawReliability,
    adjusted: torch.Tensor,
    mask: torch.Tensor,
    config: DecodeConfig,
) -> torch.Tensor:
    if config.acceptance_policy == "confidence":
        accepted = reliability.confidence >= float(config.confidence_threshold)
    else:
        assert config.entropy_threshold is not None
        accepted = adjusted <= float(config.entropy_threshold)
    return accepted & mask


def most_reliable_position(
    reliability: RawReliability,
    adjusted: torch.Tensor,
    mask: torch.Tensor,
    config: DecodeConfig,
) -> int:
    if not bool(mask.any()):
        raise ValueError("forced-one requires a masked position")
    if config.acceptance_policy == "confidence":
        scores = torch.where(
            mask,
            reliability.confidence,
            torch.full_like(reliability.confidence, -torch.inf),
        )
        return int(scores.argmax().item())
    scores = torch.where(mask, adjusted, torch.full_like(adjusted, torch.inf))
    return int(scores.argmin().item())
