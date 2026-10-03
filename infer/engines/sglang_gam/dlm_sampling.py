"""Pure-PyTorch token and position sampling for GAM diffusion decoding.

This module intentionally has no SGLang dependency so the numerical contract
can be unit-tested without starting a model server.  The policy follows the
MinerU-Diffusion HF implementation:

* temperature/top-k/top-p choose a token for every still-masked position;
* the sampled-token probability is its confidence;
* all positions above a dynamic threshold are committed when enough exist;
* otherwise a deterministic quota of the most confident positions is used.

There is deliberately no AR repetition penalty here.  A partially denoised
block is not an ordered token history, and MinerU-Diffusion does not apply one.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def transfer_schedule(mask_count: int, denoise_steps: int) -> tuple[int, ...]:
    """Distribute ``mask_count`` commits over ``denoise_steps`` exactly."""

    if mask_count < 1:
        raise ValueError("mask_count must be positive")
    if denoise_steps < 1:
        raise ValueError("denoise_steps must be positive")
    steps = min(mask_count, denoise_steps)
    base, remainder = divmod(mask_count, steps)
    return tuple(base + (index < remainder) for index in range(steps))


def _validate_sampling(temperature: float, top_p: float, top_k: int) -> None:
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and within (0, 1]")
    if top_k < 1:
        raise ValueError("top_k must be at least one")


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled ids and their post-filter probabilities.

    SGLang normalizes an HTTP ``temperature=0`` request to
    ``temperature=1, top_k=1``.  Treating that representation as a one-token
    filtered distribution would incorrectly give every position confidence
    1.0.  The greedy branch therefore takes argmax while retaining confidence
    from the unfiltered model distribution, matching MinerU's temperature-zero
    behavior.
    """

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [positions, vocabulary]")
    _validate_sampling(temperature, top_p, top_k)

    work = logits.float()
    vocabulary = int(work.shape[-1])
    effective_top_k = min(int(top_k), vocabulary)
    greedy = temperature <= 1e-6 or effective_top_k == 1
    if greedy:
        probabilities = torch.softmax(work, dim=-1)
        token_ids = work.argmax(dim=-1)
        confidence = probabilities.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, confidence

    scaled = work / float(temperature)
    if effective_top_k < vocabulary:
        candidate_logits, candidate_ids = torch.topk(
            scaled, k=effective_top_k, dim=-1
        )
    else:
        candidate_logits = scaled
        candidate_ids = None

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            candidate_logits, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
        # Retain the first token crossing the nucleus boundary.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        candidate_logits = torch.empty_like(sorted_logits).scatter(
            -1, sorted_indices, sorted_logits
        )

    probabilities = torch.softmax(candidate_logits, dim=-1)
    sampled_local = torch.multinomial(
        probabilities, num_samples=1, generator=generator
    )
    confidence = probabilities.gather(-1, sampled_local).squeeze(-1)
    if candidate_ids is None:
        token_ids = sampled_local.squeeze(-1)
    else:
        token_ids = candidate_ids.gather(-1, sampled_local).squeeze(-1)
    return token_ids, confidence


def select_transfer_positions(
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    minimum_count: int,
) -> torch.Tensor:
    """Select positions using MinerU's dynamic-threshold-plus-quota rule."""

    if confidence.ndim != 1 or mask.shape != confidence.shape:
        raise ValueError("confidence and mask must be equal one-dimensional tensors")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    remaining = int(mask.sum().item())
    if remaining < 1:
        raise ValueError("at least one masked position is required")
    quota = min(max(int(minimum_count), 1), remaining)

    masked_confidence = torch.where(
        mask, confidence, torch.full_like(confidence, -torch.inf)
    )
    accepted = mask & (masked_confidence > float(threshold))
    if int(accepted.sum().item()) >= quota:
        return accepted

    indices = torch.topk(masked_confidence, k=quota).indices
    accepted = torch.zeros_like(mask)
    accepted[indices] = True
    return accepted
