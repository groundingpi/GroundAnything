"""Token filtering and selection policies for HierarchyBlock denoising."""

from __future__ import annotations

import math

import torch


def filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Apply temperature, top-k and top-p in the production generation order.

    Repetition penalty is intentionally applied by the caller before this
    function.  A zero temperature keeps the logits unscaled and selects via
    argmax; top-k/top-p are still well-defined hard masks in that mode.
    """

    if logits.ndim != 2:
        raise ValueError(f"expected [positions, vocabulary] logits, got {tuple(logits.shape)}")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_k is not None and (isinstance(top_k, bool) or int(top_k) <= 0):
        raise ValueError("top_k must be None or a positive integer")

    filtered = logits.float().clone()
    if temperature > 0.0:
        filtered.div_(temperature)

    if top_k is not None and int(top_k) < filtered.shape[-1]:
        threshold = torch.topk(filtered, k=int(top_k), dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, -torch.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        # Keep the first token crossing the nucleus threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits.masked_fill_(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            -1, sorted_indices, sorted_logits
        )

    if not bool(torch.isfinite(filtered).any(dim=-1).all()):
        raise RuntimeError("logit filtering removed every token from a position")
    return filtered


def predictions_and_confidence(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return greedy or sampled predictions and their normalized probability."""

    filtered = filter_logits(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    probabilities = torch.softmax(filtered, dim=-1)
    predictions = (
        filtered.argmax(dim=-1)
        if temperature == 0.0
        else torch.multinomial(probabilities, num_samples=1).squeeze(1)
    )
    confidence = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
    return predictions, confidence


def step_acceptance(
    confidence: torch.Tensor,
    masked: torch.Tensor,
    *,
    step_index: int,
    total_steps: int,
) -> torch.Tensor:
    """Evenly exhaust masks over a fixed number of denoising NFEs.

    At each NFE the most confident ``ceil(masks / remaining_steps)`` positions
    are accepted. The final configured step therefore resolves every position.
    """

    if confidence.ndim != 1 or masked.shape != confidence.shape:
        raise ValueError("confidence and masked must be equal one-dimensional tensors")
    if not 0 <= step_index < total_steps:
        raise ValueError("step_index must be within total_steps")
    remaining = int(masked.sum().item())
    accepted = torch.zeros_like(masked)
    if remaining == 0:
        return accepted
    remaining_steps = total_steps - step_index
    count = min(remaining, math.ceil(remaining / remaining_steps))
    scores = confidence.masked_fill(~masked, -torch.inf)
    indices = torch.topk(scores, k=count).indices
    accepted[indices] = True
    return accepted


def dynamic_acceptance(
    confidence: torch.Tensor,
    masked: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Accept tokens above threshold and guarantee at least one-token progress."""

    if confidence.ndim != 1 or masked.shape != confidence.shape:
        raise ValueError("confidence and masked must be equal one-dimensional tensors")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    accepted = masked & confidence.gt(threshold)
    if bool(masked.any()) and not bool(accepted.any()):
        scores = confidence.masked_fill(~masked, -torch.inf)
        accepted[scores.argmax()] = True
    return accepted


def entropy_acceptance(
    entropy: torch.Tensor,
    masked: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """DecodeV3 acceptance on raw-logit entropy with forced-one progress."""

    if entropy.ndim != 1 or masked.shape != entropy.shape:
        raise ValueError("entropy and masked must be equal one-dimensional tensors")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    accepted = masked & entropy.le(threshold)
    if bool(masked.any()) and not bool(accepted.any()):
        scores = entropy.masked_fill(~masked, torch.inf)
        accepted[scores.argmin()] = True
    return accepted
