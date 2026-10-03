"""Token-choice branch for GAM diffusion decoding.

Reliability is deliberately absent from this module.  Sampling filters may
change the committed token, but can never change raw confidence or entropy.
"""

from __future__ import annotations

import torch


def sample_tokens(
    raw_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if raw_logits.ndim != 2:
        raise ValueError("raw_logits must have shape [positions, vocabulary]")
    logits = raw_logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / float(temperature)
    vocabulary = int(logits.shape[-1])
    if 0 < top_k < vocabulary:
        values, indices = torch.topk(logits, k=top_k, dim=-1)
    else:
        values, indices = logits, None

    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(values, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_values, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_values = sorted_values.masked_fill(remove, -torch.inf)
        values = torch.empty_like(sorted_values).scatter(
            -1, sorted_indices, sorted_values
        )

    probabilities = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probabilities, 1, generator=generator)
    if indices is not None:
        sampled = indices.gather(-1, sampled)
    return sampled.squeeze(-1)


def best_non_token(raw_logits: torch.Tensor, banned_token_id: int) -> torch.Tensor:
    """Return row-wise argmax after excluding one token without mutating input."""

    adjusted = raw_logits.float().clone()
    adjusted[:, int(banned_token_id)] = -torch.inf
    return adjusted.argmax(dim=-1)
