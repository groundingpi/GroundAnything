"""Allocation-saving index operations shared by the GAM decoders.

The reference adapters materialize ``cat([logits[:1], logits[:-1]])`` for
every denoise forward.  With a Qwen vocabulary this copies tens of megabytes
even when DecodeV4 consumes only a four-row sub-block.  These helpers preserve
the exact token-shift indexing while slicing first.
"""

from __future__ import annotations

import torch


def shifted_local_logits(
    full_logits: torch.Tensor,
    start: int,
    end: int,
    token_shift: int,
) -> torch.Tensor:
    """Return a slice of shifted logits without materializing the full slice."""

    if full_logits.ndim != 2:
        raise ValueError("full_logits must have shape [tokens, vocabulary]")
    if not 0 <= int(start) <= int(end) <= int(full_logits.shape[0]):
        raise ValueError(
            f"invalid logits slice [{start}, {end}) for {full_logits.shape[0]} rows"
        )
    if not token_shift:
        return full_logits[start:end]
    if start > 0:
        # shifted[i] == full_logits[i-1] for every i>0.
        return full_logits[start - 1 : end - 1]
    width = end - start
    if width <= 1:
        return full_logits[:1]
    # Only the first sub-block can start at zero; this concat is at most the
    # sub-block width rather than the complete B32 workspace.
    return torch.cat((full_logits[:1], full_logits[: width - 1]), dim=0)


def shifted_argmax(full_logits: torch.Tensor, token_shift: int) -> torch.Tensor:
    """Argmax of shifted logits without a full-vocabulary concat."""

    if full_logits.ndim != 2:
        raise ValueError("full_logits must have shape [tokens, vocabulary]")
    if not token_shift:
        return full_logits.argmax(dim=-1)
    predictions = torch.empty(
        (full_logits.shape[0],), dtype=torch.long, device=full_logits.device
    )
    predictions[0] = full_logits[0].argmax()
    if full_logits.shape[0] > 1:
        predictions[1:] = full_logits[:-1].argmax(dim=-1)
    return predictions

