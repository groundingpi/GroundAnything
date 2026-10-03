"""Pure ordered-output helpers for EOS and grounding cardinality."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def ordered_eos_cut(
    *,
    ids_after_commit: torch.Tensor,
    accepted: torch.Tensor,
    candidates: torch.Tensor,
    rel_start: int,
    mask_id: int,
    eos_token_id: int,
) -> int | None:
    """Return exclusive cut after the first accepted, prefix-ordered EOS."""

    eos_locations = torch.nonzero(
        accepted & candidates.eq(int(eos_token_id)), as_tuple=False
    ).flatten()
    for local_tensor in eos_locations:
        absolute = rel_start + int(local_tensor.item())
        if not bool(ids_after_commit[:absolute].eq(int(mask_id)).any()):
            return absolute + 1
    return None


def pending_eos_cut(
    *,
    ids_after_commit: torch.Tensor,
    pending_eos_position: int,
    mask_id: int,
    eos_token_id: int,
) -> int | None:
    """Cut a previously committed EOS once its ordered prefix is complete."""

    position = int(pending_eos_position)
    if not 0 <= position < int(ids_after_commit.numel()):
        raise ValueError("pending EOS position is outside the decode workspace")
    if int(ids_after_commit[position].item()) != int(eos_token_id):
        raise ValueError("pending EOS position no longer contains EOS")
    if bool(ids_after_commit[:position].eq(int(mask_id)).any()):
        return None
    return position + 1


def cardinality_cutoff(
    output_ids: Sequence[int], contract: Mapping[str, int]
) -> int | None:
    """Return the exclusive end of the requested coordinate tuple count."""

    start_token = int(contract["box_start_token_id"])
    coord_min = int(contract["coord_token_min_id"])
    coord_max = int(contract["coord_token_max_id"])
    width = int(contract["coordinate_limit"])
    cardinality = int(contract.get("expected_cardinality", 1))
    if width not in (2, 4) or cardinality < 1:
        raise ValueError("invalid grounding cardinality contract")
    try:
        start = list(output_ids).index(start_token)
    except ValueError:
        return None
    coordinates = [
        index
        for index in range(start + 1, len(output_ids))
        if coord_min <= int(output_ids[index]) <= coord_max
    ]
    required = width * cardinality
    return coordinates[required - 1] + 1 if len(coordinates) >= required else None
