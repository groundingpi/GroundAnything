"""Ordered-history repetition risks for the GAM DLM path."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from .config import DecodeConfig


def _clean(tokens: Iterable[int], ignored: set[int]) -> list[int]:
    return [int(token) for token in tokens if int(token) not in ignored]


def _ngram_risk(sequence: Sequence[int], candidate: int, size: int) -> float:
    formed = list(sequence) + [int(candidate)]
    if len(formed) < size:
        return 0.0
    target = tuple(formed[-size:])
    earlier = formed[:-1]
    return float(
        any(tuple(earlier[index : index + size]) == target for index in range(len(earlier) - size + 1))
    )


def _block_risk(sequence: Sequence[int], candidate: int, maximum: int) -> float:
    formed = list(sequence) + [int(candidate)]
    limit = min(maximum, len(formed) // 2)
    if limit < 2:
        return 0.0
    best = 0
    for width in range(2, limit + 1):
        suffix = formed[-width:]
        search = formed[:-width]
        if any(search[index : index + width] == suffix for index in range(len(search) - width + 1)):
            best = width
    return best / float(maximum)


def _coordinate_tuples(tokens: Sequence[int], config: DecodeConfig) -> set[tuple[int, ...]]:
    if (
        config.coord_token_min_id is None
        or config.coord_token_max_id is None
        or config.coordinates_per_target is None
    ):
        return set()
    coordinates = [
        int(token)
        for token in tokens
        if config.coord_token_min_id <= int(token) <= config.coord_token_max_id
    ]
    width = int(config.coordinates_per_target)
    return {
        tuple(coordinates[index : index + width])
        for index in range(0, len(coordinates) - width + 1, width)
    }


def repetition_risks(
    *,
    history: Sequence[int],
    current_ids: torch.Tensor,
    candidate_tokens: torch.Tensor,
    mask_id: int,
    global_start: int,
    config: DecodeConfig,
) -> torch.Tensor:
    """Score every local candidate against history plus committed B32 tokens."""

    risks = torch.zeros(candidate_tokens.numel(), device=candidate_tokens.device)
    if config.repetition_mode == "none" or config.repetition_weight == 0:
        return risks
    ignored = set(config.repetition_ignore_token_ids) | {int(mask_id)}
    if config.eos_token_id is not None:
        ignored.add(int(config.eos_token_id))
    history_tail = list(history)[-config.repetition_window :]
    for local_index, candidate_tensor in enumerate(candidate_tokens):
        candidate = int(candidate_tensor.item())
        if candidate in ignored:
            continue
        before = [
            int(token)
            for token in current_ids[: global_start + local_index].detach().cpu().tolist()
            if int(token) != int(mask_id)
        ]
        sequence = _clean(history_tail + before, ignored)[-config.repetition_window :]
        if config.repetition_mode == "ngram":
            risk = _ngram_risk(sequence, candidate, config.repetition_ngram_size)
        else:
            risk = _block_risk(sequence, candidate, config.repetition_block_size)

        # Exact coordinate tuples are much stronger evidence than repeated
        # wrapper/punctuation tokens, which are excluded above.
        if (
            config.coord_token_min_id is not None
            and config.coord_token_max_id is not None
            and config.coordinates_per_target is not None
            and config.coord_token_min_id <= candidate <= config.coord_token_max_id
        ):
            prior_coordinates = _coordinate_tuples(sequence, config)
            coordinate_stream = [
                int(token)
                for token in sequence + [candidate]
                if config.coord_token_min_id <= int(token) <= config.coord_token_max_id
            ]
            width = int(config.coordinates_per_target)
            if len(coordinate_stream) >= width and len(coordinate_stream) % width == 0:
                completed = tuple(coordinate_stream[-width:])
                if completed in prior_coordinates:
                    risk = 1.0
        risks[local_index] = min(max(float(risk), 0.0), 1.0)
    return risks
