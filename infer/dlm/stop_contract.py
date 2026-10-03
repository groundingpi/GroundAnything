"""Fail-closed OpenAI stop normalization for GAM DLM inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def normalize_openai_stop(stop: object) -> tuple[str, ...]:
    """Normalize the OpenAI string-or-list stop contract."""

    if stop is None:
        return ()
    if isinstance(stop, str):
        values = [stop]
    elif isinstance(stop, Sequence) and not isinstance(stop, (bytes, bytearray)):
        values = list(stop)
    else:
        raise TypeError("OpenAI stop must be a string or a sequence of strings")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("every OpenAI stop entry must be a string")
    values = [value for value in values if value]
    if len(values) > 4:
        raise ValueError("OpenAI stop supports at most four sequences")
    return tuple(values)


def single_token_stop_ids(
    tokenizer: Any, stop: object
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Tokenize stop strings and reject unsupported multi-token sequences.

    The formal GAM contract uses double-newline, which is one atomic Qwen3.5
    token. Rejecting other shapes prevents the server from silently accepting
    a stop request that the decoder cannot enforce during generation.
    """

    normalized = normalize_openai_stop(stop)
    token_ids: list[int] = []
    for value in normalized:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"DLM runtime stop must encode to exactly one token: {value!r} -> {encoded}"
            )
        token_ids.append(int(encoded[0]))
    return normalized, tuple(token_ids)


def token_stop_sequences(
    tokenizer: Any, stop: object
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Tokenize OpenAI stop strings without assuming one-token boundaries.

    OpenAI ``stop`` is a decoded-text contract.  A tokenizer may encode the
    complete string atomically while the model can still emit an equivalent
    sequence through smaller tokens (for example ``"\n\n"`` as either token
    271 or two newline tokens).  Include the canonical encoding, every
    two-piece split and the character-wise encoding; every candidate is
    decode-verified before it is accepted.
    """

    normalized = normalize_openai_stop(stop)
    sequences: list[tuple[int, ...]] = []

    def add(value: str, pieces: list[str]) -> None:
        encoded = tuple(
            int(token)
            for piece in pieces
            for token in tokenizer.encode(piece, add_special_tokens=False)
        )
        if not encoded or encoded in sequences:
            return
        decoded = tokenizer.decode(
            list(encoded),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
        if decoded == value:
            sequences.append(encoded)

    for value in normalized:
        add(value, [value])
        for split in range(1, len(value)):
            add(value, [value[:split], value[split:]])
        if len(value) > 1:
            add(value, list(value))
    return normalized, tuple(sequences)


def first_token_sequence_match(
    token_ids: Sequence[int], sequences: Sequence[Sequence[int]]
) -> tuple[int, int] | None:
    """Return the earliest ``[start, end)`` stop-sequence match."""

    best: tuple[int, int] | None = None
    values = [int(token) for token in token_ids]
    for sequence in sequences:
        candidate = tuple(int(token) for token in sequence)
        if not candidate:
            continue
        width = len(candidate)
        for start in range(0, len(values) - width + 1):
            if tuple(values[start : start + width]) != candidate:
                continue
            match = (start, start + width)
            if best is None or match < best:
                best = match
            break
    return best
