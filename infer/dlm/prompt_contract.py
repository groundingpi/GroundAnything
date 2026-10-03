"""Infer the GAM spatial coordinate arity from canonical request prompts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from grounding_anything.protocol import (
    DENSE_POINT_PROMPT_PREFIX,
    GUI_PROMPT_PREFIX,
    REFER_POINT_PROMPT_PREFIX,
)


_POINT_PROMPT_PREFIXES = (
    DENSE_POINT_PROMPT_PREFIX,
    REFER_POINT_PROMPT_PREFIX,
    GUI_PROMPT_PREFIX,
)


def _user_texts(messages: Iterable[Mapping[str, Any]]) -> Iterable[str]:
    """Yield text parts from user messages without decoding image payloads."""

    for message in messages:
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            yield content
            continue
        if not isinstance(content, (list, tuple)):
            continue
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                yield item["text"]


def infer_coordinate_group_width(messages: Iterable[Mapping[str, Any]]) -> int:
    """Return two for canonical Point/GUI prompts and four otherwise.

    GAM deliberately uses the same ``box_start``/``box_end`` wrapper for point
    and bounding-box answers.  The prompt family is therefore the only
    request-local source of truth for the coordinate tuple arity.
    """

    for text in _user_texts(messages):
        candidate = text.strip()
        if any(candidate.startswith(prefix) for prefix in _POINT_PROMPT_PREFIXES):
            return 2
    return 4
