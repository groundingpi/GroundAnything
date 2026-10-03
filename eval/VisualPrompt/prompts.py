"""Prompt registry for Rex-Omni visual-prompt evaluation."""

from __future__ import annotations

from typing import Sequence


VLM_VISUAL_PROMPT = (
    "One or more reference objects are marked by red bounding boxes. Detect every object "
    "of the same category, including the marked examples. Return only a JSON array. "
    'Each item must be {"bbox_2d": [x1, y1, x2, y2], "label": "object"}. '
    "Use integer coordinates normalized to 0-1000 and return [] if no object matches."
)


def visual_prompt(reference_boxes: Sequence[Sequence[int]], *, gam_mode: bool) -> str:
    if not gam_mode:
        return VLM_VISUAL_PROMPT
    from prompt_mode import build_visual_prompt

    return build_visual_prompt(reference_boxes)
