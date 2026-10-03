"""VLM/GAM prompt registry for Rex-Omni document-layout benchmarks."""

from __future__ import annotations

from typing import Iterable


VLM_LAYOUT_PROMPT = (
    "Detect every document layout element belonging to these categories: {categories}. "
    "Return only a JSON array. Each item must be "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "exact category name"}}. '
    "Use integer coordinates normalized to 0-1000, preserve the exact requested label, "
    "output one item per instance, and return [] if nothing matches."
)


def layout_prompt(categories: Iterable[str], *, gam_mode: bool) -> str:
    labels = [str(category) for category in categories]
    if gam_mode:
        from prompt_mode import build_layout_prompt

        return build_layout_prompt(labels)
    return VLM_LAYOUT_PROMPT.format(categories=", ".join(labels))
