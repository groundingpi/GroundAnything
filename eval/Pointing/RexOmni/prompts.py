"""VLM/GAM prompt registry for Rex-Omni Point-in-mask benchmarks."""

from __future__ import annotations

from typing import Iterable


VLM_DENSE_POINT_PROMPT = (
    "Point to the center of every visible object belonging to these categories: "
    "{categories}. Return only a JSON array. Each item must be "
    '{{"point_2d": [x, y], "label": "exact category name"}}. '
    "Use integer coordinates normalized to 0-1000, output one point per instance, "
    "and return [] if nothing matches."
)
VLM_REFER_POINT_PROMPT = (
    'Point to every visible target matching the description "{description}". '
    "Return only a JSON array. Each item must be "
    '{{"point_2d": [x, y], "label": "{description}"}}. '
    "Use integer coordinates normalized to 0-1000 and return [] if no target matches."
)


def point_prompt(
    categories: Iterable[str], *, referring: bool, gam_mode: bool
) -> str:
    labels = [str(value) for value in categories]
    if not labels:
        raise ValueError("Point-in-mask prompt requires at least one category")
    if referring and len(labels) != 1:
        raise ValueError("referring Point-in-mask prompt requires exactly one description")
    if gam_mode:
        from prompt_mode import build_dense_point_prompt, build_refer_point_prompt

        return (
            build_refer_point_prompt(labels[0])
            if referring
            else build_dense_point_prompt(labels)
        )
    if referring:
        return VLM_REFER_POINT_PROMPT.format(description=labels[0])
    return VLM_DENSE_POINT_PROMPT.format(categories=", ".join(labels))
