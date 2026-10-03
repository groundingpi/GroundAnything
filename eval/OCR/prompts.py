"""Prompt registry for the GAM scene-text E2E benchmarks."""

from __future__ import annotations

import os
from typing import Final


OCR_PROMPT: Final[str] = "OCR task detect all the text in box format."
VLM_OCR_BASE_PROMPT: Final[str] = (
    "OCR task locate and transcribe all legible text in the image."
)
VLM_OCR_PROMPT: Final[str] = (
    VLM_OCR_BASE_PROMPT
    + ' Return only a JSON array; each item must be '
    '{"text": "<transcription>", "bbox_2d": [x1, y1, x2, y2]}. '
    "Use coordinates normalized to 0-1000."
)
VLM_ABS_OCR_PROMPT: Final[str] = (
    "Detect and transcribe every legible text region in the image. "
    "Return only one JSON array in reading order. Every item must contain "
    'exactly these two fields in this order: {"bbox_2d": [x1, y1, x2, y2], '
    '"text": "<transcription>"}. '
    "bbox_2d must use absolute pixel coordinates in the original image. "
    "Never omit bbox_2d, never output text-only items, and never repeat an item. "
    "If the image contains no legible text, return []."
)

GAM_OCR_UNITS: Final[dict[str, str]] = {
    "gam_hiertext": "text line",
    "gam_icdar2015": "word",
    "gam_totaltext": "word",
    "gam_sroie": "text line",
}


def prompt_for_task(task_name: str, *, gam_mode: bool) -> str:
    base_task = task_name.removesuffix("_Boxonly")
    if base_task not in GAM_OCR_UNITS:
        raise KeyError(f"unsupported OCR task: {task_name}")
    if gam_mode:
        from prompt_mode import get_eval_mode

        mode = get_eval_mode()
        if mode == "REXOMNI":
            unit = GAM_OCR_UNITS[base_task]
            return (
                f"Can you detect all the {unit} in this image in box format like "
                "[x0, y0, x1, y1] and then recognize them?"
            )
        if mode == "LOCATEANYTHING":
            return "Detect all the text in box format."
        return OCR_PROMPT
    # Qwen2.5-VL's native localization contract uses original-image absolute
    # pixels.  Asking it for Qwen3's normalized grid is internally
    # contradictory and frequently makes the model emit text-only objects,
    # followed by a repetition loop.  Keep this explicit coordinate-mode
    # branch aligned with the scorer instead of guessing from numeric ranges.
    if os.environ.get("GAM_COORD_MODE", "qwen3").strip().lower() in {
        "abs",
        "qwen2",
    }:
        return VLM_ABS_OCR_PROMPT
    return VLM_OCR_PROMPT
