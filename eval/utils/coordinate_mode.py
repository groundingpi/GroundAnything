"""Model-family routed coordinate conversions for VLM-mode evaluation.

Qwen3-VL emits a 0..1000 grid, while Qwen2.5-VL emits original-image
absolute pixels.  Values alone cannot distinguish those protocols on images
smaller than 1000 pixels, so the runner resolves Qwen2-family checkpoints to
``qwen2`` and Qwen3-or-newer/default checkpoints to ``qwen3`` before scoring.
An explicit ``GAM_COORD_MODE`` remains available for non-Qwen protocols.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable


VALID_COORD_MODES = frozenset({"qwen2", "qwen3", "norm01", "abs", "auto"})


def coordinate_mode_for_model(model_path: str | os.PathLike[str]) -> str:
    """Return the native Qwen VLM coordinate protocol for a checkpoint.

    Qwen2/Qwen2.5 use original-image absolute pixels.  Qwen3 and later use a
    relative 0..1000 grid.  Unknown families deliberately follow the modern
    relative protocol instead of value-magnitude guessing.
    """

    path = Path(model_path).expanduser()
    hints = [str(path)]
    try:
        config_path = path.resolve() / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError, TypeError):
        config = {}
    for key in ("model_type", "architectures"):
        value = config.get(key)
        if isinstance(value, str):
            hints.append(value)
        elif isinstance(value, list):
            hints.extend(str(item) for item in value)
    joined = " ".join(hints).lower().replace("-", "_")
    return "qwen2" if "qwen2" in joined and "qwen3" not in joined else "qwen3"


def vlm_coord_mode() -> str:
    mode = os.environ.get("GAM_COORD_MODE", "qwen3").strip().lower()
    if mode not in VALID_COORD_MODES:
        raise ValueError(
            f"unsupported GAM_COORD_MODE={mode!r}; expected one of "
            f"{sorted(VALID_COORD_MODES)}"
        )
    return mode


def _finite(values: Iterable[float], expected: int) -> list[float]:
    output = [float(value) for value in values]
    if len(output) != expected or not all(math.isfinite(value) for value in output):
        raise ValueError(f"expected {expected} finite coordinates, got {output!r}")
    return output


def _resolved_mode(values: list[float]) -> str:
    mode = vlm_coord_mode()
    if mode == "qwen2":
        return "abs"
    if mode != "auto":
        return mode
    magnitude = max(abs(value) for value in values)
    if magnitude <= 1.5:
        return "norm01"
    if magnitude <= 1000.0:
        return "qwen3"
    return "abs"


def vlm_point_to_pixel(point, width: float, height: float) -> tuple[float, float]:
    values = _finite(point, 2)
    mode = _resolved_mode(values)
    if mode == "norm01":
        return values[0] * width, values[1] * height
    if mode == "qwen3":
        return values[0] / 1000.0 * width, values[1] / 1000.0 * height
    return values[0], values[1]


def vlm_point_to_norm(point, width: float, height: float) -> tuple[float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    x, y = vlm_point_to_pixel(point, width, height)
    return x / float(width), y / float(height)


def vlm_box_to_pixel(box, width: float, height: float) -> list[float]:
    values = _finite(box, 4)
    mode = _resolved_mode(values)
    if mode == "norm01":
        return [
            values[0] * width,
            values[1] * height,
            values[2] * width,
            values[3] * height,
        ]
    if mode == "qwen3":
        return [
            values[0] / 1000.0 * width,
            values[1] / 1000.0 * height,
            values[2] / 1000.0 * width,
            values[3] / 1000.0 * height,
        ]
    return values


def vlm_box_to_norm1000(box, width: float, height: float) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    values = vlm_box_to_pixel(box, width, height)
    return [
        values[0] / float(width) * 1000.0,
        values[1] / float(height) * 1000.0,
        values[2] / float(width) * 1000.0,
        values[3] / float(height) * 1000.0,
    ]
