"""OSWorld-G evaluation aligned with ScreenSpot-Pro prompting and scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

from PIL import Image


_HERE = Path(__file__).resolve().parent
_GUI_ROOT = _HERE.parent
_UTILS = _GUI_ROOT.parent / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from prompt_mode import (  # noqa: E402
    TaskType,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_point_to_norm,
    parse_native_predictions,
    parse_mode_predictions_for_scoring,
)
from eval_io import record_image_failure, recorded_image_failure, valid_results  # noqa: E402
from coordinate_mode import vlm_point_to_norm  # noqa: E402
from eval_data_root import dataset_path  # noqa: E402

import importlib.util

_SCREEN_PATH = _GUI_ROOT / "screenspot_pro" / "screenspot_utils.py"
_SCREEN_SPEC = importlib.util.spec_from_file_location("_gam_osworld_screen", _SCREEN_PATH)
if _SCREEN_SPEC is None or _SCREEN_SPEC.loader is None:
    raise ImportError(f"cannot load ScreenSpot-Pro utility: {_SCREEN_PATH}")
_screen = importlib.util.module_from_spec(_SCREEN_SPEC)
_SCREEN_SPEC.loader.exec_module(_screen)

DATA_ROOT = Path(
    os.getenv(
        "GAM_OSWORLDG_ROOT",
        str(dataset_path("osworld_g")),
    )
)

def doc_to_visual(doc):
    path = DATA_ROOT / "images" / str(doc["image_path"])
    try:
        with Image.open(path) as image:
            return [image.convert("RGB")]
    except Exception as exc:
        record_image_failure("gam_osworld_g", doc.get("id", ""), path, exc)
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    adapted = {
        "instruction": str(doc["instruction"]),
        "img_size": list(doc["image_size"]),
    }
    return _screen.screenspot_pro_doc_to_text(adapted, lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Reuse ScreenSpot-Pro's message contract without changing its prompt."""

    if is_gam_mode() and not doc_to_visual(doc):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": doc_to_text(doc)}],
            }
        ]

    adapted = {
        "instruction": str(doc["instruction"]),
        "img_size": list(doc["image_size"]),
        "img_filename": str(doc["image_path"]),
    }
    kwargs = dict(lmms_eval_specific_kwargs or {})
    kwargs["dataset_path"] = str(DATA_ROOT)
    return _screen.screenspot_pro_doc_to_messages(adapted, kwargs)


_BOX_POINT_RE = re.compile(
    r"<\|box_start\|>\s*\(?\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>"
)
_PAIR_RE = re.compile(
    r"[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]"
)
_POINT_TAG_RE = re.compile(
    r"<point>\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*</point>"
)
_ACTION_START_BOX_RE = re.compile(
    r"Action\s*:\s*click\s*\(\s*start_box\s*="
    r"[^0-9]{0,16}(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_vlm_pair(x: float, y: float, image_size):
    try:
        width, height = (float(value) for value in image_size)
        normalized = vlm_point_to_norm((x, y), width, height)
    except (TypeError, ValueError):
        return None
    if all(0.0 <= value <= 1.0 for value in normalized):
        return list(normalized)
    return None


def parse_vlm_point(response: str, image_size):
    """Parse supported Qwen/GUI outputs without inferring from prose."""

    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]

    native = parse_native_predictions(response, TaskType.POINT)
    if native:
        return mode_grid_point_to_norm(native[-1][1])

    # Qwen computer-use tool calls and JSON bbox variants.
    candidates = [response]
    if "<tool_call>" in response and "</tool_call>" in response:
        candidates.insert(
            0, response.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        )
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        arguments = payload.get("arguments", payload) if isinstance(payload, dict) else {}
        coordinate = (
            arguments.get("coordinate")
            or arguments.get("coordinates")
            or arguments.get("bbox_2d")
            or arguments.get("bbox")
        )
        if not isinstance(coordinate, (list, tuple)):
            continue
        if len(coordinate) == 2:
            return _normalize_vlm_pair(
                float(coordinate[0]), float(coordinate[1]), image_size
            )
        if len(coordinate) == 4:
            x = (float(coordinate[0]) + float(coordinate[2])) / 2.0
            y = (float(coordinate[1]) + float(coordinate[3])) / 2.0
            return _normalize_vlm_pair(x, y, image_size)

    matches = _BOX_POINT_RE.findall(response)
    if not matches:
        matches = _POINT_TAG_RE.findall(response)
    if not matches:
        matches = _PAIR_RE.findall(response)
    if not matches:
        # Some Qwen checkpoints retain the Action/click contract but emit a
        # damaged quote/parenthesis wrapper, e.g. start_box='='744,310)'.
        # Anchor recovery to the full action signature so prose numbers cannot
        # be mistaken for coordinates.
        matches = _ACTION_START_BOX_RE.findall(response)
    if not matches:
        return None
    x, y = (float(value) for value in matches[-1])
    return _normalize_vlm_pair(x, y, image_size)


def _point_in_polygon(point, polygon):
    x, y = point
    vertices = list(zip(polygon[0::2], polygon[1::2]))
    if len(vertices) < 3:
        return False
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _is_refusal(doc) -> bool:
    coordinates = doc.get("box_coordinates") or []
    return str(doc.get("box_type", "")).lower() == "refusal" or (
        len(coordinates) >= 4 and all(float(value) == 0.0 for value in coordinates)
    )


def _hit(doc, point_norm) -> bool:
    width, height = (float(value) for value in doc["image_size"])
    point = (point_norm[0] * width, point_norm[1] * height)
    kind = str(doc.get("box_type", "bbox")).lower()
    coordinates = [float(value) for value in doc.get("box_coordinates", [])]
    if kind == "bbox" and len(coordinates) >= 4:
        x, y, box_width, box_height = coordinates[:4]
        return x <= point[0] <= x + box_width and y <= point[1] <= y + box_height
    if kind == "polygon":
        return _point_in_polygon(point, coordinates)
    return False


def process_results(doc, results, lmms_eval_specific_kwargs=None):
    response = results[0] if isinstance(results, list) and results else str(results or "")
    marker = recorded_image_failure("gam_osworld_g", doc.get("id", ""))
    if marker:
        marker = dict(marker)
        marker.update({"raw_response": response, "data_id": str(doc.get("id", ""))})
        return {"exact_acc": marker, "parse_error_rate": marker}
    if is_native_spatial_mode():
        parsed = parse_mode_predictions_for_scoring(response, TaskType.POINT)
        points = [mode_grid_point_to_norm(parsed[0][1])] if parsed else []
    else:
        point = parse_vlm_point(response, doc["image_size"])
        points = [point] if point is not None else []

    refusal = _is_refusal(doc)
    refusal_signal = "call_user" in response.lower() or "none" in response.lower()
    parse_error = not points and not (refusal and refusal_signal)
    if refusal:
        correct = not points and refusal_signal
    else:
        correct = bool(points) and _hit(doc, points[0])
    return {
        "exact_acc": {
            "is_correct": correct,
            "correctness": "correct" if correct else ("wrong" if points else "wrong_format"),
            "parse_error": parse_error,
            "pred_points": points,
            "raw_response": response,
            "data_id": str(doc.get("id", "")),
            "box_type": str(doc.get("box_type", "")),
        },
        "parse_error_rate": {
            "parse_error": parse_error,
            "raw_response": response,
            "data_id": str(doc.get("id", "")),
        },
    }


def aggregate_results(results, args=None, lmms_eval_specific_kwargs=None):
    results = valid_results(results, context="gam_osworld_g/exact_acc")
    if not results:
        return 0.0
    return sum(bool(result.get("is_correct")) for result in results) / len(results)


def aggregate_parse_error_rate(results, args=None, lmms_eval_specific_kwargs=None):
    results = valid_results(results, context="gam_osworld_g/parse_error_rate")
    if not results:
        return 0.0
    return sum(bool(result.get("parse_error")) for result in results) / len(results)
