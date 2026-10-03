"""GAM-native Rex-Omni Point-in-mask evaluation."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


_POINT_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _POINT_DIR.parents[1]
_SHARED_UTILS = _EVAL_ROOT / "utils"
if str(_SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(_SHARED_UTILS))

from prompt_mode import (  # noqa: E402
    TaskType,
    build_mode_dense_point_prompt,
    build_mode_refer_point_prompt,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_point_to_pixel,
    parse_mode_predictions_for_scoring,
)
from eval_io import image_failure_marker, record_image_failure, valid_results  # noqa: E402
from coordinate_mode import vlm_point_to_pixel  # noqa: E402
from vlm_output_fallbacks import fallback_points  # noqa: E402
from eval_data_root import dataset_path  # noqa: E402

_PROMPT_SPEC = importlib.util.spec_from_file_location(
    "_gam_point_in_mask_prompts", _POINT_DIR / "prompts.py"
)
if _PROMPT_SPEC is None or _PROMPT_SPEC.loader is None:
    raise ImportError("cannot load Pointing/RexOmni/prompts.py")
_prompts = importlib.util.module_from_spec(_PROMPT_SPEC)
_PROMPT_SPEC.loader.exec_module(_prompts)

DATA_ROOT = Path(
    os.getenv(
        "GAM_REXOMNI_DATA_ROOT",
        str(dataset_path("images")),
    )
)


def _json_field(doc: Mapping[str, Any], name: str, default):
    value = doc.get(name)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _categories(doc: Mapping[str, Any]) -> list[str]:
    values = doc.get("categories") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def _is_referring(doc: Mapping[str, Any]) -> bool:
    return str(doc.get("task_name", "")) == "pointing_referring"


def _image_path(doc: Mapping[str, Any]) -> Path:
    return DATA_ROOT / str(doc["image_path"]).lstrip("/")


def doc_to_visual(doc: Mapping[str, Any]):
    try:
        with Image.open(_image_path(doc)) as image:
            return [image.convert("RGB")]
    except Exception as exc:
        record_image_failure(
            doc.get("dataset_name", "rex_point"),
            doc.get("id", doc.get("image_path", "")),
            _image_path(doc),
            exc,
        )
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]


def doc_to_text(doc: Mapping[str, Any], lmms_eval_specific_kwargs=None) -> str:
    if is_native_spatial_mode():
        categories = _categories(doc)
        return (
            build_mode_refer_point_prompt(categories[0])
            if _is_referring(doc)
            else build_mode_dense_point_prompt(categories)
        )
    return _prompts.point_prompt(
        _categories(doc), referring=_is_referring(doc), gam_mode=False
    )


def _strip_wrappers(text: object) -> str:
    cleaned = str(text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _parse_vlm_points(text: object) -> list[tuple[str, list[float]]]:
    cleaned = _strip_wrappers(text)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return fallback_points(cleaned)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback_points(cleaned)
    if isinstance(payload, dict):
        nested = next(
            (
                payload.get(key)
                for key in ("points", "predictions", "results")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        payload = nested if nested is not None else [payload]
    if not isinstance(payload, list):
        return fallback_points(cleaned)
    output = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        point = (
            item.get("point_2d")
            or item.get("point")
            or item.get("coordinate")
            or item.get("coordinates")
        )
        label = item.get("label") or item.get("category") or item.get("name") or ""
        if (
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in point)
        ):
            output.append((str(label), [float(point[0]), float(point[1])]))
    if not output:
        output.extend(
            fallback_points(cleaned)
        )
    return output


def _decode_mask(mask_info: Mapping[str, Any]) -> np.ndarray:
    counts = mask_info.get("counts", "")
    size = mask_info.get("size", [])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return np.zeros((0, 0), dtype=np.uint8)
    rle = {"counts": str(counts).encode("utf-8"), "size": list(size)}
    try:
        return coco_mask.decode(rle)
    except Exception:
        return np.zeros((int(size[0]), int(size[1])), dtype=np.uint8)


def _point_hits(point: list[float], mask: np.ndarray) -> bool:
    if mask.ndim != 2 or mask.size == 0:
        return False
    x, y = int(point[0]), int(point[1])
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def _score_sample(
    sample: Mapping[str, Any], *, ignore_labels: bool = False
) -> tuple[float, float]:
    gt_masks = sample.get("gt") or {}
    pred_points = sample.get("extracted_predictions") or {}
    total_gt = sum(len(values or []) for values in gt_masks.values())
    total_pred = sum(len(values or []) for values in pred_points.values())
    if total_gt == 0:
        return (1.0, 1.0) if total_pred == 0 else (0.0, 0.0)
    if total_pred == 0:
        return 0.0, 0.0

    predictions = [
        (str(label), point)
        for label, points in pred_points.items()
        for point in (points or [])
    ]
    used: set[int] = set()
    matches = 0
    for label, masks in gt_masks.items():
        for mask_info in masks or []:
            mask = _decode_mask(mask_info)
            matched_index = next(
                (
                    index
                    for index, (pred_label, point) in enumerate(predictions)
                    if index not in used
                    and (ignore_labels or pred_label == str(label))
                    and _point_hits(point, mask)
                ),
                None,
            )
            if matched_index is not None:
                used.add(matched_index)
                matches += 1
    return matches / total_gt, matches / total_pred


def process_results(doc: Mapping[str, Any], results: list[str]):
    raw = results[0] if results else ""
    try:
        with Image.open(_image_path(doc)) as image:
            width, height = image.size
    except Exception as exc:
        marker = image_failure_marker(
            doc.get("dataset_name", "rex_point"),
            doc.get("id", doc.get("image_path", "")),
            _image_path(doc),
            exc,
        )
        return {name: dict(marker) for name in ("recall", "precision", "f1")}
    parsed = (
        parse_mode_predictions_for_scoring(raw, TaskType.POINT)
        if is_native_spatial_mode()
        else _parse_vlm_points(raw)
    )
    extracted: dict[str, list[list[float]]] = {}
    seen_points: set[tuple[Any, ...]] = set()
    for label, point in parsed:
        absolute = (
            list(mode_grid_point_to_pixel(point, width, height))
            if is_native_spatial_mode()
            else list(vlm_point_to_pixel(point, width, height))
        )
        coordinate_key = (float(absolute[0]), float(absolute[1]))
        ignore_labels = is_gam_mode() and _is_referring(doc)
        key = coordinate_key if ignore_labels else (str(label), *coordinate_key)
        if key not in seen_points:
            seen_points.add(key)
            extracted.setdefault(str(label), []).append(absolute)

    gt_masks = _json_field(doc, "gt_mask", {})
    if is_gam_mode() and _is_referring(doc):
        if isinstance(gt_masks, Mapping):
            pooled_masks = [
                mask for values in gt_masks.values() for mask in (values or [])
            ]
        elif isinstance(gt_masks, list):
            pooled_masks = list(gt_masks)
        else:
            pooled_masks = []
        extracted = {
            "object": [point for values in extracted.values() for point in values]
        }
        gt_masks = {"object": pooled_masks}
    elif _is_referring(doc) and isinstance(gt_masks, dict) and len(gt_masks) == 1:
        gt_label = str(next(iter(gt_masks)))
        extracted = {
            gt_label: [point for values in extracted.values() for point in values]
        }
    # Only GAM referring-point tasks are geometry-only.  Category point tasks
    # (COCO/LVIS/Dense/VisDrone) remain label-aware.
    sample = {"gt": gt_masks, "extracted_predictions": extracted}
    recall, precision = _score_sample(sample)
    return {
        "recall": recall,
        "precision": precision,
        "f1": {"recall": recall, "precision": precision},
    }


def _mean(values) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def agg_recall(results, args=None):
    return _mean(valid_results(results, context="rex_point/recall"))


def agg_precision(results, args=None):
    return _mean(valid_results(results, context="rex_point/precision"))


def agg_f1(results, args=None):
    results = valid_results(results, context="rex_point/f1")
    if not results:
        return 0.0
    recall = _mean([value["recall"] for value in results])
    precision = _mean([value["precision"] for value in results])
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0
