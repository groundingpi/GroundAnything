"""Category-exploded Rex-Omni visual_prompt_eval implementation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import datasets
from PIL import Image


_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_ROOT = os.path.dirname(_HERE)
_SHARED_UTILS_DIR = os.path.join(_EVAL_ROOT, "utils")
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

import detection_utils as _detection  # noqa: E402
from prompt_mode import (  # noqa: E402
    TaskType,
    build_mode_visual_prompt,
    is_gam_mode,
    is_locateanything_mode,
    is_native_spatial_mode,
    mode_grid_to_abs,
    parse_mode_predictions_for_scoring,
    pixel_boxes_to_mode_grid,
)
from eval_io import image_failure_marker, record_image_failure  # noqa: E402

_PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "_gam_visual_prompt_prompts", os.path.join(_HERE, "prompts.py")
)
if _PROMPTS_SPEC is None or _PROMPTS_SPEC.loader is None:
    raise ImportError("cannot load VisualPrompt/prompts.py")
_prompts = importlib.util.module_from_spec(_PROMPTS_SPEC)
_PROMPTS_SPEC.loader.exec_module(_prompts)


def _mapping(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def explode_visual_prompt_dataset(dataset):
    """Match the official evaluator: one inference per image/category pair."""

    rows = []
    for source_index, source in enumerate(dataset):
        gt = _mapping(source.get("gt"))
        references = _mapping(source.get("visual_prompt"))
        categories = source.get("categories") or list(references)
        for category in categories:
            category = str(category)
            gt_boxes = gt.get(category)
            reference_boxes = references.get(category)
            if not gt_boxes or not reference_boxes:
                continue
            rows.append(
                {
                    "source_index": source_index,
                    "record_id": f"{source['image_path']}::{category}",
                    "image_path": str(source["image_path"]),
                    "category": category,
                    "gt_json": json.dumps({category: gt_boxes}, ensure_ascii=False),
                    "visual_prompt_json": json.dumps(
                        {category: reference_boxes}, ensure_ascii=False
                    ),
                    "task_name": "visual_prompt_detection",
                    "dataset_name": str(source.get("dataset_name", "unknown")),
                }
            )
    if not rows:
        raise ValueError("visual-prompt dataset became empty after category explosion")
    return datasets.Dataset.from_list(rows)


def _full_image_path(doc):
    return os.path.join(_detection.IMAGE_ROOT, str(doc["image_path"]))


def _reference_boxes(doc):
    return _detection._visual_prompt_boxes(_mapping(doc["visual_prompt_json"]))


def doc_to_visual(doc):
    path = _full_image_path(doc)
    try:
        with Image.open(path) as image:
            visual = image.convert("RGB")
    except Exception as exc:
        record_image_failure(
            doc.get("dataset_name", "visual_prompt"),
            doc.get("record_id", doc.get("source_index", "")),
            path,
            exc,
        )
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]
    if is_locateanything_mode():
        crops = []
        width, height = visual.size
        for box in _reference_boxes(doc):
            x1, y1, x2, y2 = (float(value) for value in box)
            left = max(0, min(width - 1, round(x1)))
            top = max(0, min(height - 1, round(y1)))
            right = max(left + 1, min(width, round(x2)))
            bottom = max(top + 1, min(height, round(y2)))
            crops.append(visual.crop((left, top, right, bottom)).convert("RGB"))
        if not crops:
            raise ValueError("LocateAnything visual prompt has no valid reference crop")
        return [_detection._maybe_downscale(visual), *crops]
    if not is_native_spatial_mode():
        visual = _detection._draw_visual_prompt(
            visual, _mapping(doc["visual_prompt_json"])
        )
    return [_detection._maybe_downscale(visual)]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if is_native_spatial_mode():
        try:
            width, height = _detection._image_size(_full_image_path(doc))
        except Exception as exc:
            record_image_failure(
                doc.get("dataset_name", "visual_prompt"),
                doc.get("record_id", doc.get("source_index", "")),
                _full_image_path(doc),
                exc,
            )
            return build_mode_visual_prompt(((0, 0, 1, 1),))
        boxes = pixel_boxes_to_mode_grid(_reference_boxes(doc), width, height)
        return build_mode_visual_prompt(boxes)
    return _prompts.visual_prompt([], gam_mode=False)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Match LocateAnything's official source/text/reference-crops ordering."""

    visuals = doc_to_visual(doc)
    text = doc_to_text(doc, lmms_eval_specific_kwargs)
    if is_locateanything_mode():
        content = [{"type": "image", "url": visuals[0]}]
        content.append({"type": "text", "text": text})
        content.extend({"type": "image", "url": image} for image in visuals[1:])
    else:
        content = [{"type": "image", "url": image} for image in visuals]
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def process_results(doc, results):
    raw_response = results[0] if results else ""
    try:
        width, height = _detection._image_size(_full_image_path(doc))
    except Exception as exc:
        marker = image_failure_marker(
            doc.get("dataset_name", "visual_prompt"),
            doc.get("record_id", doc.get("source_index", "")),
            _full_image_path(doc),
            exc,
        )
        return _detection._failure_metric_payload(marker)
    parsed = (
        parse_mode_predictions_for_scoring(raw_response, TaskType.BBOX)
        if is_native_spatial_mode()
        else _detection._parse_predictions(raw_response)
    )
    all_boxes = []
    for label, coordinates in parsed:
        if is_gam_mode() and str(label) != "object":
            continue
        all_boxes.append(
            mode_grid_to_abs(coordinates, width, height)
            if is_native_spatial_mode()
            else _detection._to_abs(coordinates, width, height)
        )
    category = str(doc["category"])
    all_boxes = _detection._deduplicate_box_mapping({category: all_boxes})[category]
    gt = _mapping(doc["gt_json"])
    sample = {
        "gt": gt,
        "extracted_predictions": {category: all_boxes},
        "task_name": "visual_prompt_detection",
        "dataset_name": str(doc["dataset_name"]),
        # The task has one opaque canonical label (``object`` is validated
        # above, then mapped to the dataset category), so the GAM scorer can
        # reuse its precomputed ten-threshold IoU sweep without changing the
        # metric contract.
        "_gam_category_aware": is_gam_mode(),
    }
    recalls, precisions, mae = _detection._score_sample_iou_sweep(sample)
    return _detection._metric_payload(recalls, precisions, mae)


def agg_recall(results, args=None):
    return _detection.agg_recall(results, args)


def agg_precision(results, args=None):
    return _detection.agg_precision(results, args)


def agg_f1(results, args=None):
    return _detection.agg_f1(results, args)


def agg_mean_iou(results, args=None):
    return _detection.agg_mean_iou(results, args)


def agg_f1_mean_iou(results, args=None):
    return _detection.agg_f1_mean_iou(results, args)


def agg_mae(results, args=None):
    return _detection.agg_mae(results, args)
