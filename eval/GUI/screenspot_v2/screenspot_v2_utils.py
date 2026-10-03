"""ScreenSpot-v2 adapter sharing the canonical GAM GUI prompt/scorer stack.

The released annotation is ``xywh``.  This adapter performs an explicit,
fail-closed conversion to ``xyxy`` and only scores the first predicted point
against that box.  No bbox-generation view is exposed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import datasets
from PIL import Image


_SHARED_UTILS = Path(__file__).resolve().parents[2] / "utils"
if str(_SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(_SHARED_UTILS))
from eval_data_root import dataset_path  # noqa: E402


DEFAULT_DATA_ROOT = dataset_path("screenspot_v2")
IMAGE_SUBDIR = "screenspotv2_image"
TASK_ID = "gam_screenspot_v2"

_SHARED_PATH = Path(__file__).resolve().parents[1] / "screenspot_pro" / "screenspot_utils.py"
_SHARED_SPEC = importlib.util.spec_from_file_location(
    "_gam_screenspot_v2_shared", _SHARED_PATH
)
if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
    raise ImportError(f"cannot load shared ScreenSpot adapter: {_SHARED_PATH}")
_SHARED = importlib.util.module_from_spec(_SHARED_SPEC)
_SHARED_SPEC.loader.exec_module(_SHARED)


def _kwargs(lmms_eval_specific_kwargs=None) -> dict[str, Any]:
    result = dict(lmms_eval_specific_kwargs or {})
    result.setdefault("dataset_path", str(DEFAULT_DATA_ROOT))
    result["image_subdir"] = IMAGE_SUBDIR
    result["task_id"] = TASK_ID
    return result


def _platform(filename: str) -> str:
    if filename.startswith("mobile_"):
        return "mobile"
    if filename.startswith("pc_"):
        return "desktop"
    if filename.startswith("web_"):
        return "web"
    raise ValueError(f"cannot infer ScreenSpot-v2 platform: {filename!r}")


def normalize_screenspot_v2_doc(item: dict[str, Any]) -> dict[str, Any]:
    doc = dict(item)
    filename = str(doc.get("img_filename", ""))
    bbox = doc.get("bbox")
    if not filename or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"invalid ScreenSpot-v2 row: filename={filename!r}, bbox={bbox!r}")
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive ScreenSpot-v2 xywh: {bbox!r}")

    image_path = DEFAULT_DATA_ROOT / IMAGE_SUBDIR / filename
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    x2, y2 = x + width, y + height
    # Keep the released 1,272-row evaluation cardinality.  One official row
    # exceeds the right edge by exactly one pixel; record that fact explicitly
    # instead of silently clipping it.  The training materializer is stricter
    # and rejects that row, so no out-of-range coordinate reaches training.
    bbox_out_of_bounds = x < 0 or y < 0 or x2 > image_width or y2 > image_height
    ui_type = str(doc.get("data_type", "")).lower()
    if ui_type not in {"text", "icon"}:
        raise ValueError(f"unsupported ScreenSpot-v2 data_type: {ui_type!r}")
    instruction = str(doc.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"empty ScreenSpot-v2 instruction: {filename!r}")

    platform = _platform(filename)
    doc.update(
        {
            "id": f"screenspot_v2:{platform}:{filename}:{int(x)}:{int(y)}:{int(width)}:{int(height)}",
            "bbox_xywh": [x, y, width, height],
            "bbox": [x, y, x2, y2],
            "bbox_out_of_bounds": bbox_out_of_bounds,
            "img_size": [image_width, image_height],
            "instruction": instruction,
            "platform": platform,
            "application": str(doc.get("data_source", "unknown")),
            "ui_type": ui_type,
            "group": platform,
        }
    )
    return doc


def screenspot_v2_process_docs(dataset):
    return datasets.Dataset.from_list(
        [normalize_screenspot_v2_doc(dict(item)) for item in dataset]
    )


def screenspot_v2_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return _SHARED.screenspot_pro_doc_to_visual(doc, _kwargs(lmms_eval_specific_kwargs))


def screenspot_v2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _SHARED.screenspot_pro_doc_to_text(doc, _kwargs(lmms_eval_specific_kwargs))


def screenspot_v2_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _SHARED.screenspot_pro_doc_to_messages(doc, _kwargs(lmms_eval_specific_kwargs))


def screenspot_v2_process_results(doc, results, lmms_eval_specific_kwargs=None):
    return _SHARED.screenspot_pro_process_results(
        doc, results, _kwargs(lmms_eval_specific_kwargs)
    )


def screenspot_v2_aggregate_results(results, args=None, lmms_eval_specific_kwargs=None):
    return _SHARED.screenspot_pro_aggregate_results(
        results, args, _kwargs(lmms_eval_specific_kwargs)
    )


def screenspot_v2_aggregate_parse_error_rate(
    results, args=None, lmms_eval_specific_kwargs=None
):
    return _SHARED.screenspot_pro_aggregate_parse_error_rate(
        results, args, _kwargs(lmms_eval_specific_kwargs)
    )
