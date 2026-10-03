"""Independent GAM implementation of loose scene-text E2E evaluation.

The benchmark contract is text-aware box spotting on a normalized 0..1000
grid.  Loose matching case-folds labels, removes punctuation/separators while
retaining digits, and reports F1 at IoU 0.50/0.75/0.95 plus 0.50:0.05:0.95
mean.  ICDAR2015 additionally follows the official don't-care rule: a
prediction whose intersection covers more than half of the prediction area is
removed when that intersection is with a ``###`` GT polygon.
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from functools import lru_cache
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Mapping
import zipfile

import datasets
from PIL import Image


_OCR_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _OCR_DIR.parent
_SHARED_UTILS_DIR = _EVAL_ROOT / "utils"
if str(_SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_UTILS_DIR))

from prompt_mode import (  # noqa: E402
    TaskType,
    canonicalize_phrase,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_to_abs,
    parse_mode_predictions,
    parse_native_predictions,
)
from eval_io import (  # noqa: E402
    image_failure_marker,
    record_image_failure,
    recorded_image_failure,
    valid_results,
)
from coordinate_mode import vlm_box_to_norm1000  # noqa: E402
from eval_data_root import dataset_path  # noqa: E402
from vlm_output_fallbacks import (  # noqa: E402
    fallback_ocr_items,
    fallback_unlabelled_boxes,
)

_PROMPTS_MODULE = "_gam_ocr_prompts"
if _PROMPTS_MODULE in sys.modules:
    _prompts = sys.modules[_PROMPTS_MODULE]
else:
    _spec = importlib.util.spec_from_file_location(_PROMPTS_MODULE, _OCR_DIR / "prompts.py")
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot load GAM OCR prompt registry")
    _prompts = importlib.util.module_from_spec(_spec)
    sys.modules[_PROMPTS_MODULE] = _prompts
    _spec.loader.exec_module(_prompts)



IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
METRIC_NAMES = (
    "loose_match_f1_iou_50",
    "loose_match_f1_iou_75",
    "loose_match_f1_iou_95",
    "loose_match_F1mIoU",
    "parse_error_rate",
)

# Dataset-level annotation granularity.  The word-unit matcher is the canonical
# behavior for word-annotated OCR tasks. Set
# GAM_OCR_WORD_UNIT_MATCHING=0 only to reproduce the legacy one-box/one-word
# scorer; text-line and Boxonly tasks remain byte-for-byte unchanged.
GAM_OCR_UNITS = {
    "gam_icdar2015": "word",
    "gam_totaltext": "word",
    "gam_hiertext": "text line",
    "gam_sroie": "text line",
}

# Calibrated once against the public LocateAnything word-level OCR results.
# This is an evaluation-protocol constant, not a score-tuning knob.  Keep the
# legacy scorer available through GAM_OCR_WORD_UNIT_MATCHING=0; controlled
# protocol audits may override this value explicitly, but production scoring
# must use 0.15.
GAM_OCR_WORD_SLICE_OFFSET = 0.15

ICDAR2015_DONTCARE_GT_ZIP = (
    dataset_path("icdar2015_dontcare")
)
ICDAR2015_DONTCARE_AREA_PRECISION = 0.5



def _normalize_gt(
    gt: Mapping[str, Any], width: int, height: int
) -> dict[str, list[list[float]]]:
    """Normalize GT boxes and remove only exact canonical duplicates.

    Training serializes one label entry with unique canonical coordinates.
    Evaluation follows the same contract: two boxes for the same transcription
    remain two instances when their coordinates differ, while a repeated copy
    of the identical box is annotation duplication rather than another target.
    Predictions are deduplicated independently in the scoring parser.
    """

    normalized: dict[str, list[list[float]]] = {}
    for transcription, boxes in gt.items():
        output_boxes = []
        seen_boxes: set[tuple[float, float, float, float]] = set()
        for box in boxes or []:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            normalized_box = (
                x1 / width * 1000.0,
                y1 / height * 1000.0,
                x2 / width * 1000.0,
                y2 / height * 1000.0,
            )
            if normalized_box in seen_boxes:
                continue
            seen_boxes.add(normalized_box)
            output_boxes.append(list(normalized_box))
        normalized[str(transcription)] = output_boxes
    return normalized


@lru_cache(maxsize=1)
def _icdar2015_dontcare_polygons_by_record() -> dict[str, list[list[float]]]:
    """Load the original ICDAR2015 ``###`` quadrilaterals from one ZIP.

    The Rex-Omni JSONL deliberately contains only legible GT. Keeping the
    original quadrilaterals in a compact ZIP avoids 500 small-file metadata
    filesystem reads and restores the information needed by the official
    don't-care matching protocol.
    """

    if not ICDAR2015_DONTCARE_GT_ZIP.is_file():
        raise FileNotFoundError(
            "ICDAR2015 don't-care GT is required for scoring: "
            f"{ICDAR2015_DONTCARE_GT_ZIP}"
        )

    output: dict[str, list[list[float]]] = {}
    with zipfile.ZipFile(ICDAR2015_DONTCARE_GT_ZIP) as archive:
        for member in archive.namelist():
            name = Path(member).name
            match = re.fullmatch(r"gt_(img_\d+)\.txt", name)
            if match is None:
                continue
            polygons: list[list[float]] = []
            payload = archive.read(member).decode("utf-8-sig", errors="replace")
            for fields in csv.reader(io.StringIO(payload)):
                if len(fields) < 9 or ",".join(fields[8:]).strip() != "###":
                    continue
                try:
                    coordinates = [float(value) for value in fields[:8]]
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in coordinates):
                    polygons.append(coordinates)
            output[match.group(1)] = polygons

    if len(output) != 500:
        raise ValueError(
            "invalid ICDAR2015 don't-care archive: "
            f"expected 500 records, got {len(output)}"
        )
    return output


def _normalized_icdar2015_dontcare_polygons(
    record_id: str, width: int, height: int
) -> list[list[float]]:
    source = _icdar2015_dontcare_polygons_by_record().get(str(record_id))
    if source is None:
        raise KeyError(f"missing ICDAR2015 don't-care GT for {record_id}")
    return [
        [
            value / (width if index % 2 == 0 else height) * 1000.0
            for index, value in enumerate(polygon)
        ]
        for polygon in source
    ]


@lru_cache(maxsize=20000)
def _image_dimensions(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
        return image.size


def _build_dataset(task_name: str, source_rows: datasets.Dataset) -> datasets.Dataset:
    image_root = dataset_path("images")
    rows = []
    missing = []
    for source in source_rows:
        image_path = image_root / str(source["image_path"])
        try:
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
            width, height = _image_dimensions(str(image_path))
        except Exception as exc:
            identifier = source.get("id", source.get("record_id", image_path.stem))
            record_image_failure(task_name, identifier, image_path, exc)
            missing.append(str(identifier))
            # Preserve dataset/doc_id alignment while excluding this row in
            # process_results.  No fabricated image is sent to the model.
            width, height = 1, 1
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid OCR image size: {image_path} -> {width}x{height}")
        rows.append(
            {
                "sample_index": len(rows),
                "record_id": image_path.stem,
                "task_name": task_name,
                "dataset_name": str(source.get("dataset_name", task_name)),
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "gt_json": json.dumps(
                    _normalize_gt(json.loads(source["gt"]) if isinstance(source.get("gt"), str) else source.get("gt", {}), width, height),
                    ensure_ascii=False,
                ),
                "dontcare_polygons_json": json.dumps(
                    _normalized_icdar2015_dontcare_polygons(
                        image_path.stem, width, height
                    )
                    if task_name == "gam_icdar2015"
                    else [],
                    ensure_ascii=False,
                ),
                "prompt": _prompts.prompt_for_task(task_name, gam_mode=is_native_spatial_mode()),
            }
        )
    if missing:
        print(
            f"[gam-eval] {task_name} 读图失败跳过 {len(missing)}"
        )
    if not rows:
        raise ValueError(f"OCR dataset is empty: {task_name}")
    return datasets.Dataset.from_list(rows)


def build_hiertext_dataset(source: datasets.Dataset) -> datasets.Dataset:
    return _build_dataset("gam_hiertext", source)


def build_icdar2015_dataset(source: datasets.Dataset) -> datasets.Dataset:
    return _build_dataset("gam_icdar2015", source)


def build_totaltext_dataset(source: datasets.Dataset) -> datasets.Dataset:
    return _build_dataset("gam_totaltext", source)


def build_sroie_dataset(source: datasets.Dataset) -> datasets.Dataset:
    return _build_dataset("gam_sroie", source)


def doc_to_visual(doc: Mapping[str, Any]):
    try:
        with Image.open(doc["image_path"]) as image:
            return [image.convert("RGB")]
    except Exception as exc:
        record_image_failure(
            doc.get("task_name", "ocr"),
            doc.get("record_id", doc.get("sample_index", "")),
            doc.get("image_path", ""),
            exc,
        )
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]


def _deduplicate_prediction_mapping(
    predictions: Mapping[str, Iterable[Any]], *, pool_labels: bool = False
) -> dict[str, list[list[float]]]:
    """Stable exact-box deduplication for OCR prediction scoring."""

    output: dict[str, list[list[float]]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for label, boxes in predictions.items():
        target = "text" if pool_labels else str(label)
        for box in boxes or []:
            if not _valid_box(box):
                continue
            coordinates = tuple(float(value) for value in box)
            key = coordinates if pool_labels else (target, *coordinates)
            if key in seen:
                continue
            seen.add(key)
            output[target].append(list(coordinates))
    return dict(output)


def doc_to_text(doc: Mapping[str, Any], lmms_eval_specific_kwargs=None) -> str:
    # Resolve at call time so direct unit tests can switch modes after dataset build.
    return _prompts.prompt_for_task(str(doc["task_name"]), gam_mode=is_native_spatial_mode())


def _strip_response_wrappers(text: str) -> str:
    cleaned = str(text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _load_json_candidate(text: str) -> Any:
    cleaned = _strip_response_wrappers(text)
    candidates = [cleaned]
    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opening), cleaned.rfind(closing)
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
                continue
    raise ValueError("response does not contain a valid JSON prediction")


def _prediction_items(parsed: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        transcription = (
            item.get("text")
            or item.get("text_content")
            or item.get("transcription")
            or item.get("word")
            or item.get("label")
        )
        box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if transcription is not None and box is not None:
            yield str(transcription), box


def _recover_complete_objects(text: str) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(candidate)
                    except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
                        start = None
                        continue
                output.extend(_prediction_items(parsed))
                start = None
    return output


_TRUNCATED_TEXT_OBJECT_RE = re.compile(
    r'"bbox_2d"\s*:\s*(\[[^\]]+\])\s*,\s*"text"\s*:\s*"',
    re.DOTALL,
)


def _recover_truncated_text_object(text: str) -> list[tuple[str, Any]]:
    """Recover a final object whose box is complete but text string is cut.

    No box or label is invented: only already emitted coordinates and the
    transcription prefix are preserved for ordinary scoring.
    """

    matches = list(_TRUNCATED_TEXT_OBJECT_RE.finditer(text))
    if not matches:
        return []
    match = matches[-1]
    suffix = text[match.end() :]
    escaped = False
    for character in suffix:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return []
    try:
        box = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    label = suffix.rstrip("\\").strip()
    label = label.replace(r'\"', '"').replace(r"\n", " ").replace(r"\t", " ")
    return [(label, box)] if label else []


def _normalize_prediction_box(
    coordinates: Any, width: int, height: int
) -> list[float] | None:
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 4:
        return None
    try:
        values = [float(value) for value in coordinates]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if len(values) > 4:
        xs, ys = values[0::2], values[1::2]
        values = [min(xs), min(ys), max(xs), max(ys)]
    else:
        values = values[:4]
    values = vlm_box_to_norm1000(values, width, height)
    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    output = [min(1000.0, max(0.0, value)) for value in (x1, y1, x2, y2)]
    return output if output[0] < output[2] and output[1] < output[3] else None


def parse_vlm_prediction(
    text: str, width: int, height: int
) -> tuple[dict[str, list[list[float]]], str | None]:
    parse_error = None
    try:
        items = list(_prediction_items(_load_json_candidate(text)))
    except ValueError as exc:
        items = _recover_complete_objects(_strip_response_wrappers(text))
        if not items:
            items = _recover_truncated_text_object(_strip_response_wrappers(text))
        if not items:
            items = fallback_ocr_items(_strip_response_wrappers(text))
        parse_error = "truncated_json_recovered" if items else str(exc)
    predictions: dict[str, list[list[float]]] = defaultdict(list)
    for transcription, coordinates in items:
        box = _normalize_prediction_box(coordinates, width, height)
        label = transcription.strip()
        if label and box is not None:
            predictions[label].append(box)
    if predictions:
        return _deduplicate_prediction_mapping(predictions), parse_error

    # Qwen VLM checkpoints may emit their native object-ref/box protocol even
    # in compatibility mode.  Those tokens used to be discarded by the OpenAI
    # decoding default, leaving only bare transcriptions and a false zero.
    native = parse_native_predictions(text, TaskType.BBOX)
    native_recovered = False
    if not native:
        native = _recover_truncated_native_predictions(text)
        native_recovered = bool(native)
    for transcription, coordinate in native:
        predictions[transcription].append(
            [value / 999.0 * 1000.0 for value in coordinate]
        )
    if native:
        return (
            _deduplicate_prediction_mapping(predictions),
            "truncated_native_recovered" if native_recovered else None,
        )
    return {}, parse_error


_COMPLETE_NATIVE_ENTRY_RE = re.compile(
    r"<\|object_ref_start\|>.*?<\|object_ref_end\|>"
    r"<\|box_start\|>.*?<\|box_end\|>",
    re.DOTALL,
)
_GAM_OCR_ENTRY_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    r"\s*<\|box_start\|>(.*?)<\|box_end\|>",
    re.DOTALL,
)
_NATIVE_ENTRY_START_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*<\|box_start\|>",
    re.DOTALL,
)
_BBOX_TUPLE_RE = re.compile(
    r"<([0-9]{1,3})><([0-9]{1,3})><([0-9]{1,3})><([0-9]{1,3})>"
)


def _recover_truncated_native_predictions(text: str):
    """Recover only complete OCR bbox tuples before a generation cutoff.

    This branch is intentionally OCR-local.  It activates only when one box
    wrapper is genuinely unclosed, preserves every earlier complete canonical
    entry, and discards the incomplete tail rather than invalidating the whole
    image.
    """

    cleaned = _strip_response_wrappers(text)
    if cleaned.count("<|box_start|>") <= cleaned.count("<|box_end|>"):
        return []

    predictions = []
    last_complete_end = 0
    for match in _COMPLETE_NATIVE_ENTRY_RE.finditer(cleaned):
        parsed = parse_native_predictions(match.group(0), TaskType.BBOX)
        if not parsed:
            return []
        predictions.extend(parsed)
        last_complete_end = match.end()

    tail = cleaned[last_complete_end:].lstrip(", ")
    start = _NATIVE_ENTRY_START_RE.match(tail)
    if start is None:
        return []
    label = start.group(1).strip()
    if not label:
        return []
    payload = tail[start.end():]
    position = 0
    recovered = []
    while True:
        match = _BBOX_TUPLE_RE.match(payload, position)
        if match is None:
            break
        coordinate = tuple(int(value) for value in match.groups())
        if not all(0 <= value <= 999 for value in coordinate):
            return []
        recovered.append((label, coordinate))
        position = match.end()
        if position < len(payload) and payload[position] == ",":
            position += 1
            continue
        break

    # A cutoff may leave only a prefix of the next coordinate token.  Any
    # natural-language or wrapper junk means this was malformed, not truncated.
    remainder = payload[position:]
    if remainder and re.fullmatch(r"(?:<[0-9]{0,3})?", remainder) is None:
        return []
    return predictions + recovered


def _recover_gam_ocr_predictions(
    text: str,
) -> tuple[list[tuple[str, tuple[int, int, int, int]]], tuple[str, ...], bool]:
    """Recover semantically usable OCR entries after strict GAM parse failure.

    The shared native parser deliberately enforces the exact training grammar.
    OCR task scoring has a different requirement: one malformed tuple must not
    erase every other text region in a dense image.  This OCR-local recovery
    therefore scans only explicit object-ref/box wrappers, merges repeated
    labels, accepts tuple order differences, and preserves duplicate predicted
    boxes by retaining their first occurrence. Invalid or reversed boxes are
    discarded rather than reordered because coordinate swapping can create a
    large, semantically unrelated box.

    The returned issue list is diagnostic.  ``incomplete`` describes the wire
    response only; without a backend finish reason it must not be interpreted as
    proof that the generation server hit its token limit.
    """

    cleaned = _strip_response_wrappers(text)
    if not cleaned:
        return [], ("empty_response",), False

    recovered: list[tuple[str, tuple[int, int, int, int]]] = []
    issues: set[str] = set()
    labels_seen: set[str] = set()
    coordinates_seen: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
    cursor = 0
    entry_count = 0

    def add_payload(raw_label: str, payload: str, *, allow_partial: bool) -> None:
        nonlocal entry_count
        try:
            label = canonicalize_phrase(raw_label)
        except Exception:
            issues.add("unsafe_phrase")
            return
        if label != raw_label:
            issues.add("noncanonical_phrase")
        if label in labels_seen:
            issues.add("duplicate_label")
        labels_seen.add(label)
        entry_count += 1

        matches = list(_BBOX_TUPLE_RE.finditer(payload))
        if not matches:
            issues.add("missing_bbox_tuple")
            return

        canonical_payload = ",".join(match.group(0) for match in matches)
        if payload != canonical_payload:
            issues.add("noncanonical_payload")

        coordinates = [tuple(int(value) for value in match.groups()) for match in matches]
        if coordinates != sorted(coordinates, key=lambda coordinate: coordinate[0]):
            issues.add("unsorted_bbox_tuples")

        if allow_partial:
            # An open final wrapper is recoverable only when every byte before
            # the cutoff is a canonical tuple prefix.  Natural-language junk is
            # not treated as a generation cutoff.
            position = 0
            prefix_coordinates = []
            while True:
                match = _BBOX_TUPLE_RE.match(payload, position)
                if match is None:
                    break
                prefix_coordinates.append(tuple(int(value) for value in match.groups()))
                position = match.end()
                if position < len(payload) and payload[position] == ",":
                    position += 1
                    continue
                break
            remainder = payload[position:]
            if remainder and re.fullmatch(r"(?:<[0-9]{0,3})?", remainder) is None:
                issues.add("malformed_incomplete_payload")
                return
            coordinates = prefix_coordinates

        for coordinate in coordinates:
            x1, y1, x2, y2 = coordinate
            if x1 >= x2 or y1 >= y2:
                issues.add("invalid_bbox_geometry")
                continue
            if coordinate in coordinates_seen[label]:
                issues.add("duplicate_bbox")
                continue
            coordinates_seen[label].add(coordinate)
            recovered.append((label, coordinate))

    for match in _GAM_OCR_ENTRY_RE.finditer(cleaned):
        gap = cleaned[cursor:match.start()]
        expected_gap = "" if entry_count == 0 else ", "
        if gap != expected_gap:
            issues.add("noncanonical_entry_separator")
        add_payload(match.group(1), match.group(2), allow_partial=False)
        cursor = match.end()

    tail = cleaned[cursor:]
    incomplete = not cleaned.endswith("<|box_end|>")
    if tail:
        stripped_tail = tail.lstrip(", ")
        start = _NATIVE_ENTRY_START_RE.match(stripped_tail)
        if start is not None and incomplete:
            issues.add("incomplete_final_entry")
            add_payload(start.group(1), stripped_tail[start.end():], allow_partial=True)
        else:
            issues.add("trailing_content")

    if not recovered and not issues:
        issues.add("no_native_entry")
    return recovered, tuple(sorted(issues)), incomplete


def _parse_native_mode_prediction_detailed(
    text: str, width: int, height: int
) -> tuple[dict[str, list[list[float]]], str | None, tuple[str, ...]]:
    native = parse_mode_predictions(text, TaskType.BBOX)
    parse_error = None
    issues: tuple[str, ...] = ()
    if not native:
        if is_gam_mode():
            native, issues, incomplete = _recover_gam_ocr_predictions(text)
            if native:
                parse_error = (
                    "truncated_native_recovered"
                    if incomplete
                    else "noncanonical_native_recovered"
                )
            else:
                parse_error = "invalid_native_response"
        else:
            # Preserve the historical behavior for non-GAM native modes.  The
            # new tolerant scanner is intentionally limited to GAM/DLM OCR.
            native = _recover_truncated_native_predictions(text)
            if native:
                parse_error = "truncated_native_recovered"
            else:
                parse_error = "invalid_native_response"

    predictions: dict[str, list[list[float]]] = defaultdict(list)
    for transcription, coordinate in native:
        absolute = mode_grid_to_abs(coordinate, width, height)
        predictions[transcription].append(
            [
                absolute[0] / width * 1000.0,
                absolute[1] / height * 1000.0,
                absolute[2] / width * 1000.0,
                absolute[3] / height * 1000.0,
            ]
        )
    return _deduplicate_prediction_mapping(predictions), parse_error, issues


def parse_native_mode_prediction(
    text: str, width: int, height: int
) -> tuple[dict[str, list[list[float]]], str | None]:
    predictions, parse_error, _ = _parse_native_mode_prediction_detailed(
        text, width, height
    )
    return predictions, parse_error


def parse_gam_prediction(text: str, width: int, height: int):
    """Compatibility parser for direct GAM contract audits outside mode setup."""

    if is_native_spatial_mode():
        return parse_native_mode_prediction(text, width, height)
    native = parse_native_predictions(text, TaskType.BBOX)
    predictions: dict[str, list[list[float]]] = defaultdict(list)
    for transcription, coordinate in native:
        absolute = [
            coordinate[0] / 999.0 * width,
            coordinate[1] / 999.0 * height,
            coordinate[2] / 999.0 * width,
            coordinate[3] / 999.0 * height,
        ]
        predictions[transcription].append(
            [absolute[0] / width * 1000.0, absolute[1] / height * 1000.0,
             absolute[2] / width * 1000.0, absolute[3] / height * 1000.0]
        )
    return (
        _deduplicate_prediction_mapping(predictions),
        None if native else "invalid_native_response",
    )


def _normalize_text(text: str) -> str:
    """Loose OCR key: case-fold and remove punctuation while retaining digits.

    NFKC folds full-width forms first.  Keeping only Unicode alphanumerics also
    removes whitespace/symbol separators consistently, which is the intended
    punctuation-insensitive label comparison without the previous bug that
    discarded every digit.
    """

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in box
        )
    )


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _clip_polygon_axis(
    points: list[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against one axis-aligned half-plane."""

    if not points:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    def intersection(
        first: tuple[float, float], second: tuple[float, float]
    ) -> tuple[float, float]:
        denominator = second[axis] - first[axis]
        if denominator == 0:
            return first
        ratio = (boundary - first[axis]) / denominator
        return (
            first[0] + ratio * (second[0] - first[0]),
            first[1] + ratio * (second[1] - first[1]),
        )

    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous, previous_inside = current, current_inside
    return output


def _ignore_intersection_over_prediction(
    box: list[float], polygon: list[float]
) -> float:
    if not (_valid_box(box) and len(polygon) == 8):
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    points = [
        (float(polygon[index]), float(polygon[index + 1]))
        for index in range(0, 8, 2)
    ]
    for axis, boundary, keep_greater in (
        (0, x1, True),
        (0, x2, False),
        (1, y1, True),
        (1, y2, False),
    ):
        points = _clip_polygon_axis(
            points,
            axis=axis,
            boundary=boundary,
            keep_greater=keep_greater,
        )
    return _polygon_area(points) / ((x2 - x1) * (y2 - y1))


def _filter_icdar2015_dontcare_predictions(
    predictions: Mapping[str, Iterable[Any]],
    ignore_polygons: list[list[float]],
) -> tuple[dict[str, list[list[float]]], int, dict[str, int]]:
    """Remove detections covered by an ICDAR2015 don't-care polygon.

    This follows the official area-precision criterion. The transcription is
    deliberately not used as a bypass: emitting ``###`` only receives
    don't-care treatment when its predicted region actually overlaps a
    corresponding GT ignore polygon.
    """

    kept: dict[str, list[list[float]]] = defaultdict(list)
    removed_by_label: dict[str, int] = defaultdict(int)
    for label, boxes in predictions.items():
        for raw_box in boxes or []:
            box = list(raw_box) if isinstance(raw_box, (list, tuple)) else raw_box
            ignored = _valid_box(box) and any(
                _ignore_intersection_over_prediction(box, polygon)
                > ICDAR2015_DONTCARE_AREA_PRECISION
                for polygon in ignore_polygons
            )
            if ignored:
                removed_by_label[str(label)] += 1
            elif _valid_box(box):
                kept[str(label)].append(list(box))
    return (
        _deduplicate_prediction_mapping(kept),
        sum(removed_by_label.values()),
        dict(sorted(removed_by_label.items())),
    )


def _greedy_matches(
    gt_boxes: list[list[float]], pred_boxes: list[list[float]], threshold: float
) -> int:
    used: set[int] = set()
    matched = 0
    for gt_box in gt_boxes:
        best_index = -1
        best_iou = 0.0
        for index, pred_box in enumerate(pred_boxes):
            if index in used:
                continue
            overlap = _iou(gt_box, pred_box)
            if overlap >= threshold and overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0:
            used.add(best_index)
            matched += 1
    return matched


def _word_unit_matching_enabled(row: Mapping[str, Any]) -> bool:
    return (
        os.environ.get("GAM_OCR_WORD_UNIT_MATCHING", "1").strip().lower()
        in {"1", "true", "yes", "on"}
        and not bool(row.get("boxonly", False))
        and GAM_OCR_UNITS.get(str(row.get("task_name"))) == "word"
    )


def _word_slice_offset() -> float:
    """Return the calibrated word-group IoU margin.

    ``GAM_OCR_WORD_GROUP_IOU_OFFSET`` is retained only as a deprecated audit
    alias so historical sweeps remain reproducible.  The explicitly named
    ``GAM_OCR_WORD_SLICE_OFFSET`` always wins.
    """

    raw = os.environ.get("GAM_OCR_WORD_SLICE_OFFSET")
    if raw is None:
        raw = os.environ.get(
            "GAM_OCR_WORD_GROUP_IOU_OFFSET", str(GAM_OCR_WORD_SLICE_OFFSET)
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("GAM_OCR_WORD_SLICE_OFFSET must be numeric") from exc
    if not 0.0 <= value <= 0.4:
        raise ValueError("GAM_OCR_WORD_SLICE_OFFSET must be within [0, 0.4]")
    return value


def _word_tokens(text: object) -> list[str]:
    return [
        token
        for token in (_normalize_text(piece) for piece in str(text).split())
        if token
    ]


def _intersection_over_box(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return intersection / area if area > 0 else 0.0


def _word_unit_precision_recall(
    row: Mapping[str, Any], threshold: float
) -> tuple[float, float]:
    """Match a phrase box to multiple word-level GT instances.

    A multi-word prediction contributes one predicted unit per emitted token.
    Tokens are matched one-to-one to same-text GT words covered by the phrase
    box; the predicted phrase box must also overlap the union of those GT boxes
    at the requested IoU threshold.  Unmatched emitted tokens remain false
    positives, exactly preserving the requested denominator semantics.
    """

    gt = _prepare_loose(row["gt"])
    predictions = _deduplicate_prediction_mapping(row["predictions"])
    used: dict[str, set[int]] = defaultdict(set)
    predicted_items: list[tuple[list[str], list[float]]] = []
    for text, boxes in predictions.items():
        tokens = _word_tokens(text) or [_normalize_text(text)]
        for box in boxes:
            if _valid_box(box):
                predicted_items.append((tokens, list(box)))

    total_gt = sum(len(boxes) for boxes in gt.values())
    total_pred = sum(len(tokens) for tokens, _ in predicted_items)
    if total_gt == 0:
        return ((1.0, 1.0) if total_pred == 0 else (0.0, 0.0))
    if total_pred == 0:
        return 0.0, 0.0

    matched = 0
    # Resolve phrase predictions before ordinary one-word predictions so a
    # line box is evaluated as one coherent group rather than N copied boxes.
    predicted_items.sort(key=lambda item: len(item[0]), reverse=True)
    for tokens, pred_box in predicted_items:
        if len(tokens) == 1:
            token = tokens[0]
            best_index = -1
            best_iou = 0.0
            for index, gt_box in enumerate(gt.get(token, [])):
                if index in used[token]:
                    continue
                overlap = _iou(gt_box, pred_box)
                if overlap >= threshold and overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0:
                used[token].add(best_index)
                matched += 1
            continue

        selected: list[tuple[str, int, list[float]]] = []
        reserved: dict[str, set[int]] = defaultdict(set)
        for token in tokens:
            best_index = -1
            best_coverage = 0.0
            boxes = gt.get(token, [])
            for index, gt_box in enumerate(boxes):
                if index in used[token] or index in reserved[token]:
                    continue
                coverage = _intersection_over_box(gt_box, pred_box)
                if coverage >= 0.5 and coverage > best_coverage:
                    best_index, best_coverage = index, coverage
            if best_index >= 0:
                reserved[token].add(best_index)
                selected.append((token, best_index, boxes[best_index]))
        # A phrase is a coherent word group.  Do not award a partial group:
        # every emitted token must map to one covered GT word, otherwise the
        # whole prediction follows the ordinary false-positive behavior.
        if len(selected) != len(tokens):
            continue
        union = [
            min(item[2][0] for item in selected),
            min(item[2][1] for item in selected),
            max(item[2][2] for item in selected),
            max(item[2][3] for item in selected),
        ]
        group_threshold = min(0.95, threshold + _word_slice_offset())
        if _iou(union, pred_box) < group_threshold:
            continue
        for token, index, _ in selected:
            used[token].add(index)
        matched += len(selected)

    return matched / total_pred, matched / total_gt


def _prepare_loose(items: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    """Normalize loose transcription keys without dropping colliding boxes."""

    normalized: dict[str, list[list[float]]] = defaultdict(list)
    for text, boxes in items.items():
        key = _normalize_text(str(text).lower())
        normalized[key].extend(
            box for box in boxes or [] if _valid_box(box)
        )
    return _deduplicate_prediction_mapping(normalized)


def _prepared_text_mappings(
    row: Mapping[str, Any], *, strict: bool
) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    """Prepare the two text-to-box mappings once for one scored sample."""

    deduplicated_predictions = _deduplicate_prediction_mapping(row["predictions"])
    if not strict:
        return _prepare_loose(row["gt"]), _prepare_loose(deduplicated_predictions)
    gt = {
        str(text): [box for box in boxes or [] if _valid_box(box)]
        for text, boxes in row["gt"].items()
    }
    pred = {
        str(text): [box for box in boxes or [] if _valid_box(box)]
        for text, boxes in deduplicated_predictions.items()
    }
    return gt, pred


def _precision_recall_sweep(
    row: Mapping[str, Any], *, strict: bool
) -> tuple[list[float], list[float]]:
    """Score all formal IoU thresholds while computing each IoU only once.

    OCR Boxonly pools every text region under one key, so recomputing the full
    GT-by-prediction overlap table at each of ten thresholds is needlessly
    quadratic ten times over.  Candidate order and the ``>`` tie-breaking rule
    below exactly mirror :func:`_greedy_matches`.
    """

    if not strict and _word_unit_matching_enabled(row):
        values = [_word_unit_precision_recall(row, threshold) for threshold in IOU_THRESHOLDS]
        return [value[0] for value in values], [value[1] for value in values]

    gt, pred = _prepared_text_mappings(row, strict=strict)
    total_gt = sum(len(boxes) for boxes in gt.values())
    total_pred = sum(len(boxes) for boxes in pred.values())
    if total_gt == 0:
        value = 1.0 if total_pred == 0 else 0.0
        return [value] * len(IOU_THRESHOLDS), [value] * len(IOU_THRESHOLDS)
    if total_pred == 0:
        return [0.0] * len(IOU_THRESHOLDS), [0.0] * len(IOU_THRESHOLDS)

    minimum_threshold = min(IOU_THRESHOLDS)
    overlap_tables = []
    for text, gt_boxes in gt.items():
        pred_boxes = pred.get(text, [])
        rows = []
        for gt_box in gt_boxes:
            # IoUs below the smallest formal threshold can never participate
            # in any match.  Keeping only candidates avoids a large Python
            # object matrix on dense OCR pages.
            candidates = []
            for index, pred_box in enumerate(pred_boxes):
                overlap = _iou(gt_box, pred_box)
                if overlap >= minimum_threshold:
                    candidates.append((index, overlap))
            rows.append(candidates)
        overlap_tables.append(rows)

    precisions = []
    recalls = []
    for threshold in IOU_THRESHOLDS:
        matched = 0
        for rows in overlap_tables:
            used: set[int] = set()
            for candidates in rows:
                best_index = -1
                best_iou = 0.0
                for index, overlap in candidates:
                    if index in used:
                        continue
                    if overlap >= threshold and overlap > best_iou:
                        best_index, best_iou = index, overlap
                if best_index >= 0:
                    used.add(best_index)
                    matched += 1
        precisions.append(matched / total_pred)
        recalls.append(matched / total_gt)
    return precisions, recalls


def _sample_precision_recall(
    row: Mapping[str, Any], threshold: float, *, strict: bool
) -> tuple[float, float]:
    if not strict and _word_unit_matching_enabled(row):
        return _word_unit_precision_recall(row, threshold)
    gt, pred = _prepared_text_mappings(row, strict=strict)
    total_gt = sum(len(boxes) for boxes in gt.values())
    total_pred = sum(len(boxes) for boxes in pred.values())
    matched = sum(
        _greedy_matches(gt_boxes, pred.get(text, []), threshold)
        for text, gt_boxes in gt.items()
    )

    if total_gt == 0:
        return (1.0, 1.0) if total_pred == 0 else (0.0, 0.0)
    if total_pred == 0:
        return 0.0, 0.0
    return matched / total_pred, matched / total_gt


def _harmonic_mean(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def _sample_loose_miou(row: Mapping[str, Any]) -> float:
    precisions, recalls = _precision_recall_sweep(row, strict=False)
    values = [
        _harmonic_mean(precision, recall)
        for precision, recall in zip(precisions, recalls)
    ]
    return sum(values) / len(values)


def _process_results(doc: Mapping[str, Any], results: list[str], *, boxonly: bool):
    raw_response = results[0] if results else ""
    width, height = int(doc["image_width"]), int(doc["image_height"])
    marker = recorded_image_failure(
        doc.get("task_name", "ocr"),
        doc.get("record_id", doc.get("sample_index", "")),
    )
    if marker:
        marker = dict(marker)
        marker.update(
            {
                "record_id": str(doc.get("record_id", "")),
                "sample_index": int(doc.get("sample_index", -1)),
            }
        )
        return {metric: dict(marker) for metric in METRIC_NAMES}
    if is_native_spatial_mode():
        predictions, parse_diagnostic, parse_recovery_issues = (
            _parse_native_mode_prediction_detailed(raw_response, width, height)
        )
    else:
        predictions, parse_diagnostic = parse_vlm_prediction(raw_response, width, height)
        parse_recovery_issues = ()
    if boxonly and not predictions:
        recovered_boxes = [
            normalized
            for coordinates in fallback_unlabelled_boxes(raw_response)
            if (normalized := _normalize_prediction_box(coordinates, width, height))
            is not None
        ]
        if recovered_boxes:
            predictions = {"text": recovered_boxes}
            parse_diagnostic = "coordinate_only_box_recovered"
    # A non-canonical/truncated response that was safely recovered is a format
    # diagnostic, not a parsing failure.  ``parse_error_rate`` counts only rows
    # that remain unscorable after recovery.
    parse_error = parse_diagnostic if not predictions else None
    gt = json.loads(doc["gt_json"])
    predictions = _deduplicate_prediction_mapping(predictions)
    dontcare_polygons = json.loads(doc.get("dontcare_polygons_json", "[]"))
    dontcare_filtered_count = 0
    dontcare_filtered_labels: dict[str, int] = {}
    if str(doc.get("task_name")) == "gam_icdar2015":
        (
            predictions,
            dontcare_filtered_count,
            dontcare_filtered_labels,
        ) = _filter_icdar2015_dontcare_predictions(
            predictions, dontcare_polygons
        )
    if boxonly:
        gt = {"text": [box for boxes in gt.values() for box in (boxes or [])]}
        predictions = _deduplicate_prediction_mapping(predictions, pool_labels=True)
    payload = {
        "sample_index": int(doc.get("sample_index", -1)),
        "record_id": str(doc["record_id"]),
        "task_name": str(doc["task_name"]),
        "dataset_name": str(doc["dataset_name"]),
        "image_path": str(doc["image_path"]),
        "image_width": width,
        "image_height": height,
        "gt": gt,
        "predictions": predictions,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "parse_diagnostic": parse_diagnostic,
        "parse_recovery_issues": list(parse_recovery_issues),
        "boxonly": bool(boxonly),
        "ocr_unit": GAM_OCR_UNITS.get(str(doc.get("task_name")), "unknown"),
        "dontcare_polygon_count": len(dontcare_polygons),
        "dontcare_filtered_prediction_count": dontcare_filtered_count,
        "dontcare_filtered_prediction_labels": dontcare_filtered_labels,
    }
    precisions, recalls = _precision_recall_sweep(payload, strict=False)
    payload["loose_precisions"] = precisions
    payload["loose_recalls"] = recalls
    payload["sample_score"] = sum(
        _harmonic_mean(precision, recall)
        for precision, recall in zip(precisions, recalls)
    ) / len(IOU_THRESHOLDS)
    return {metric: payload for metric in METRIC_NAMES}


def process_results(doc: Mapping[str, Any], results: list[str]):
    return _process_results(doc, results, boxonly=False)


def process_results_boxonly(doc: Mapping[str, Any], results: list[str]):
    """OCR metric branch that evaluates boxes after pooling transcriptions."""

    return _process_results(doc, results, boxonly=True)


_AGGREGATE_CACHE: dict[
    tuple[str, bool, str, str, tuple[str, ...]], dict[str, float]
] = {}


def _aggregate(results: list[Mapping[str, Any]], task_name: str) -> dict[str, float]:
    rows = [dict(row) for row in valid_results(results, context=f"{task_name}/OCR")]
    key = (
        task_name,
        bool(rows[0].get("boxonly", False)) if rows else False,
        os.environ.get("GAM_OCR_WORD_UNIT_MATCHING", "1"),
        str(_word_slice_offset()),
        tuple(str(row.get("record_id", "")) for row in rows),
    )
    if key in _AGGREGATE_CACHE:
        return _AGGREGATE_CACHE[key]

    precision_rows = []
    recall_rows = []
    for row in rows:
        precisions = row.get("loose_precisions")
        recalls = row.get("loose_recalls")
        if not (
            isinstance(precisions, (list, tuple))
            and isinstance(recalls, (list, tuple))
            and len(precisions) == len(IOU_THRESHOLDS)
            and len(recalls) == len(IOU_THRESHOLDS)
        ):
            precisions, recalls = _precision_recall_sweep(row, strict=False)
        precision_rows.append(precisions)
        recall_rows.append(recalls)

    loose_metrics: dict[float, float] = {}
    for index, threshold in enumerate(IOU_THRESHOLDS):
        precisions = [values[index] for values in precision_rows]
        recalls = [values[index] for values in recall_rows]
        mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        loose_metrics[threshold] = _harmonic_mean(mean_precision, mean_recall)

    metrics = {
        "loose_match_f1_iou_50": loose_metrics[0.50],
        "loose_match_f1_iou_75": loose_metrics[0.75],
        "loose_match_f1_iou_95": loose_metrics[0.95],
        "loose_match_F1mIoU": sum(loose_metrics.values())
        / len(IOU_THRESHOLDS),
        "parse_error_rate": (
            sum(row.get("parse_error") is not None for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }
    _AGGREGATE_CACHE[key] = metrics
    return metrics


def _metric(results, task_name: str, metric_name: str) -> float:
    return float(_aggregate(results, task_name)[metric_name])


def aggregate_parse_error_rate(results, args=None):
    rows = [dict(row) for row in valid_results(results, context="OCR/parse_error")]
    if not rows:
        return 0.0
    return sum(row.get("parse_error") is not None for row in rows) / len(rows)


def aggregate_hiertext_loose_50(results, args=None):
    return _metric(results, "gam_hiertext", "loose_match_f1_iou_50")


def aggregate_hiertext_loose_75(results, args=None):
    return _metric(results, "gam_hiertext", "loose_match_f1_iou_75")


def aggregate_hiertext_loose_95(results, args=None):
    return _metric(results, "gam_hiertext", "loose_match_f1_iou_95")


def aggregate_hiertext_loose_mean(results, args=None):
    return _metric(results, "gam_hiertext", "loose_match_F1mIoU")


def aggregate_hiertext_strict_50(results, args=None):
    return _metric(results, "gam_hiertext", "strict_match_f1_iou_50")


def aggregate_hiertext_strict_95(results, args=None):
    return _metric(results, "gam_hiertext", "strict_match_f1_iou_95")


def aggregate_hiertext_strict_mean(results, args=None):
    return _metric(results, "gam_hiertext", "strict_match_F1mIoU")


def aggregate_icdar2015_loose_50(results, args=None):
    return _metric(results, "gam_icdar2015", "loose_match_f1_iou_50")


def aggregate_icdar2015_loose_75(results, args=None):
    return _metric(results, "gam_icdar2015", "loose_match_f1_iou_75")


def aggregate_icdar2015_loose_95(results, args=None):
    return _metric(results, "gam_icdar2015", "loose_match_f1_iou_95")


def aggregate_icdar2015_loose_mean(results, args=None):
    return _metric(results, "gam_icdar2015", "loose_match_F1mIoU")


def aggregate_icdar2015_strict_50(results, args=None):
    return _metric(results, "gam_icdar2015", "strict_match_f1_iou_50")


def aggregate_icdar2015_strict_95(results, args=None):
    return _metric(results, "gam_icdar2015", "strict_match_f1_iou_95")


def aggregate_icdar2015_strict_mean(results, args=None):
    return _metric(results, "gam_icdar2015", "strict_match_F1mIoU")


def aggregate_totaltext_loose_50(results, args=None):
    return _metric(results, "gam_totaltext", "loose_match_f1_iou_50")


def aggregate_totaltext_loose_75(results, args=None):
    return _metric(results, "gam_totaltext", "loose_match_f1_iou_75")


def aggregate_totaltext_loose_95(results, args=None):
    return _metric(results, "gam_totaltext", "loose_match_f1_iou_95")


def aggregate_totaltext_loose_mean(results, args=None):
    return _metric(results, "gam_totaltext", "loose_match_F1mIoU")


def aggregate_totaltext_strict_50(results, args=None):
    return _metric(results, "gam_totaltext", "strict_match_f1_iou_50")


def aggregate_totaltext_strict_95(results, args=None):
    return _metric(results, "gam_totaltext", "strict_match_f1_iou_95")


def aggregate_totaltext_strict_mean(results, args=None):
    return _metric(results, "gam_totaltext", "strict_match_F1mIoU")


def aggregate_sroie_loose_50(results, args=None):
    return _metric(results, "gam_sroie", "loose_match_f1_iou_50")


def aggregate_sroie_loose_75(results, args=None):
    return _metric(results, "gam_sroie", "loose_match_f1_iou_75")


def aggregate_sroie_loose_95(results, args=None):
    return _metric(results, "gam_sroie", "loose_match_f1_iou_95")


def aggregate_sroie_loose_mean(results, args=None):
    return _metric(results, "gam_sroie", "loose_match_F1mIoU")


def aggregate_sroie_strict_50(results, args=None):
    return _metric(results, "gam_sroie", "strict_match_f1_iou_50")


def aggregate_sroie_strict_95(results, args=None):
    return _metric(results, "gam_sroie", "strict_match_f1_iou_95")


def aggregate_sroie_strict_mean(results, args=None):
    return _metric(results, "gam_sroie", "strict_match_F1mIoU")
