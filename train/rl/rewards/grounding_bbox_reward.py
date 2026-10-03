"""Reward functions for JSON and compact LVIS bbox GRPO outputs.

Expected model output:

<bbox>
<label>[x1,y1,x2,y2|...]#N
</bbox>

Expected dataset fields:
- solution.bbox_norm1000: dict[label, list[[x1,y1,x2,y2]]]
- target_labels: list[str]

This file is importable by Swift-style GRPO jobs and registers three reward
variants for controlled ablations. The legacy `grounding_bbox_reward` name is
kept as an alias of the original task319 formula.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Callable


BBOX_BLOCK_RE = re.compile(r"<bbox>\s*(.*?)\s*</bbox>", re.S)
BBOX_LINE_RE = re.compile(r"^<([^<>]+)>\[([^\]]*)\]#(\d+)$")
BOX_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
COORD_RE = re.compile(r"(?<!\d)(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})(?!\d)")
LABEL_SUFFIX_RE = re.compile(r"\|\s*([A-Za-z][A-Za-z0-9_() -]{0,80})")
GAM_GROUP_RE = re.compile(
    r"<\|object_ref_start\|>\s*(.*?)\s*<\|object_ref_end\|>\s*"
    r"<\|box_start\|>\s*(.*?)\s*<\|box_end\|>",
    re.S,
)
GAM_BOX_RE = re.compile(
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
)
GAM_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
UNLABELED = "__unlabeled__"
IOU_THRESHOLDS = (0.3, 0.5, 0.75)
STRICT_BASELINE_IOU_THRESHOLDS = (0.5, 0.75, 0.95)
RECALL_BETA = 2.0
SMALL_OBJECT_MEDIAN_AREA_GATE = 0.0015

# Exact task319 aggregation, retained as the experiment baseline.
ORIGINAL_REWARD_WEIGHTS = {
    "format": 0.10,
    "count": 0.10,
    "f1": 0.50,
    "matched_iou": 0.25,
    "sort": 0.05,
}

# Recall-dominant thresholded reward. Recall, F2 and GT coverage carry 70%.
RECALL_V2_REWARD_WEIGHTS = {
    "format": 0.05,
    "count": 0.10,
    "precision": 0.10,
    "recall": 0.25,
    "f2": 0.25,
    "gt_coverage": 0.20,
    "sort": 0.05,
}

# Recommended reward: combine hard detection metrics with dense bidirectional
# IoU coverage. Dense terms remain informative before boxes cross an IoU
# threshold, while F2 and GT coverage retain a deliberate recall preference.
BEST_V3_REWARD_WEIGHTS = {
    "format": 0.05,
    "count": 0.10,
    "hard_recall": 0.15,
    "hard_f2": 0.20,
    "soft_f2": 0.30,
    "gt_coverage": 0.15,
    "sort": 0.05,
}

# Density-adaptive Pareto reward. Sparse examples use F1-like balance to
# preserve general localization, while dense examples transition smoothly to
# F2. Precision and strict format receive explicit anchors so recall gains do
# not come from verbose false positives or malformed/truncated JSON.
PARETO_V4_REWARD_WEIGHTS = {
    "format": 0.10,
    "precision": 0.15,
    "recall": 0.05,
    "adaptive_hard_fbeta": 0.20,
    "adaptive_soft_fbeta": 0.25,
    "balanced_count": 0.10,
    "gt_coverage": 0.10,
    "sort": 0.05,
}

# Recall-balanced follow-up to v4. The v4 probe showed higher precision but
# lower recall and pred/GT ratios than best_v3, especially on dense examples.
PARETO_V5_REWARD_WEIGHTS = {
    "format": 0.08,
    "precision": 0.05,
    "recall": 0.15,
    "adaptive_hard_fbeta": 0.20,
    "adaptive_soft_fbeta": 0.20,
    "balanced_count": 0.10,
    "gt_coverage": 0.17,
    "sort": 0.05,
}

# Dense-only residual objective. Sparse examples retain the proven best_v3
# objective exactly; the blend increases smoothly from 0 above 10 GT boxes to
# 1 at 100 GT boxes.
PARETO_V6_DENSE_REWARD_WEIGHTS = {
    "format": 0.05,
    "precision": 0.05,
    "recall": 0.20,
    "adaptive_hard_fbeta": 0.20,
    "adaptive_soft_fbeta": 0.15,
    "balanced_count": 0.10,
    "gt_coverage": 0.25,
}

# Extreme-density tail objective. It preserves v5 below 60 GT boxes and only
# increases recall/count pressure where long dense outputs still undercount.
PARETO_V7_EXTREME_REWARD_WEIGHTS = {
    "format": 0.05,
    "precision": 0.03,
    "recall": 0.27,
    "adaptive_hard_fbeta": 0.20,
    "adaptive_soft_fbeta": 0.15,
    "balanced_count": 0.10,
    "gt_coverage": 0.20,
}

# Follow-up to v7: retain the extra dense predictions, but require stronger
# localization/precision evidence so count gains do not become false positives.
PARETO_V8_EXTREME_REWARD_WEIGHTS = {
    "format": 0.05,
    "precision": 0.10,
    "recall": 0.20,
    "adaptive_hard_fbeta": 0.25,
    "adaptive_soft_fbeta": 0.20,
    "balanced_count": 0.08,
    "gt_coverage": 0.12,
}

ORIGINAL_BIGBOX_PENALTY_WEIGHT = 0.07
ORIGINAL_DUPLICATE_PENALTY_WEIGHT = 0.03
RECALL_V2_BIGBOX_PENALTY_WEIGHT = 0.07
RECALL_V2_DUPLICATE_PENALTY_WEIGHT = 0.05
RECALL_V2_UNDER_COUNT_PENALTY_WEIGHT = 0.10
BEST_V3_BIGBOX_PENALTY_WEIGHT = 0.07
BEST_V3_DUPLICATE_PENALTY_WEIGHT = 0.05
BEST_V3_UNDER_COUNT_PENALTY_WEIGHT = 0.12
BEST_V3_SOFT_FP_PENALTY_WEIGHT = 0.08
PARETO_V4_BIGBOX_PENALTY_WEIGHT = 0.05
PARETO_V4_DUPLICATE_PENALTY_WEIGHT = 0.05
PARETO_V4_OVER_COUNT_PENALTY_WEIGHT = 0.08
PARETO_V4_DENSE_UNDER_COUNT_PENALTY_WEIGHT = 0.04
PARETO_V5_BIGBOX_PENALTY_WEIGHT = 0.05
PARETO_V5_DUPLICATE_PENALTY_WEIGHT = 0.03
PARETO_V5_OVER_COUNT_PENALTY_WEIGHT = 0.02
PARETO_V5_BASE_UNDER_COUNT_PENALTY_WEIGHT = 0.08
PARETO_V5_DENSE_UNDER_COUNT_PENALTY_WEIGHT = 0.08
PARETO_V6_BIGBOX_PENALTY_WEIGHT = 0.05
PARETO_V6_DUPLICATE_PENALTY_WEIGHT = 0.03
PARETO_V6_OVER_COUNT_PENALTY_WEIGHT = 0.02
PARETO_V6_UNDER_COUNT_PENALTY_WEIGHT = 0.18
PARETO_V7_BIGBOX_PENALTY_WEIGHT = 0.05
PARETO_V7_DUPLICATE_PENALTY_WEIGHT = 0.03
PARETO_V7_OVER_COUNT_PENALTY_WEIGHT = 0.01
PARETO_V7_UNDER_COUNT_PENALTY_WEIGHT = 0.26
PARETO_V7_EXTREME_START = 60
PARETO_V7_EXTREME_FULL = 140
PARETO_V8_BIGBOX_PENALTY_WEIGHT = 0.05
PARETO_V8_DUPLICATE_PENALTY_WEIGHT = 0.05
PARETO_V8_OVER_COUNT_PENALTY_WEIGHT = 0.05
PARETO_V8_UNDER_COUNT_PENALTY_WEIGHT = 0.18


def _as_solution(obj: Any) -> dict[str, Any]:
    if isinstance(obj, str):
        return json.loads(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported solution type: {type(obj)!r}")


def strip_json_fence(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"<\|[^>]+?\|>", "", s).strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    if not s.startswith("[") or not s.endswith("]"):
        start = s.find("[")
        end = s.rfind("]")
        if 0 <= start < end:
            s = s[start : end + 1].strip()
    return s


def _valid_box(vals: list[int]) -> bool:
    x1, y1, x2, y2 = vals
    return 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000 and x1 < x2 and y1 < y2


def _coerce_box(vals: list[int]) -> list[int] | None:
    vals = [max(0, min(1000, int(v))) for v in vals]
    return vals if vals[0] < vals[2] and vals[1] < vals[3] else None


def parse_json_bbox(text: str) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    meta = {"format_ok": 0.0, "bad_items": [], "json_error": None}
    try:
        data = json.loads(strip_json_fence(text))
    except Exception as exc:
        meta["json_error"] = repr(exc)
        return {}, meta
    if not isinstance(data, list):
        meta["json_error"] = "json_not_list"
        return {}, meta
    result: dict[str, list[list[int]]] = {}
    for item in data:
        if not isinstance(item, dict):
            meta["bad_items"].append(str(item)[:120])
            continue
        label = item.get("label")
        box = item.get("bbox_2d") or item.get("bbox")
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            meta["bad_items"].append(str(item)[:120])
            continue
        vals = _coerce_box([int(x) for x in box])
        if vals is None:
            meta["bad_items"].append(str(item)[:120])
            continue
        result.setdefault(label, []).append(vals)
    if not meta["bad_items"]:
        meta["format_ok"] = 1.0
    elif result:
        meta["format_ok"] = 0.5
    return result, meta


def parse_json_bbox_rexomni_paper(text: str) -> dict[str, list[list[int]]]:
    """Adapt Rex-Omni's released box parser to this repo's JSON output."""
    try:
        data = json.loads(strip_json_fence(text))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    result: dict[str, list[list[int]]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("bbox_2d") or item.get("bbox")
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            vals = [max(0, min(1000, int(value))) for value in box]
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((vals[0], vals[2]))
        y1, y2 = sorted((vals[1], vals[3]))
        if x1 < x2 and y1 < y2:
            result.setdefault(label.strip(), []).append([x1, y1, x2, y2])
    return result


def parse_bbox_rexomni_paper(
    text: str, expected_format: str
) -> dict[str, list[list[int]]]:
    """Decode boxes without changing the paper's matching or reward equations."""
    if expected_format.startswith("gam_bbox"):
        parsed, _ = parse_gam_bbox(text)
        return parsed
    if expected_format.startswith("compact_bbox"):
        parsed, _ = parse_compact_bbox(text)
        return parsed
    return parse_json_bbox_rexomni_paper(text)


def _extract_boxes(text: str) -> list[list[int]]:
    boxes: list[list[int]] = []
    for mm in COORD_RE.finditer(text):
        vals = _coerce_box([int(mm.group(i)) for i in range(1, 5)])
        if vals is not None:
            boxes.append(vals)
    return boxes


def _label_from_line(cleaned: str) -> str:
    label = cleaned.split("[", 1)[0].strip() if "[" in cleaned else ""
    label = re.sub(r"^<|>$", "", label).strip()
    label = re.sub(r"^(label|bbox|box)\s*[:>]*\s*", "", label, flags=re.I).strip()
    if label and label.lower() not in {"label", "bbox", "box", "div", "json"}:
        return re.sub(r"\s+", "_", label)
    first = COORD_RE.search(cleaned)
    if first:
        suffix = cleaned[first.end() : first.end() + 120]
        sm = LABEL_SUFFIX_RE.search(suffix)
        if sm:
            label = sm.group(1).strip()
            if label.lower() not in {"label", "bbox", "box"}:
                return re.sub(r"\s+", "_", label)
    return UNLABELED


def _parse_weak_bbox_line(line: str) -> tuple[str, list[list[int]]] | None:
    # Common Qwen grounding prior: <label>street_sign [x1,y1,x2,y2]#0</label>
    cleaned = re.sub(r"^<label>\s*", "", line.strip())
    cleaned = re.sub(r"\s*</label>\s*$", "", cleaned)
    cleaned = re.sub(r"<\|[^>]+?\|>", "", cleaned)
    boxes = _extract_boxes(cleaned)
    if not boxes:
        return None
    return _label_from_line(cleaned), boxes


def parse_compact_bbox(text: str) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    meta = {
        "format_ok": 0.0,
        "count_mismatches": [],
        "bad_lines": [],
        "bad_boxes": [],
        "duplicate_labels": [],
        "weak_lines": [],
        "missing_bbox": False,
        "missing_close": False,
    }
    text = text or ""
    m = BBOX_BLOCK_RE.search(text)
    if not m:
        start = re.search(r"<bbox>\s*", text)
        if not start:
            body = text.strip()
            meta["missing_bbox"] = True
        else:
            body = text[start.end() :].strip()
            meta["missing_close"] = True
    else:
        body = m.group(1).strip()
    result: dict[str, list[list[int]]] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        mm = BBOX_LINE_RE.match(line)
        if not mm:
            weak = _parse_weak_bbox_line(line)
            if weak is None:
                meta["bad_lines"].append(line[:120])
                continue
            label, boxes = weak
            result.setdefault(label, []).extend(boxes)
            meta["weak_lines"].append(line[:120])
            continue
        label, inside, n_s = mm.group(1), mm.group(2), mm.group(3)
        if label in result:
            meta["duplicate_labels"].append(label)
        boxes: list[list[int]] = []
        if inside.strip():
            for item in inside.split("|"):
                try:
                    vals = [int(x) for x in item.split(",")]
                except ValueError:
                    meta["bad_boxes"].append([label, item])
                    continue
                if len(vals) != 4:
                    meta["bad_boxes"].append([label, item])
                    continue
                if not _valid_box(vals):
                    meta["bad_boxes"].append([label, vals])
                    continue
                boxes.append(vals)
        expected_count = int(n_s)
        if expected_count != len(boxes):
            meta["count_mismatches"].append([label, expected_count, len(boxes)])
        result[label] = boxes
    if (
        not meta["missing_close"]
        and not meta["weak_lines"]
        and not meta["bad_lines"]
        and not meta["bad_boxes"]
        and not meta["duplicate_labels"]
        and not meta["count_mismatches"]
    ):
        meta["format_ok"] = 1.0
    elif result and meta["weak_lines"]:
        meta["format_ok"] = 0.1 if meta["missing_bbox"] else 0.25
    elif result:
        meta["format_ok"] = 0.5
    return result, meta


def parse_gam_bbox(text: str) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    """Parse GAM-native labeled coordinate-token groups.

    Expected output:
    ``<|object_ref_start|>label<|object_ref_end|><|box_start|>``
    ``<x1><y1><x2><y2>,...<|box_end|>``.
    """
    meta = {
        "format_ok": 0.0,
        "bad_groups": [],
        "bad_boxes": [],
        "duplicate_labels": [],
        "residual_text": "",
        "missing_groups": False,
    }
    cleaned = GAM_THINK_RE.sub("", text or "")
    cleaned = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", "", cleaned)
    matches = list(GAM_GROUP_RE.finditer(cleaned))
    if not matches:
        meta["missing_groups"] = True
        meta["residual_text"] = cleaned.strip()[:160]
        return {}, meta

    result: dict[str, list[list[int]]] = {}
    for match in matches:
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        body = match.group(2).strip()
        if not label:
            meta["bad_groups"].append("empty_label")
            continue
        if label in result:
            meta["duplicate_labels"].append(label)
        boxes: list[list[int]] = []
        for box_match in GAM_BOX_RE.finditer(body):
            vals = [int(box_match.group(i)) for i in range(1, 5)]
            if _valid_box(vals):
                boxes.append(vals)
            else:
                meta["bad_boxes"].append([label, vals])
        body_residual = GAM_BOX_RE.sub("", body)
        body_residual = re.sub(r"[\s,]", "", body_residual)
        if body_residual:
            meta["bad_groups"].append([label, body_residual[:120]])
        if not boxes:
            meta["bad_groups"].append([label, "no_valid_boxes"])
        result.setdefault(label, []).extend(boxes)

    outside = GAM_GROUP_RE.sub("", cleaned)
    outside = re.sub(r"[\s,]", "", outside)
    if outside:
        meta["residual_text"] = outside[:160]

    has_errors = any(
        (
            meta["bad_groups"],
            meta["bad_boxes"],
            meta["duplicate_labels"],
            meta["residual_text"],
        )
    )
    if result and not has_errors:
        meta["format_ok"] = 1.0
    elif any(result.values()):
        meta["format_ok"] = 0.5
    return result, meta


def box_area(b: list[int]) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def box_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _match_pairs(pred: list[list[int]], gt: list[list[int]]) -> list[tuple[int, int, float]]:
    if not pred or not gt:
        return []
    ious = [[box_iou(p, g) for g in gt] for p in pred]
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment([[-x for x in row] for row in ious])
        return [(int(r), int(c), ious[int(r)][int(c)]) for r, c in zip(rows, cols)]
    except Exception:
        triples = sorted(
            ((ious[i][j], i, j) for i in range(len(pred)) for j in range(len(gt))),
            reverse=True,
        )
        used_p: set[int] = set()
        used_g: set[int] = set()
        out: list[tuple[int, int, float]] = []
        for iou, i, j in triples:
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, iou))
        return out


def precision_recall_fbeta_at_threshold(
    pred_by_label: dict[str, list[list[int]]],
    gt_by_label: dict[str, list[list[int]]],
    labels: list[str],
    threshold: float,
    beta: float = RECALL_BETA,
) -> tuple[float, float, float, float]:
    tp = fp = fn = 0
    for label in labels:
        pred = pred_by_label.get(label, [])
        gt = gt_by_label.get(label, [])
        pairs = _match_pairs(pred, gt)
        matched = sum(1 for _, _, iou in pairs if iou >= threshold)
        tp += matched
        fp += max(0, len(pred) - matched)
        fn += max(0, len(gt) - matched)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if beta2 * precision + recall else 0.0
    return precision, recall, f1, fbeta


def f1_at_threshold(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str], threshold: float) -> float:
    return precision_recall_fbeta_at_threshold(pred_by_label, gt_by_label, labels, threshold, beta=1.0)[2]


def mean_matched_iou(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    vals = []
    for label in labels:
        vals.extend(iou for _, _, iou in _match_pairs(pred_by_label.get(label, []), gt_by_label.get(label, [])))
    return sum(vals) / len(vals) if vals else 0.0


def gt_coverage_iou(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    """Average best IoU for every GT box, counting missed GT boxes as zero."""
    vals = []
    for label in labels:
        pred = pred_by_label.get(label, [])
        for gt in gt_by_label.get(label, []):
            vals.append(max((box_iou(p, gt) for p in pred), default=0.0))
    return sum(vals) / len(vals) if vals else 1.0


def pred_coverage_iou(
    pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]
) -> float:
    """Average best IoU for every prediction, counting false positives as zero."""
    vals = []
    for label in labels:
        gt = gt_by_label.get(label, [])
        for pred in pred_by_label.get(label, []):
            vals.append(max((box_iou(pred, box) for box in gt), default=0.0))
    if vals:
        return sum(vals) / len(vals)
    has_gt = any(gt_by_label.get(label, []) for label in labels)
    return 0.0 if has_gt else 1.0


def fbeta_from_precision_recall(precision: float, recall: float, beta: float = RECALL_BETA) -> float:
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denominator if denominator else 0.0


def density_adaptive_beta(gt_total: int) -> float:
    """Use beta=1 up to 10 GT boxes and increase log-linearly to beta=2 at 100."""
    if gt_total <= 10:
        return 1.0
    return 1.0 + min(1.0, math.log10(gt_total / 10.0))


def balanced_count_reward(
    pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]
) -> float:
    """Symmetric, smooth count agreement in log space for each requested label."""
    vals = []
    for label in labels:
        pred_count = len(pred_by_label.get(label, []))
        gt_count = len(gt_by_label.get(label, []))
        vals.append(math.exp(-abs(math.log((pred_count + 1.0) / (gt_count + 1.0)))))
    return sum(vals) / len(vals) if vals else 1.0


def over_count_penalty(
    pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]
) -> float:
    gt_total = sum(len(gt_by_label.get(label, [])) for label in labels)
    extra = sum(
        max(0, len(pred_by_label.get(label, [])) - len(gt_by_label.get(label, []))) for label in labels
    )
    return min(1.0, extra / max(1, gt_total))


def count_reward(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    vals = []
    for label in labels:
        p = len(pred_by_label.get(label, []))
        g = len(gt_by_label.get(label, []))
        vals.append(math.exp(-abs(p - g) / max(1, g)))
    return sum(vals) / len(vals) if vals else 1.0


def recall_count_reward(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    """Count score that penalizes missing boxes twice as much as extra boxes."""
    vals = []
    for label in labels:
        p = len(pred_by_label.get(label, []))
        g = len(gt_by_label.get(label, []))
        missing = max(0, g - p)
        extra = max(0, p - g)
        vals.append(math.exp(-(2.0 * missing + extra) / max(1, g)))
    return sum(vals) / len(vals) if vals else 1.0


def under_count_penalty(
    pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]
) -> float:
    gt_total = sum(len(gt_by_label.get(label, [])) for label in labels)
    if gt_total == 0:
        return 0.0
    missing = sum(
        max(0, len(gt_by_label.get(label, [])) - len(pred_by_label.get(label, []))) for label in labels
    )
    return missing / gt_total


def sort_reward(pred_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    vals = []
    for label in labels:
        boxes = pred_by_label.get(label, [])
        n = len(boxes)
        if n < 2:
            vals.append(1.0)
            continue
        keys = [(round(b[1] / 20), b[0], b[1], b[2], b[3]) for b in boxes]
        inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                inv += int(keys[i] > keys[j])
        vals.append(1.0 - inv / (n * (n - 1) / 2))
    return sum(vals) / len(vals) if vals else 1.0


def bigbox_penalty(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    penalties = []
    for label in labels:
        pred = pred_by_label.get(label, [])
        gt = gt_by_label.get(label, [])
        for pi, gi, iou in _match_pairs(pred, gt):
            if iou <= 0:
                continue
            pa = max(1.0, box_area(pred[pi]))
            ga = max(1.0, box_area(gt[gi]))
            ratio = pa / ga
            penalties.append(max(0.0, math.log(max(1.0, ratio)) / math.log(4.0)))
        for p in pred:
            if box_area(p) / 1_000_000 > 0.5:
                penalties.append(1.0)
    return min(1.0, sum(penalties) / len(penalties)) if penalties else 0.0


def duplicate_penalty(pred_by_label: dict[str, list[list[int]]], labels: list[str]) -> float:
    dup = total = 0
    for label in labels:
        boxes = pred_by_label.get(label, [])
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total += 1
                dup += int(box_iou(boxes[i], boxes[j]) >= 0.7)
    return dup / total if total else 0.0


def assign_unlabeled_boxes(pred_by_label: dict[str, list[list[int]]], gt_by_label: dict[str, list[list[int]]], labels: list[str]) -> dict[str, list[list[int]]]:
    out = {label: [box[:] for box in boxes] for label, boxes in pred_by_label.items() if label != UNLABELED}
    unlabeled = pred_by_label.get(UNLABELED, [])
    valid_labels = [label for label in labels if label != UNLABELED]
    if not unlabeled or not valid_labels:
        return out
    for box in unlabeled:
        best_label = valid_labels[0]
        best_iou = -1.0
        for label in valid_labels:
            for gt in gt_by_label.get(label, []):
                iou = box_iou(box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_label = label
        out.setdefault(best_label, []).append(box)
    return out


def solution_bbox_by_label(solution: dict[str, Any]) -> dict[str, list[list[int]]]:
    bbox = solution.get("bbox_norm1000")
    if isinstance(bbox, dict):
        return {str(label): [[int(x) for x in box] for box in boxes] for label, boxes in bbox.items()}
    result: dict[str, list[list[int]]] = {}
    for item in solution.get("items") or []:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("bbox_2d") or item.get("bbox")
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            continue
        vals = _coerce_box([int(x) for x in box])
        if vals is not None:
            result.setdefault(label, []).append(vals)
    return result


@dataclass
class RewardBreakdown:
    reward: float
    format_reward: float
    count_reward: float
    f1_reward: float
    iou_reward: float
    sort_reward: float
    bigbox_penalty: float
    duplicate_penalty: float
    precision_reward: float = 0.0
    recall_reward: float = 0.0
    f2_reward: float = 0.0
    gt_coverage_reward: float = 0.0
    under_count_penalty: float = 0.0
    recall_count_reward: float = 0.0
    pred_coverage_reward: float = 0.0
    soft_f2_reward: float = 0.0
    excess_duplicate_penalty: float = 0.0
    adaptive_beta: float = 1.0
    adaptive_fbeta_reward: float = 0.0
    adaptive_soft_fbeta_reward: float = 0.0
    balanced_count_reward: float = 0.0
    over_count_penalty: float = 0.0
    precision_iou50: float = 0.0
    recall_iou50: float = 0.0
    f2_iou50: float = 0.0
    pred_gt_ratio: float = 0.0
    size_aware_gt_coverage: float = 0.0


@dataclass(frozen=True)
class RexOmniPaperRewardBreakdown:
    reward: float
    precision: float
    recall: float
    pred_count: int
    gt_count: int


def score_completion_rexomni_paper(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RexOmniPaperRewardBreakdown:
    """Reproduce Rex-Omni paper equations 3-4 for the configured box format.

    Each GT first selects the highest-IoU prediction without label filtering.
    That IoU contributes only when the selected prediction label matches the
    GT label. The same matched-IoU sum is divided by GT and prediction counts
    to form soft recall and precision, respectively. ``target_labels`` is
    accepted for the Swift reward interface but deliberately does not filter
    either set.
    """
    del target_labels
    gt_by_label = solution_bbox_by_label(solution)
    pred_by_label = parse_bbox_rexomni_paper(
        completion, str(solution.get("format") or "")
    )
    all_gt = [(box, label) for label, boxes in gt_by_label.items() for box in boxes]
    all_pred = [(box, label) for label, boxes in pred_by_label.items() for box in boxes]
    gt_count = len(all_gt)
    pred_count = len(all_pred)

    if gt_count == 0 and pred_count == 0:
        return RexOmniPaperRewardBreakdown(1.0, 1.0, 1.0, pred_count, gt_count)
    if gt_count == 0:
        return RexOmniPaperRewardBreakdown(0.0, 0.0, 0.0, pred_count, gt_count)

    matched_iou_sum = 0.0
    for gt_box, gt_label in all_gt:
        best_iou = 0.0
        best_pred_label = None
        for pred_box, pred_label in all_pred:
            iou = box_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_label = pred_label
        if best_pred_label == gt_label:
            matched_iou_sum += best_iou

    recall = matched_iou_sum / gt_count
    precision = matched_iou_sum / pred_count if pred_count else 0.0
    reward = (
        2.0 * precision * recall / (precision + recall + 1e-8)
        if precision + recall
        else 0.0
    )
    return RexOmniPaperRewardBreakdown(reward, precision, recall, pred_count, gt_count)


def size_aware_gt_coverage_iou(
    pred_by_label: dict[str, list[list[int]]],
    gt_by_label: dict[str, list[list[int]]],
    labels: list[str],
) -> float:
    """Average best-match IoU with inverse-sqrt GT area weighting.

    Coordinates are normalized to 0..1000. Inverse-sqrt area prevents a few
    large objects from masking localization failures on VisDrone-scale boxes.
    """
    weighted_iou = 0.0
    weight_sum = 0.0
    for label in labels:
        preds = pred_by_label.get(label, [])
        for gt in gt_by_label.get(label, []):
            width = max(1, gt[2] - gt[0])
            height = max(1, gt[3] - gt[1])
            weight = 1.0 / math.sqrt(width * height)
            weighted_iou += weight * max((box_iou(pred, gt) for pred in preds), default=0.0)
            weight_sum += weight
    return weighted_iou / weight_sum if weight_sum else 0.0


def _score_components(
    completion: str,
    solution: dict[str, Any],
    target_labels: list[str] | None = None,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> RewardBreakdown:
    gt_by_label = solution_bbox_by_label(solution)
    target = target_labels or solution.get("target_labels") or list(gt_by_label)
    expected_format = str(solution.get("format") or "")
    parser = "compact"
    if expected_format.startswith("gam_bbox"):
        pred_by_label, meta = parse_gam_bbox(completion)
        parser = "gam"
    elif expected_format == "qwen_json_bbox_v1":
        pred_by_label, meta = parse_json_bbox(completion)
        parser = "json"
        if not pred_by_label and meta.get("format_ok", 0.0) == 0.0:
            pred_by_label, meta = parse_compact_bbox(completion)
            parser = "weak"
    else:
        pred_by_label, meta = parse_compact_bbox(completion)
    base_labels = list(dict.fromkeys([*target, *gt_by_label.keys()]))
    pred_by_label = assign_unlabeled_boxes(pred_by_label, gt_by_label, base_labels)
    labels = list(dict.fromkeys([*base_labels, *pred_by_label.keys()]))
    r_format = float(meta["format_ok"])
    if expected_format == "qwen_json_bbox_v1" and parser != "json":
        r_format = min(r_format, 0.1)
    gt_total = sum(len(v) for v in gt_by_label.values())
    pred_total = sum(len(v) for v in pred_by_label.values())
    if expected_format == "qwen_json_bbox_v1" and gt_total > 0 and pred_total == 0:
        r_format = min(r_format, 0.2)
    if r_format == 0.0 and not pred_by_label:
        return RewardBreakdown(0.0, r_format, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r_count = count_reward(pred_by_label, gt_by_label, labels)
    r_recall_count = recall_count_reward(pred_by_label, gt_by_label, labels)
    threshold_metrics = [
        precision_recall_fbeta_at_threshold(pred_by_label, gt_by_label, labels, threshold)
        for threshold in iou_thresholds
    ]
    r_precision = sum(x[0] for x in threshold_metrics) / len(threshold_metrics)
    r_recall = sum(x[1] for x in threshold_metrics) / len(threshold_metrics)
    r_f1 = sum(x[2] for x in threshold_metrics) / len(threshold_metrics)
    r_f2 = sum(x[3] for x in threshold_metrics) / len(threshold_metrics)
    r_iou = mean_matched_iou(pred_by_label, gt_by_label, labels)
    r_pred_coverage = pred_coverage_iou(pred_by_label, gt_by_label, labels)
    r_gt_coverage = gt_coverage_iou(pred_by_label, gt_by_label, labels)
    r_soft_f2 = fbeta_from_precision_recall(r_pred_coverage, r_gt_coverage)
    adaptive_beta = density_adaptive_beta(gt_total)
    r_adaptive_fbeta = sum(
        fbeta_from_precision_recall(metrics[0], metrics[1], adaptive_beta) for metrics in threshold_metrics
    ) / len(threshold_metrics)
    r_adaptive_soft_fbeta = fbeta_from_precision_recall(
        r_pred_coverage, r_gt_coverage, adaptive_beta
    )
    r_balanced_count = balanced_count_reward(pred_by_label, gt_by_label, labels)
    precision_iou50, recall_iou50, _, f2_iou50 = precision_recall_fbeta_at_threshold(
        pred_by_label, gt_by_label, labels, 0.5
    )
    pred_gt_ratio = pred_total / gt_total if gt_total else float(pred_total == 0)
    r_size_aware_gt_coverage = size_aware_gt_coverage_iou(pred_by_label, gt_by_label, labels)
    if expected_format == "qwen_json_bbox_v1":
        r_sort = 1.0 if pred_total > 0 else 0.0
    else:
        r_sort = sort_reward(pred_by_label, labels)
    p_big = bigbox_penalty(pred_by_label, gt_by_label, labels)
    p_dup = duplicate_penalty(pred_by_label, labels)
    p_excess_dup = (
        p_dup
        if any(len(pred_by_label.get(label, [])) > len(gt_by_label.get(label, [])) for label in labels)
        else 0.0
    )
    p_under = under_count_penalty(pred_by_label, gt_by_label, labels)
    p_over = over_count_penalty(pred_by_label, gt_by_label, labels)
    return RewardBreakdown(
        reward=0.0,
        format_reward=r_format,
        count_reward=r_count,
        f1_reward=r_f1,
        iou_reward=r_iou,
        sort_reward=r_sort,
        bigbox_penalty=p_big,
        duplicate_penalty=p_dup,
        precision_reward=r_precision,
        recall_reward=r_recall,
        f2_reward=r_f2,
        gt_coverage_reward=r_gt_coverage,
        under_count_penalty=p_under,
        recall_count_reward=r_recall_count,
        pred_coverage_reward=r_pred_coverage,
        soft_f2_reward=r_soft_f2,
        excess_duplicate_penalty=p_excess_dup,
        adaptive_beta=adaptive_beta,
        adaptive_fbeta_reward=r_adaptive_fbeta,
        adaptive_soft_fbeta_reward=r_adaptive_soft_fbeta,
        balanced_count_reward=r_balanced_count,
        over_count_penalty=p_over,
        precision_iou50=precision_iou50,
        recall_iou50=recall_iou50,
        f2_iou50=f2_iou50,
        pred_gt_ratio=pred_gt_ratio,
        size_aware_gt_coverage=r_size_aware_gt_coverage,
    )


def _clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_completion_original(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Task319 baseline aggregation with the shared JSON/compact parser."""
    score = _score_components(completion, solution, target_labels)
    if score.format_reward == 0.0 and not any(
        (score.count_reward, score.f1_reward, score.iou_reward, score.sort_reward)
    ):
        return score
    reward = (
        ORIGINAL_REWARD_WEIGHTS["format"] * score.format_reward
        + ORIGINAL_REWARD_WEIGHTS["count"] * score.count_reward
        + ORIGINAL_REWARD_WEIGHTS["f1"] * score.f1_reward
        + ORIGINAL_REWARD_WEIGHTS["matched_iou"] * score.iou_reward
        + ORIGINAL_REWARD_WEIGHTS["sort"] * score.sort_reward
        - ORIGINAL_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - ORIGINAL_DUPLICATE_PENALTY_WEIGHT * score.duplicate_penalty
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_original_strict_iou(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Task319 baseline with F1 thresholds changed to 0.5, 0.75 and 0.95."""
    score = _score_components(
        completion,
        solution,
        target_labels,
        iou_thresholds=STRICT_BASELINE_IOU_THRESHOLDS,
    )
    if score.format_reward == 0.0 and not any(
        (score.count_reward, score.f1_reward, score.iou_reward, score.sort_reward)
    ):
        return score
    reward = (
        ORIGINAL_REWARD_WEIGHTS["format"] * score.format_reward
        + ORIGINAL_REWARD_WEIGHTS["count"] * score.count_reward
        + ORIGINAL_REWARD_WEIGHTS["f1"] * score.f1_reward
        + ORIGINAL_REWARD_WEIGHTS["matched_iou"] * score.iou_reward
        + ORIGINAL_REWARD_WEIGHTS["sort"] * score.sort_reward
        - ORIGINAL_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - ORIGINAL_DUPLICATE_PENALTY_WEIGHT * score.duplicate_penalty
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_recall_v2(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Thresholded recall-oriented reward used in the recall ablation."""
    score = _score_components(completion, solution, target_labels)
    reward = (
        RECALL_V2_REWARD_WEIGHTS["format"] * score.format_reward
        + RECALL_V2_REWARD_WEIGHTS["count"] * score.recall_count_reward
        + RECALL_V2_REWARD_WEIGHTS["precision"] * score.precision_reward
        + RECALL_V2_REWARD_WEIGHTS["recall"] * score.recall_reward
        + RECALL_V2_REWARD_WEIGHTS["f2"] * score.f2_reward
        + RECALL_V2_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        + RECALL_V2_REWARD_WEIGHTS["sort"] * score.sort_reward
        - RECALL_V2_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - RECALL_V2_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - RECALL_V2_UNDER_COUNT_PENALTY_WEIGHT * score.under_count_penalty
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_best_v3(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Dense recall-first set reward with an explicit soft false-positive guard."""
    score = _score_components(completion, solution, target_labels)
    reward = (
        BEST_V3_REWARD_WEIGHTS["format"] * score.format_reward
        + BEST_V3_REWARD_WEIGHTS["count"] * score.recall_count_reward
        + BEST_V3_REWARD_WEIGHTS["hard_recall"] * score.recall_reward
        + BEST_V3_REWARD_WEIGHTS["hard_f2"] * score.f2_reward
        + BEST_V3_REWARD_WEIGHTS["soft_f2"] * score.soft_f2_reward
        + BEST_V3_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        + BEST_V3_REWARD_WEIGHTS["sort"] * score.sort_reward
        - BEST_V3_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - BEST_V3_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - BEST_V3_UNDER_COUNT_PENALTY_WEIGHT * score.under_count_penalty
        - BEST_V3_SOFT_FP_PENALTY_WEIGHT * (1.0 - score.pred_coverage_reward)
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_pareto_v4(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Density-adaptive reward with sparse precision and format preservation."""
    score = _score_components(completion, solution, target_labels)
    density = max(0.0, score.adaptive_beta - 1.0)
    reward = (
        PARETO_V4_REWARD_WEIGHTS["format"] * score.format_reward
        + PARETO_V4_REWARD_WEIGHTS["precision"] * score.precision_reward
        + PARETO_V4_REWARD_WEIGHTS["recall"] * score.recall_reward
        + PARETO_V4_REWARD_WEIGHTS["adaptive_hard_fbeta"] * score.adaptive_fbeta_reward
        + PARETO_V4_REWARD_WEIGHTS["adaptive_soft_fbeta"] * score.adaptive_soft_fbeta_reward
        + PARETO_V4_REWARD_WEIGHTS["balanced_count"] * score.balanced_count_reward
        + PARETO_V4_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        + PARETO_V4_REWARD_WEIGHTS["sort"] * score.sort_reward
        - PARETO_V4_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - PARETO_V4_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - PARETO_V4_OVER_COUNT_PENALTY_WEIGHT * score.over_count_penalty
        - PARETO_V4_DENSE_UNDER_COUNT_PENALTY_WEIGHT * density * score.under_count_penalty
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_pareto_v5(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Recall-balanced Pareto reward with stronger dense undercount pressure."""
    score = _score_components(completion, solution, target_labels)
    density = max(0.0, score.adaptive_beta - 1.0)
    reward = (
        PARETO_V5_REWARD_WEIGHTS["format"] * score.format_reward
        + PARETO_V5_REWARD_WEIGHTS["precision"] * score.precision_reward
        + PARETO_V5_REWARD_WEIGHTS["recall"] * score.recall_reward
        + PARETO_V5_REWARD_WEIGHTS["adaptive_hard_fbeta"] * score.adaptive_fbeta_reward
        + PARETO_V5_REWARD_WEIGHTS["adaptive_soft_fbeta"] * score.adaptive_soft_fbeta_reward
        + PARETO_V5_REWARD_WEIGHTS["balanced_count"] * score.balanced_count_reward
        + PARETO_V5_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        + PARETO_V5_REWARD_WEIGHTS["sort"] * score.sort_reward
        - PARETO_V5_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - PARETO_V5_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - PARETO_V5_OVER_COUNT_PENALTY_WEIGHT * score.over_count_penalty
        - (
            PARETO_V5_BASE_UNDER_COUNT_PENALTY_WEIGHT
            + PARETO_V5_DENSE_UNDER_COUNT_PENALTY_WEIGHT * density
        )
        * score.under_count_penalty
    )
    return replace(score, reward=_clamp_reward(reward))


def score_completion_pareto_v6(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Blend best_v3 with a recall-heavy residual only on dense examples."""
    score = _score_components(completion, solution, target_labels)
    base_reward = score_completion_best_v3(completion, solution, target_labels).reward
    density = max(0.0, score.adaptive_beta - 1.0)
    dense_reward = (
        PARETO_V6_DENSE_REWARD_WEIGHTS["format"] * score.format_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["precision"] * score.precision_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["recall"] * score.recall_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["adaptive_hard_fbeta"] * score.adaptive_fbeta_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["adaptive_soft_fbeta"] * score.adaptive_soft_fbeta_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["balanced_count"] * score.balanced_count_reward
        + PARETO_V6_DENSE_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        - PARETO_V6_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - PARETO_V6_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - PARETO_V6_OVER_COUNT_PENALTY_WEIGHT * score.over_count_penalty
        - PARETO_V6_UNDER_COUNT_PENALTY_WEIGHT * score.under_count_penalty
    )
    reward = (1.0 - density) * base_reward + density * _clamp_reward(dense_reward)
    return replace(score, reward=_clamp_reward(reward))


def score_completion_pareto_v7(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Preserve v5 except for a stronger recall objective on extreme density."""
    score = _score_components(completion, solution, target_labels)
    base_reward = score_completion_pareto_v5(completion, solution, target_labels).reward
    gt_total = sum(len(boxes) for boxes in solution_bbox_by_label(solution).values())
    extreme = min(
        1.0,
        max(
            0.0,
            (gt_total - PARETO_V7_EXTREME_START)
            / (PARETO_V7_EXTREME_FULL - PARETO_V7_EXTREME_START),
        ),
    )
    extreme_reward = (
        PARETO_V7_EXTREME_REWARD_WEIGHTS["format"] * score.format_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["precision"] * score.precision_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["recall"] * score.recall_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["adaptive_hard_fbeta"] * score.adaptive_fbeta_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["adaptive_soft_fbeta"] * score.adaptive_soft_fbeta_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["balanced_count"] * score.balanced_count_reward
        + PARETO_V7_EXTREME_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        - PARETO_V7_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - PARETO_V7_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - PARETO_V7_OVER_COUNT_PENALTY_WEIGHT * score.over_count_penalty
        - PARETO_V7_UNDER_COUNT_PENALTY_WEIGHT * score.under_count_penalty
    )
    reward = (1.0 - extreme) * base_reward + extreme * _clamp_reward(extreme_reward)
    return replace(score, reward=_clamp_reward(reward))


def score_completion_pareto_v8(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Refine v7 dense outputs with stronger precision and localization guards."""
    score = _score_components(completion, solution, target_labels)
    base_reward = score_completion_pareto_v5(completion, solution, target_labels).reward
    gt_total = sum(len(boxes) for boxes in solution_bbox_by_label(solution).values())
    extreme = min(
        1.0,
        max(
            0.0,
            (gt_total - PARETO_V7_EXTREME_START)
            / (PARETO_V7_EXTREME_FULL - PARETO_V7_EXTREME_START),
        ),
    )
    extreme_reward = (
        PARETO_V8_EXTREME_REWARD_WEIGHTS["format"] * score.format_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["precision"] * score.precision_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["recall"] * score.recall_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["adaptive_hard_fbeta"] * score.adaptive_fbeta_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["adaptive_soft_fbeta"] * score.adaptive_soft_fbeta_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["balanced_count"] * score.balanced_count_reward
        + PARETO_V8_EXTREME_REWARD_WEIGHTS["gt_coverage"] * score.gt_coverage_reward
        - PARETO_V8_BIGBOX_PENALTY_WEIGHT * score.bigbox_penalty
        - PARETO_V8_DUPLICATE_PENALTY_WEIGHT * score.excess_duplicate_penalty
        - PARETO_V8_OVER_COUNT_PENALTY_WEIGHT * score.over_count_penalty
        - PARETO_V8_UNDER_COUNT_PENALTY_WEIGHT * score.under_count_penalty
    )
    reward = (1.0 - extreme) * base_reward + extreme * _clamp_reward(extreme_reward)
    return replace(score, reward=_clamp_reward(reward))


def _dc_pareto_objectives(score: RewardBreakdown) -> tuple[float, ...]:
    """Verifiable objectives used by density-conditioned Pareto ranking."""
    return (
        score.f2_iou50,
        score.size_aware_gt_coverage,
        score.precision_iou50,
        score.balanced_count_reward,
    )


def _dc_constraint_violation(score: RewardBreakdown) -> float:
    """Density-conditioned feasibility violation; zero means feasible."""
    density = max(0.0, min(1.0, score.adaptive_beta - 1.0))
    precision_floor = 0.45 - 0.20 * density
    count_floor = 0.60 - 0.20 * density
    count_ceiling = 1.20 + 0.30 * density
    return (
        2.0 * max(0.0, 1.0 - score.format_reward)
        + max(0.0, precision_floor - score.precision_iou50)
        + max(0.0, count_floor - score.pred_gt_ratio)
        + 0.5 * max(0.0, score.pred_gt_ratio - count_ceiling)
        + 0.25 * score.bigbox_penalty
        + 0.25 * score.excess_duplicate_penalty
    )


def _dc_scalar_quality(score: RewardBreakdown) -> float:
    """Chebyshev scalarization used for eval and the controlled ablation."""
    worst_objective = min(_dc_pareto_objectives(score))
    return _clamp_reward(worst_objective - _dc_constraint_violation(score))


def _dc_dominates(left: RewardBreakdown, right: RewardBreakdown, eps: float = 1e-9) -> bool:
    left_violation = _dc_constraint_violation(left)
    right_violation = _dc_constraint_violation(right)
    left_feasible = left_violation <= eps
    right_feasible = right_violation <= eps
    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        return left_violation + eps < right_violation
    left_obj = _dc_pareto_objectives(left)
    right_obj = _dc_pareto_objectives(right)
    return all(a + eps >= b for a, b in zip(left_obj, right_obj)) and any(
        a > b + eps for a, b in zip(left_obj, right_obj)
    )


def density_conditioned_pareto_rank_rewards(scores: list[RewardBreakdown]) -> list[float]:
    """Convert one prompt's rollout vectors into ordinal constrained-Pareto rewards."""
    if len(scores) <= 1:
        return [_dc_scalar_quality(score) for score in scores]

    remaining = set(range(len(scores)))
    front_rank = [0] * len(scores)
    rank = 0
    while remaining:
        front = [
            i
            for i in remaining
            if not any(_dc_dominates(scores[j], scores[i]) for j in remaining if j != i)
        ]
        for i in front:
            front_rank[i] = rank
        remaining.difference_update(front)
        rank += 1

    # Front rank is primary. Constraint violation and max-min objective quality
    # only order points within a front; no reward magnitudes or tuned weights
    # enter the GRPO advantage.
    keys = [
        (
            front_rank[i],
            round(_dc_constraint_violation(score), 12),
            -round(min(_dc_pareto_objectives(score)), 12),
            -round(score.f2_iou50, 12),
        )
        for i, score in enumerate(scores)
    ]
    ordered_keys = sorted(set(keys))
    key_rank = {key: idx for idx, key in enumerate(ordered_keys)}
    denominator = max(1, len(ordered_keys) - 1)
    return [1.0 - key_rank[key] / denominator for key in keys]


def score_completion_dcgrpo_scalar_v1(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Static Chebyshev baseline over the same objectives as DC-Pareto GRPO."""
    score = _score_components(completion, solution, target_labels)
    return replace(score, reward=_dc_scalar_quality(score))


@dataclass
class JointPRBreakdown:
    format_reward: float
    soft_precision: float
    soft_recall: float
    nash_reward: float
    hard_precision: float
    hard_recall: float
    hard_f1: float
    pred_gt_ratio: float


def _threshold_centered_iou_credit(iou: float, tau: float = 0.08) -> float:
    """Smooth credit around IoU=.5 with exactly zero credit at IoU=0."""
    if iou <= 0.0:
        return 0.0
    baseline = 1.0 / (1.0 + math.exp(0.5 / tau))
    value = 1.0 / (1.0 + math.exp(-(iou - 0.5) / tau))
    return (value - baseline) / (1.0 - baseline)


def _joint_pr_components(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> JointPRBreakdown:
    gt_by_label = solution_bbox_by_label(solution)
    labels = target_labels or solution.get("target_labels") or list(gt_by_label)
    if str(solution.get("format") or "").startswith("gam_bbox"):
        pred_by_label, meta = parse_gam_bbox(completion)
    else:
        pred_by_label, meta = parse_json_bbox(completion)
        if not pred_by_label and meta.get("format_ok", 0.0) == 0.0:
            pred_by_label, meta = parse_compact_bbox(completion)
    pred_by_label = assign_unlabeled_boxes(pred_by_label, gt_by_label, labels)
    gt_total = sum(len(gt_by_label.get(label, [])) for label in labels)
    pred_total = sum(len(pred_by_label.get(label, [])) for label in labels)
    soft_tp = 0.0
    hard_tp = 0
    for label in labels:
        pairs = _match_pairs(pred_by_label.get(label, []), gt_by_label.get(label, []))
        soft_tp += sum(_threshold_centered_iou_credit(iou) for _, _, iou in pairs)
        hard_tp += sum(iou >= 0.5 for _, _, iou in pairs)
    soft_precision = soft_tp / pred_total if pred_total else 0.0
    soft_recall = soft_tp / gt_total if gt_total else 0.0
    hard_precision = hard_tp / pred_total if pred_total else 0.0
    hard_recall = hard_tp / gt_total if gt_total else 0.0
    hard_f1 = (
        2.0 * hard_precision * hard_recall / (hard_precision + hard_recall)
        if hard_precision + hard_recall
        else 0.0
    )
    format_reward = float(meta.get("format_ok", 0.0))
    if solution.get("format") == "qwen_json_bbox_v1" and format_reward < 1.0:
        format_reward = 0.0
    nash_reward = math.sqrt(max(0.0, soft_precision * soft_recall))
    return JointPRBreakdown(
        format_reward=format_reward,
        soft_precision=soft_precision,
        soft_recall=soft_recall,
        nash_reward=nash_reward,
        hard_precision=hard_precision,
        hard_recall=hard_recall,
        hard_f1=hard_f1,
        pred_gt_ratio=pred_total / gt_total if gt_total else float(pred_total == 0),
    )


def joint_pr_nash_ordinal_rewards(scores: list[JointPRBreakdown]) -> list[float]:
    """Rank rollouts by joint localization precision-recall welfare."""
    if len(scores) <= 1:
        return [score.format_reward * score.nash_reward for score in scores]
    keys = [
        (
            -round(score.format_reward, 12),
            -round(score.nash_reward, 12),
            -round(min(score.soft_precision, score.soft_recall), 12),
            -round(score.hard_f1, 12),
        )
        for score in scores
    ]
    ordered = sorted(set(keys))
    rank = {key: i for i, key in enumerate(ordered)}
    denominator = max(1, len(ordered) - 1)
    return [1.0 - rank[key] / denominator for key in keys]


def _ordinal_rewards_from_values(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return values
    keys = [-round(value, 12) for value in values]
    ordered = sorted(set(keys))
    rank = {key: i for i, key in enumerate(ordered)}
    denominator = max(1, len(ordered) - 1)
    return [1.0 - rank[key] / denominator for key in keys]


def pareto_consensus_ordinal_rewards(
    v5_values: list[float], joint_scores: list[JointPRBreakdown]
) -> list[float]:
    """Reward rollouts only when v5 preservation and Joint-PR agree.

    Both objectives are converted to within-prompt ordinal utilities, so their
    scales cannot dominate one another. The max-min term implements a Pareto
    safeguard; the geometric mean is the Nash tie-break among equally safe
    candidates.
    """
    if len(v5_values) != len(joint_scores):
        raise ValueError("v5_values and joint_scores must have equal length")
    if len(v5_values) <= 1:
        return [
            math.sqrt(max(0.0, value) * max(0.0, score.nash_reward))
            * score.format_reward
            for value, score in zip(v5_values, joint_scores)
        ]
    v5_utility = _ordinal_rewards_from_values(v5_values)
    nash_utility = joint_pr_nash_ordinal_rewards(joint_scores)
    qualities = [
        (
            round(min(v5_rank, nash_rank), 12),
            round(math.sqrt(max(0.0, v5_rank * nash_rank)), 12),
            round(score.hard_f1, 12),
            round(value, 12),
            round(score.nash_reward, 12),
        )
        for value, score, v5_rank, nash_rank in zip(
            v5_values, joint_scores, v5_utility, nash_utility
        )
    ]
    ordered = sorted(set(qualities))
    rank = {quality: i for i, quality in enumerate(ordered)}
    denominator = max(1, len(ordered) - 1)
    return [rank[quality] / denominator for quality in qualities]


def _median_normalized_gt_area(solution: dict[str, Any]) -> float:
    areas = sorted(
        box_area(box) / 1_000_000.0
        for boxes in solution_bbox_by_label(solution).values()
        for box in boxes
    )
    if not areas:
        return 1.0
    return areas[len(areas) // 2]


def score_completion(
    completion: str, solution: dict[str, Any], target_labels: list[str] | None = None
) -> RewardBreakdown:
    """Backward-compatible direct API for the original task319 reward."""
    return score_completion_original(completion, solution, target_labels)


class _GroundingBBoxRewardBase:
    score_fn: Callable[[str, dict[str, Any], list[str] | None], RewardBreakdown] = staticmethod(
        score_completion_original
    )

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        rewards = []
        solution = solution or kwargs.get("solutions")
        if solution is None:
            return [0.0 for _ in completions]
        target_labels = target_labels or kwargs.get("target_label") or [None] * len(completions)
        for completion, sol, labels in zip(completions, solution, target_labels):
            try:
                rewards.append(self.score_fn(str(completion), _as_solution(sol), labels).reward)
            except Exception:
                rewards.append(0.0)
        return rewards


class GroundingBBoxReward(_GroundingBBoxRewardBase):
    """Legacy Python API: original task319 reward."""


class GroundingBBoxRewardOriginalStrictIoU(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_original_strict_iou)


class GroundingBBoxRecallV2Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_recall_v2)


class GroundingBBoxBestV3Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_best_v3)


class GroundingBBoxRexOmniPaperReward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_rexomni_paper)


class GroundingBBoxParetoV4Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_pareto_v4)


class GroundingBBoxParetoV5Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_pareto_v5)


class GroundingBBoxParetoV6Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_pareto_v6)


class GroundingBBoxParetoV7Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_pareto_v7)


class GroundingBBoxParetoV8Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_pareto_v8)


class GroundingBBoxDCGRPOScalarV1Reward(_GroundingBBoxRewardBase):
    score_fn = staticmethod(score_completion_dcgrpo_scalar_v1)


class GroundingBBoxDCParetoRankV1Reward:
    """Groupwise density-conditioned constrained Pareto ordinal reward."""

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        labels_list = target_labels or kwargs.get("target_label") or [None] * len(completions)
        scores: list[RewardBreakdown] = []
        parsed_solutions: list[dict[str, Any]] = []
        for completion, sol, labels in zip(completions, solutions, labels_list):
            try:
                parsed = _as_solution(sol)
                scores.append(_score_components(str(completion), parsed, labels))
                parsed_solutions.append(parsed)
            except Exception:
                scores.append(RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                parsed_solutions.append({})

        groups: dict[str, list[int]] = {}
        for i, (sol, labels) in enumerate(zip(parsed_solutions, labels_list)):
            key = json.dumps(
                {"solution": sol, "target_labels": labels},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            groups.setdefault(key, []).append(i)

        rewards = [0.0] * len(scores)
        for indices in groups.values():
            group_rewards = density_conditioned_pareto_rank_rewards([scores[i] for i in indices])
            for i, reward in zip(indices, group_rewards):
                rewards[i] = reward
        return rewards


class GroundingBBoxJointPRNashV1Reward:
    """Hungarian-Nash group reward that only benefits from joint P/R quality."""

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        labels_list = target_labels or kwargs.get("target_label") or [None] * len(completions)
        scores: list[JointPRBreakdown] = []
        parsed_solutions: list[dict[str, Any]] = []
        for completion, sol, labels in zip(completions, solutions, labels_list):
            try:
                parsed = _as_solution(sol)
                scores.append(_joint_pr_components(str(completion), parsed, labels))
                parsed_solutions.append(parsed)
            except Exception:
                scores.append(JointPRBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                parsed_solutions.append({})
        groups: dict[str, list[int]] = {}
        for i, (sol, labels) in enumerate(zip(parsed_solutions, labels_list)):
            key = json.dumps(
                {"solution": sol, "target_labels": labels},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            groups.setdefault(key, []).append(i)
        rewards = [0.0] * len(scores)
        for indices in groups.values():
            group_rewards = joint_pr_nash_ordinal_rewards([scores[i] for i in indices])
            for i, reward in zip(indices, group_rewards):
                rewards[i] = reward
        return rewards


class GroundingBBoxScaleGatedNashV1Reward:
    """Use Nash localization ranking only for tiny-object scenes; retain v5 otherwise."""

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        labels_list = target_labels or kwargs.get("target_label") or [None] * len(completions)
        parsed_solutions = []
        groups: dict[str, list[int]] = {}
        for i, (sol, labels) in enumerate(zip(solutions, labels_list)):
            try:
                parsed = _as_solution(sol)
            except Exception:
                parsed = {}
            parsed_solutions.append(parsed)
            key = json.dumps(
                {"solution": parsed, "target_labels": labels},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            groups.setdefault(key, []).append(i)

        rewards = [0.0] * len(completions)
        for indices in groups.values():
            solution_for_group = parsed_solutions[indices[0]]
            if _median_normalized_gt_area(solution_for_group) <= SMALL_OBJECT_MEDIAN_AREA_GATE:
                scores = []
                for i in indices:
                    try:
                        scores.append(
                            _joint_pr_components(
                                str(completions[i]), parsed_solutions[i], labels_list[i]
                            )
                        )
                    except Exception:
                        scores.append(JointPRBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                group_rewards = joint_pr_nash_ordinal_rewards(scores)
            else:
                values = []
                for i in indices:
                    try:
                        values.append(
                            score_completion_pareto_v5(
                                str(completions[i]), parsed_solutions[i], labels_list[i]
                            ).reward
                        )
                    except Exception:
                        values.append(0.0)
                group_rewards = _ordinal_rewards_from_values(values)
            for i, reward in zip(indices, group_rewards):
                rewards[i] = reward
        return rewards


class GroundingBBoxParetoConsensusJointPRV1Reward:
    """Groupwise Pareto consensus between v5 preservation and Joint-PR."""

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        labels_list = target_labels or kwargs.get("target_label") or [None] * len(completions)
        parsed_solutions = []
        groups: dict[str, list[int]] = {}
        for i, (sol, labels) in enumerate(zip(solutions, labels_list)):
            try:
                parsed = _as_solution(sol)
            except Exception:
                parsed = {}
            parsed_solutions.append(parsed)
            key = json.dumps(
                {"solution": parsed, "target_labels": labels},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            groups.setdefault(key, []).append(i)

        rewards = [0.0] * len(completions)
        for indices in groups.values():
            v5_values = []
            joint_scores = []
            for i in indices:
                try:
                    v5_values.append(
                        score_completion_pareto_v5(
                            str(completions[i]), parsed_solutions[i], labels_list[i]
                        ).reward
                    )
                    joint_scores.append(
                        _joint_pr_components(
                            str(completions[i]), parsed_solutions[i], labels_list[i]
                        )
                    )
                except Exception:
                    v5_values.append(0.0)
                    joint_scores.append(
                        JointPRBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    )
            group_rewards = pareto_consensus_ordinal_rewards(v5_values, joint_scores)
            for i, reward in zip(indices, group_rewards):
                rewards[i] = reward
        return rewards


class GroundingBBoxTileDumpReward:
    """Best-v3 reward with lossless per-rank dumping for eval-only tile probes."""

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        rewards = GroundingBBoxBestV3Reward()(
            completions, solution=solution, target_labels=target_labels, **kwargs
        )
        dump_dir = os.environ.get("GRPO_TILE_DUMP_DIR")
        solutions = solution or kwargs.get("solutions")
        if not dump_dir or solutions is None:
            return rewards
        os.makedirs(dump_dir, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        path = os.path.join(dump_dir, f"rank_{rank}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            for completion, raw_solution in zip(completions, solutions):
                try:
                    parsed_solution = _as_solution(raw_solution)
                except Exception:
                    continue
                tile_id = parsed_solution.get("tile_id")
                if not tile_id:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "tile_id": tile_id,
                            "completion": str(completion),
                            "solution": parsed_solution,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return rewards


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftGroundingBBoxRewardOriginal(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxReward()(completions, solution=solution, target_labels=target_labels, **kwargs)

    class SwiftGroundingBBoxRewardOriginalStrictIoU(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxRewardOriginalStrictIoU()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardRecallV2(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxRecallV2Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardBestV3(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxBestV3Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardRexOmniPaper(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxRexOmniPaperReward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoV4(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoV4Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoV5(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoV5Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoV6(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoV6Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoV7(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoV7Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoV8(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoV8Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardDCGRPOScalarV1(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxDCGRPOScalarV1Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardDCParetoRankV1(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxDCParetoRankV1Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardJointPRNashV1(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxJointPRNashV1Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardScaleGatedNashV1(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxScaleGatedNashV1Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardParetoConsensusJointPRV1(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxParetoConsensusJointPRV1Reward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    class SwiftGroundingBBoxRewardTileDump(ORM):
        def __call__(self, completions, solution=None, target_labels=None, **kwargs):
            return GroundingBBoxTileDumpReward()(
                completions, solution=solution, target_labels=target_labels, **kwargs
            )

    orms["grounding_bbox_reward"] = SwiftGroundingBBoxRewardOriginal
    orms["grounding_bbox_reward_original"] = SwiftGroundingBBoxRewardOriginal
    orms["grounding_bbox_reward_original_strict_iou"] = (
        SwiftGroundingBBoxRewardOriginalStrictIoU
    )
    orms["grounding_bbox_reward_recall_v2"] = SwiftGroundingBBoxRewardRecallV2
    orms["grounding_bbox_reward_best_v3"] = SwiftGroundingBBoxRewardBestV3
    orms["grounding_bbox_reward_rexomni_paper"] = SwiftGroundingBBoxRewardRexOmniPaper
    orms["grounding_bbox_reward_rexomni_paper_eq4"] = SwiftGroundingBBoxRewardRexOmniPaper
    orms["grounding_bbox_reward_pareto_v4"] = SwiftGroundingBBoxRewardParetoV4
    orms["grounding_bbox_reward_pareto_v5"] = SwiftGroundingBBoxRewardParetoV5
    orms["grounding_bbox_reward_pareto_v6"] = SwiftGroundingBBoxRewardParetoV6
    orms["grounding_bbox_reward_pareto_v7"] = SwiftGroundingBBoxRewardParetoV7
    orms["grounding_bbox_reward_pareto_v8"] = SwiftGroundingBBoxRewardParetoV8
    orms["grounding_bbox_reward_dcgrpo_scalar_v1"] = SwiftGroundingBBoxRewardDCGRPOScalarV1
    orms["grounding_bbox_reward_dc_pareto_rank_v1"] = SwiftGroundingBBoxRewardDCParetoRankV1
    orms["grounding_bbox_reward_joint_pr_nash_v1"] = SwiftGroundingBBoxRewardJointPRNashV1
    orms["grounding_bbox_reward_scale_gated_nash_v1"] = SwiftGroundingBBoxRewardScaleGatedNashV1
    orms["grounding_bbox_reward_pareto_consensus_jointpr_v1"] = (
        SwiftGroundingBBoxRewardParetoConsensusJointPRV1
    )
    orms["grounding_bbox_reward_tile_dump"] = SwiftGroundingBBoxRewardTileDump
except Exception:
    pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--completion", required=True)
    ap.add_argument("--solution-json", required=True)
    ap.add_argument(
        "--variant",
        choices=("original", "recall_v2", "best_v3", "pareto_v4", "pareto_v5", "pareto_v6", "pareto_v7", "pareto_v8"),
        default="original",
    )
    args = ap.parse_args()
    scorers = {
        "original": score_completion_original,
        "recall_v2": score_completion_recall_v2,
        "best_v3": score_completion_best_v3,
        "pareto_v4": score_completion_pareto_v4,
        "pareto_v5": score_completion_pareto_v5,
        "pareto_v6": score_completion_pareto_v6,
        "pareto_v7": score_completion_pareto_v7,
        "pareto_v8": score_completion_pareto_v8,
    }
    print(scorers[args.variant](args.completion, json.loads(args.solution_json)))
