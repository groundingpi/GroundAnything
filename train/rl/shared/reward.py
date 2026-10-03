"""Isolated eleven-route reward adapter for GAM RL V3.

BBox and OCR routes retain the released reward implementations byte-for-byte.
Only the five point-output routes need a new scorer because the released joint
recipe has no point reward.  Task metadata selects a scorer only; it never
changes rollout sampling, decoding, cardinality, or stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from train.rl.reward_adapter import JointGAMRewardAdapter, RewardResult
from train.rl.shared.multiroute_data import BBOX_ROUTES, OCR_ROUTES, POINT_ROUTES


GROUP_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.S,
)
COORD_RE = re.compile(r"<\s*(\d{1,4})\s*>")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
POINT_SIGMA = 75.0
POINT_MAX_DISTANCE = 200.0


@dataclass(frozen=True)
class ParsedPoints:
    values: dict[str, list[list[int]]]
    format_score: float
    format_valid: bool
    errors: tuple[str, ...]


def _label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def parse_gam_points(text: str) -> ParsedPoints:
    cleaned = THINK_RE.sub("", str(text or ""))
    cleaned = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", "", cleaned)
    matches = list(GROUP_RE.finditer(cleaned))
    if not matches:
        return ParsedPoints({}, 0.0, False, ("missing_group",))
    values: dict[str, list[list[int]]] = {}
    errors: list[str] = []
    for match in matches:
        label = _label(match.group(1))
        coordinates = [int(item.group(1)) for item in COORD_RE.finditer(match.group(2))]
        residual = COORD_RE.sub("", match.group(2))
        residual = re.sub(r"[\s,]", "", residual)
        if not label:
            errors.append("empty_label")
            continue
        if residual or not coordinates or len(coordinates) % 2:
            errors.append("invalid_group")
            continue
        points = [coordinates[index : index + 2] for index in range(0, len(coordinates), 2)]
        valid = [point for point in points if all(0 <= value <= 999 for value in point)]
        if len(valid) != len(points):
            errors.append("coordinate_range")
        if label in values:
            errors.append("duplicate_label")
        values.setdefault(label, []).extend(valid)
    outside = GROUP_RE.sub("", cleaned)
    if re.sub(r"[\s,]", "", outside):
        errors.append("residual_text")
    useful = any(values.values())
    if useful and not errors:
        score = 1.0
    elif useful:
        score = 0.5
    else:
        score = 0.0
    return ParsedPoints(values, score, bool(useful and not errors), tuple(errors))


def _solution_points(raw: dict[str, Any] | str) -> dict[str, list[list[int]]]:
    solution = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(solution, dict) or solution.get("format") != "gam_point_v1":
        raise ValueError("point reward requires gam_point_v1 solution")
    output: dict[str, list[list[int]]] = {}
    for item in solution.get("items") or []:
        label = _label(item.get("label", ""))
        point = [int(value) for value in item.get("point_2d", [])]
        if not label or len(point) != 2 or any(value < 0 or value > 999 for value in point):
            raise ValueError("invalid point solution item")
        output.setdefault(label, []).append(point)
    if not any(output.values()):
        raise ValueError("empty point solution")
    return output


def _similarity(left: list[int], right: list[int]) -> float:
    distance = math.hypot(float(left[0] - right[0]), float(left[1] - right[1]))
    if distance > POINT_MAX_DISTANCE:
        return 0.0
    return math.exp(-0.5 * (distance / POINT_SIGMA) ** 2)


def _assignment(predicted: list[list[int]], target: list[list[int]]) -> list[float]:
    if not predicted or not target:
        return []
    matrix = [[_similarity(left, right) for right in target] for left in predicted]
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, columns = linear_sum_assignment([[-value for value in row] for row in matrix])
        return [matrix[int(row)][int(column)] for row, column in zip(rows, columns)]
    except Exception:
        triples = sorted(
            (
                (matrix[i][j], i, j)
                for i in range(len(predicted))
                for j in range(len(target))
            ),
            reverse=True,
        )
        used_pred: set[int] = set()
        used_target: set[int] = set()
        output: list[float] = []
        for score, pred_index, target_index in triples:
            if pred_index in used_pred or target_index in used_target:
                continue
            used_pred.add(pred_index)
            used_target.add(target_index)
            output.append(score)
        return output


def point_reward(response: str, solution: dict[str, Any] | str) -> RewardResult:
    parsed = parse_gam_points(response)
    target = _solution_points(solution)
    labels = sorted(set(parsed.values) | set(target))
    soft_true_positive = 0.0
    predicted_count = 0
    target_count = 0
    for label in labels:
        predicted = parsed.values.get(label, [])
        expected = target.get(label, [])
        predicted_count += len(predicted)
        target_count += len(expected)
        soft_true_positive += sum(_assignment(predicted, expected))
    precision = soft_true_positive / predicted_count if predicted_count else 0.0
    recall = soft_true_positive / target_count if target_count else 0.0
    spatial_f2 = (
        5.0 * precision * recall / (4.0 * precision + recall)
        if precision + recall
        else 0.0
    )
    count = (
        min(predicted_count, target_count) / max(predicted_count, target_count)
        if predicted_count and target_count
        else 0.0
    )
    predicted_labels = {label for label, points in parsed.values.items() if points}
    target_labels = {label for label, points in target.items() if points}
    label_precision = len(predicted_labels & target_labels) / len(predicted_labels) if predicted_labels else 0.0
    label_recall = len(predicted_labels & target_labels) / len(target_labels) if target_labels else 0.0
    label_f1 = (
        2.0 * label_precision * label_recall / (label_precision + label_recall)
        if label_precision + label_recall
        else 0.0
    )
    return RewardResult(
        components={
            "point_spatial_f2": float(spatial_f2),
            "point_count": float(count),
            "point_label_f1": float(label_f1),
            "point_format": float(parsed.format_score),
        },
        weights={
            "point_spatial_f2": 0.55,
            "point_count": 0.25,
            "point_label_f1": 0.10,
            "point_format": 0.10,
        },
        format_valid=parsed.format_valid,
    )


class MultiRouteGAMRewardAdapter:
    """Route all eleven semantic datasets without changing the decoder."""

    def __init__(self) -> None:
        self.released = JointGAMRewardAdapter()

    def score(self, response: str, row: dict[str, Any]) -> RewardResult:
        route = str(row.get("rlv3_route", ""))
        task = str(row.get("task_type", ""))
        if route in BBOX_ROUTES:
            if task != "grounding_bbox_gam":
                raise ValueError(f"bbox route/task mismatch: {route}/{task}")
            return self.released.score(response, row)
        if route in OCR_ROUTES:
            if task != "ocr_bbox_text_gam":
                raise ValueError(f"OCR route/task mismatch: {route}/{task}")
            return self.released.score(response, row)
        if route in POINT_ROUTES:
            if task != "point_gam":
                raise ValueError(f"point route/task mismatch: {route}/{task}")
            return point_reward(response, row["solution"])
        raise ValueError(f"unsupported RLV3 semantic route: {route!r}")

