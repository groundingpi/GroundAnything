"""OCR grounding reward with independent Rex and PP-OCR references.

The reward deliberately avoids choosing a merge policy.  A completion is
scored against both references, with additional credit for instances on which
the two teachers agree.  The Swift ORM name is
``ocr_dual_reference_reward_v1``.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
except ImportError:  # pragma: no cover - portable fallback
    RapidLevenshtein = None


BBOX_BLOCK_RE = re.compile(r"<bbox\b[^>]*>(.*?)(?:</bbox\s*>|$)", re.I | re.S)
BBOX_LINE_RE = re.compile(
    r"^\s*\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]\s*\|\s*(.*?)\s*$"
)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class ParsedCompletion:
    instances: list[dict[str, Any]]
    format_reward: float
    invalid_line_count: int
    invalid_box_count: int


@dataclass(frozen=True)
class OCRDualReferenceBreakdown:
    reward: float
    ppocr_score: float
    rex_score: float
    dual_reference_score: float
    consensus_score: float
    format_reward: float
    duplicate_penalty: float
    invalid_penalty: float
    prediction_count: int


def _as_solution(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("solution must be a dict or JSON object string")


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in value if char.isalnum())


def normalize_loose_letters(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in value if char.isalpha())


def edit_similarity(left: Any, right: Any) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if RapidLevenshtein is not None:
        return float(RapidLevenshtein.normalized_similarity(a, b))
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for row_index, char_a in enumerate(a, 1):
        current = [row_index]
        for col_index, char_b in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col_index] + 1,
                    previous[col_index - 1] + (char_a != char_b),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
        and 0 <= float(box[0]) < float(box[2]) <= 1000
        and 0 <= float(box[1]) < float(box[3]) <= 1000
    )


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (float(left[3]) - float(left[1]))
    right_area = (float(right[2]) - float(right[0])) * (float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def parse_completion(text: str) -> ParsedCompletion:
    cleaned = str(text).split("<|im_end|>")[0]
    marker = re.search(r"<bbox\b[^>]*>", cleaned, flags=re.I)
    block = BBOX_BLOCK_RE.search(cleaned)
    if block:
        body = block.group(1)
    elif marker:
        body = cleaned[marker.end() :]
    else:
        body = cleaned

    instances = []
    invalid_lines = 0
    invalid_boxes = 0
    nonempty_lines = [line for line in body.splitlines() if line.strip()]
    for line in nonempty_lines:
        match = BBOX_LINE_RE.match(line)
        if not match:
            invalid_lines += 1
            continue
        box = [float(match.group(index)) for index in range(1, 5)]
        label = match.group(5).strip()
        if not label or not normalize_text(label):
            invalid_lines += 1
            continue
        if not valid_box(box):
            invalid_boxes += 1
            continue
        instances.append({"bbox_norm1000": box, "text": label})

    has_open = marker is not None
    has_close = re.search(r"</bbox\s*>", cleaned, flags=re.I) is not None
    if has_open and has_close and invalid_lines == 0 and invalid_boxes == 0:
        format_reward = 1.0
    elif has_open and instances:
        format_reward = 0.6
    elif instances:
        format_reward = 0.3
    elif has_open and has_close and not nonempty_lines:
        format_reward = 1.0
    else:
        format_reward = 0.0
    return ParsedCompletion(instances, format_reward, invalid_lines, invalid_boxes)


def maximum_assignment(matrix: list[list[float]]) -> list[tuple[int, int, float]]:
    if not matrix or not matrix[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        values = np.asarray(matrix, dtype=np.float64)
        rows, cols = linear_sum_assignment(-values)
        return [
            (int(row), int(col), float(values[row, col]))
            for row, col in zip(rows, cols)
            if values[row, col] > 0
        ]
    except Exception:
        candidates = sorted(
            (
                (float(value), row, col)
                for row, values in enumerate(matrix)
                for col, value in enumerate(values)
                if value > 0
            ),
            reverse=True,
        )
        used_rows, used_cols = set(), set()
        output = []
        for value, row, col in candidates:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            output.append((row, col, value))
        return output


def f1_from_mass(mass: float, pred_count: int, target_count: int) -> float:
    return 2.0 * mass / (pred_count + target_count) if pred_count + target_count else 1.0


def _reference_instances(reference: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in reference.get("instances") or []:
        box = item.get("bbox_norm1000")
        text = str(item.get("text") or "")
        if valid_box(box) and normalize_text(text):
            output.append({"bbox_norm1000": [float(x) for x in box], "text": text})
    return output


def score_reference(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, float]:
    pred_count, target_count = len(predictions), len(targets)
    if not predictions and not targets:
        return {
            "score": 1.0,
            "hard_f1_miou": 1.0,
            "hard_f1_iou50": 1.0,
            "soft_f1": 1.0,
            "count_balance": 1.0,
        }
    if not predictions or not targets:
        return {
            "score": 0.0,
            "hard_f1_miou": 0.0,
            "hard_f1_iou50": 0.0,
            "soft_f1": 0.0,
            "count_balance": 0.0,
        }

    hard_f1_values = []
    for threshold in IOU_THRESHOLDS:
        matrix = []
        for prediction in predictions:
            row = []
            for target in targets:
                exact = normalize_text(prediction["text"]) == normalize_text(target["text"])
                overlap = box_iou(
                    prediction["bbox_norm1000"],
                    target["bbox_norm1000"],
                )
                row.append((1.0 + overlap * 1e-3) if exact and overlap >= threshold else 0.0)
            matrix.append(row)
        matches = len(maximum_assignment(matrix))
        hard_f1_values.append(f1_from_mass(matches, pred_count, target_count))

    soft_matrix = []
    for prediction in predictions:
        row = []
        for target in targets:
            overlap = box_iou(prediction["bbox_norm1000"], target["bbox_norm1000"])
            similarity = edit_similarity(prediction["text"], target["text"])
            affinity = math.sqrt(overlap) * similarity if overlap >= 0.10 and similarity >= 0.20 else 0.0
            row.append(affinity)
        soft_matrix.append(row)
    soft_mass = sum(value for _, _, value in maximum_assignment(soft_matrix))
    soft_f1 = f1_from_mass(soft_mass, pred_count, target_count)
    count_balance = min(pred_count, target_count) / max(pred_count, target_count)
    hard_f1_miou = sum(hard_f1_values) / len(hard_f1_values)
    score = (
        0.45 * hard_f1_miou
        + 0.35 * soft_f1
        + 0.10 * hard_f1_values[0]
        + 0.10 * count_balance
    )
    return {
        "score": max(0.0, min(1.0, score)),
        "hard_f1_miou": hard_f1_miou,
        "hard_f1_iou50": hard_f1_values[0],
        "soft_f1": soft_f1,
        "count_balance": count_balance,
    }


def score_consensus(
    predictions: list[dict[str, Any]],
    consensus: list[dict[str, Any]],
) -> float:
    targets = []
    for item in consensus:
        texts = [str(text) for text in item.get("text_alternatives") or [item.get("text")]]
        boxes = [
            [float(value) for value in box]
            for box in item.get("bbox_alternatives_norm1000") or []
            if valid_box(box)
        ]
        if boxes and any(normalize_text(text) for text in texts):
            targets.append({"texts": texts, "boxes": boxes})
    pred_count, target_count = len(predictions), len(targets)
    if not targets:
        return 0.0
    if not predictions:
        return 0.0

    hard_values = []
    for threshold in IOU_THRESHOLDS:
        matrix = []
        for prediction in predictions:
            row = []
            for target in targets:
                exact = any(
                    normalize_text(prediction["text"]) == normalize_text(text)
                    for text in target["texts"]
                )
                overlap = max(
                    box_iou(prediction["bbox_norm1000"], box)
                    for box in target["boxes"]
                )
                row.append((1.0 + overlap * 1e-3) if exact and overlap >= threshold else 0.0)
            matrix.append(row)
        hard_values.append(
            f1_from_mass(len(maximum_assignment(matrix)), pred_count, target_count)
        )

    soft_matrix = []
    for prediction in predictions:
        row = []
        for target in targets:
            overlap = max(
                box_iou(prediction["bbox_norm1000"], box) for box in target["boxes"]
            )
            similarity = max(
                edit_similarity(prediction["text"], text) for text in target["texts"]
            )
            row.append(
                math.sqrt(overlap) * similarity
                if overlap >= 0.10 and similarity >= 0.20
                else 0.0
            )
        soft_matrix.append(row)
    soft_mass = sum(value for _, _, value in maximum_assignment(soft_matrix))
    soft_f1 = f1_from_mass(soft_mass, pred_count, target_count)
    return max(
        0.0,
        min(1.0, 0.60 * (sum(hard_values) / len(hard_values)) + 0.40 * soft_f1),
    )


def duplicate_penalty(predictions: list[dict[str, Any]]) -> float:
    if len(predictions) < 2:
        return 0.0
    duplicate_pairs = 0
    for left_index, left in enumerate(predictions):
        for right in predictions[left_index + 1 :]:
            if (
                normalize_text(left["text"]) == normalize_text(right["text"])
                and box_iou(left["bbox_norm1000"], right["bbox_norm1000"]) >= 0.80
            ):
                duplicate_pairs += 1
    return min(1.0, 2.0 * duplicate_pairs / len(predictions))


def score_completion(
    completion: str,
    solution: dict[str, Any] | str,
) -> OCRDualReferenceBreakdown:
    parsed = parse_completion(completion)
    parsed_solution = _as_solution(solution)
    references = {
        str(reference.get("name")): _reference_instances(reference)
        for reference in parsed_solution.get("references") or []
        if isinstance(reference, dict)
    }
    if "ppocr" not in references or "rex" not in references:
        return OCRDualReferenceBreakdown(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            parsed.format_reward,
            0.0,
            1.0,
            len(parsed.instances),
        )

    ppocr = score_reference(parsed.instances, references["ppocr"])
    rex = score_reference(parsed.instances, references["rex"])
    high, low = max(ppocr["score"], rex["score"]), min(ppocr["score"], rex["score"])
    dual_score = 0.75 * high + 0.25 * low
    consensus_items = [
        item
        for item in parsed_solution.get("consensus") or []
        if isinstance(item, dict)
    ]
    consensus = score_consensus(parsed.instances, consensus_items)
    if consensus_items:
        base_reward = 0.65 * dual_score + 0.25 * consensus + 0.10 * parsed.format_reward
    else:
        base_reward = 0.90 * dual_score + 0.10 * parsed.format_reward

    dup_penalty = duplicate_penalty(parsed.instances)
    invalid_penalty = min(
        1.0,
        0.25 * parsed.invalid_line_count + 0.50 * parsed.invalid_box_count,
    )
    reward = max(0.0, min(1.0, base_reward - 0.05 * dup_penalty - 0.05 * invalid_penalty))
    return OCRDualReferenceBreakdown(
        reward=reward,
        ppocr_score=ppocr["score"],
        rex_score=rex["score"],
        dual_reference_score=dual_score,
        consensus_score=consensus,
        format_reward=parsed.format_reward,
        duplicate_penalty=dup_penalty,
        invalid_penalty=invalid_penalty,
        prediction_count=len(parsed.instances),
    )


class OCRDualReferenceRewardV1:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        rewards = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                rewards.append(score_completion(str(completion), raw_solution).reward)
            except Exception:
                rewards.append(0.0)
        return rewards


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftOCRDualReferenceRewardV1(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRDualReferenceRewardV1()(
                completions,
                solution=solution,
                **kwargs,
            )

    orms["ocr_dual_reference_reward_v1"] = SwiftOCRDualReferenceRewardV1
except Exception:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", required=True)
    parser.add_argument("--solution-json", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(score_completion(args.completion, args.solution_json)),
            ensure_ascii=False,
            indent=2,
        )
    )
