"""Comparable multi-teacher and merged-single-teacher OCR rewards.

Both rewards consume ``multi_teacher_views_v1`` solutions and share the same
parser, reward scale and anti-hacking penalties.  They differ only in which
teacher views define semantic correctness:

* ``ocr_multiteacher_reward_v1`` uses Rex and LocateAnything slow as primary
  teachers, hybrid as a low-weight arbiter, and optional PP-OCR geometry.
* ``ocr_merged_single_teacher_reward_v1`` uses only the frozen merged
  primary-vote view.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from train.rl.rewards.ocr_dual_reference_reward import (
    BBOX_LINE_RE,
    _as_solution,
    box_iou,
    f1_from_mass,
    maximum_assignment,
    score_reference,
)


BBOX_OPEN_RE = re.compile(r"<bbox\b[^>]*>", re.I)
BBOX_CLOSE_RE = re.compile(r"</bbox\s*>", re.I)
BBOX_COMPLETE_RE = re.compile(
    r"<bbox\b[^>]*>(.*?)</bbox\s*>",
    re.I | re.S,
)
STRICT_FULL_RE = re.compile(
    r"^\s*<bbox\b[^>]*>(.*?)</bbox\s*>\s*$",
    re.I | re.S,
)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class StrictParsedCompletion:
    instances: list[dict[str, Any]]
    format_reward: float
    invalid_line_count: int
    invalid_box_count: int
    outside_content_count: int


@dataclass(frozen=True)
class OCRMultiViewRewardBreakdown:
    reward: float
    mode: str
    rex_score: float
    slow_score: float
    hybrid_score: float
    ppocr_geometry_score: float
    merged_score: float
    semantic_score: float
    format_reward: float
    duplicate_penalty: float
    overgeneration_penalty: float
    invalid_penalty: float
    oversized_box_penalty: float
    prediction_count: int
    target_count: int


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in box
        )
        and 0 <= float(box[0]) < float(box[2]) <= 1000
        and 0 <= float(box[1]) < float(box[3]) <= 1000
    )


def _semantic_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in value if char.isalnum())


def _surface_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def parse_completion_strict(text: str) -> StrictParsedCompletion:
    cleaned = str(text).split("<|im_end|>")[0]
    strict_match = STRICT_FULL_RE.fullmatch(cleaned)
    complete_match = BBOX_COMPLETE_RE.search(cleaned)
    open_match = BBOX_OPEN_RE.search(cleaned)
    close_match = BBOX_CLOSE_RE.search(cleaned)

    if complete_match is not None:
        body = complete_match.group(1)
        outside = (
            cleaned[: complete_match.start()] + cleaned[complete_match.end() :]
        ).strip()
    elif open_match is not None:
        body = cleaned[open_match.end() :]
        outside = cleaned[: open_match.start()].strip()
    else:
        body = cleaned
        outside = ""

    instances = []
    invalid_lines = 0
    invalid_boxes = 0
    nonempty_lines = [line for line in body.splitlines() if line.strip()]
    for line in nonempty_lines:
        match = BBOX_LINE_RE.match(line)
        if match is None:
            invalid_lines += 1
            continue
        box = [float(match.group(index)) for index in range(1, 5)]
        label = match.group(5).strip()
        if not label or not _surface_text(label):
            invalid_lines += 1
            continue
        if not _valid_box(box):
            invalid_boxes += 1
            continue
        instances.append({"bbox_norm1000": box, "text": label})

    open_count = len(BBOX_OPEN_RE.findall(cleaned))
    close_count = len(BBOX_CLOSE_RE.findall(cleaned))
    outside_count = int(bool(outside)) + int(open_count != 1) + int(close_count != 1)
    if (
        strict_match is not None
        and invalid_lines == 0
        and invalid_boxes == 0
        and open_count == 1
        and close_count == 1
    ):
        format_reward = 1.0
    elif open_match is not None and close_match is not None and instances:
        format_reward = 0.45
    elif open_match is not None and instances:
        format_reward = 0.35
    elif instances:
        format_reward = 0.15
    else:
        format_reward = 0.0
    return StrictParsedCompletion(
        instances=instances,
        format_reward=format_reward,
        invalid_line_count=invalid_lines,
        invalid_box_count=invalid_boxes,
        outside_content_count=outside_count,
    )


def _references(solution: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for reference in solution.get("references") or []:
        if not isinstance(reference, dict):
            continue
        if reference.get("available") is False:
            continue
        instances = []
        for item in reference.get("instances") or []:
            if not isinstance(item, dict):
                continue
            box = item.get("bbox_norm1000")
            text = str(item.get("text") or "")
            if _valid_box(box) and _surface_text(text):
                instances.append(
                    {
                        "bbox_norm1000": [float(value) for value in box],
                        "text": text,
                    }
                )
        output[str(reference.get("name") or "")] = instances
    return output


def _symbol_aware_text(text: Any) -> str:
    """Give pure-symbol OCR labels a stable key for the legacy scorer."""
    semantic = _semantic_text(text)
    if semantic:
        return str(text)
    surface = _surface_text(text)
    codepoints = "_".join(f"{ord(char):x}" for char in surface)
    return f"ocr_symbol_codepoints_{codepoints}"


def _symbol_aware_instances(
    instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "bbox_norm1000": instance["bbox_norm1000"],
            "text": _symbol_aware_text(instance["text"]),
        }
        for instance in instances
    ]


def _surface_duplicate_penalty(
    predictions: list[dict[str, Any]],
) -> float:
    if len(predictions) < 2:
        return 0.0
    duplicate_pairs = 0
    for left_index, left in enumerate(predictions):
        for right in predictions[left_index + 1 :]:
            if (
                _surface_text(left["text"]) == _surface_text(right["text"])
                and box_iou(
                    left["bbox_norm1000"],
                    right["bbox_norm1000"],
                )
                >= 0.80
            ):
                duplicate_pairs += 1
    return min(1.0, 2.0 * duplicate_pairs / len(predictions))


def surface_f1(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> float:
    if not predictions or not targets:
        return 0.0
    matrix = []
    for prediction in predictions:
        row = []
        for target in targets:
            exact = _surface_text(prediction["text"]) == _surface_text(target["text"])
            overlap = box_iou(
                prediction["bbox_norm1000"],
                target["bbox_norm1000"],
            )
            row.append(1.0 if exact and overlap >= 0.50 else 0.0)
        matrix.append(row)
    matches = len(maximum_assignment(matrix))
    return f1_from_mass(matches, len(predictions), len(targets))


def score_text_view(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> float:
    if not targets:
        return 0.0
    shaped = score_reference(
        _symbol_aware_instances(predictions),
        _symbol_aware_instances(targets),
    )["score"]
    surface = surface_f1(predictions, targets)
    return max(0.0, min(1.0, 0.85 * shaped + 0.15 * surface))


def score_geometry_view(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> float:
    pred_count, target_count = len(predictions), len(targets)
    if not predictions or not targets:
        return 0.0
    hard_values = []
    for threshold in IOU_THRESHOLDS:
        matrix = [
            [
                1.0
                if box_iou(
                    prediction["bbox_norm1000"],
                    target["bbox_norm1000"],
                )
                >= threshold
                else 0.0
                for target in targets
            ]
            for prediction in predictions
        ]
        hard_values.append(
            f1_from_mass(
                len(maximum_assignment(matrix)),
                pred_count,
                target_count,
            )
        )
    soft_matrix = [
        [
            math.sqrt(
                box_iou(
                    prediction["bbox_norm1000"],
                    target["bbox_norm1000"],
                )
            )
            for target in targets
        ]
        for prediction in predictions
    ]
    soft_mass = sum(value for _, _, value in maximum_assignment(soft_matrix))
    soft_f1 = f1_from_mass(soft_mass, pred_count, target_count)
    count_balance = min(pred_count, target_count) / max(pred_count, target_count)
    return max(
        0.0,
        min(
            1.0,
            0.60 * statistics.mean(hard_values)
            + 0.30 * soft_f1
            + 0.10 * count_balance,
        ),
    )


def unsupported_oversized_penalty(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> float:
    if not predictions:
        return 0.0
    unsupported = 0
    for prediction in predictions:
        box = prediction["bbox_norm1000"]
        area = (box[2] - box[0]) * (box[3] - box[1]) / 1_000_000
        if area < 0.80:
            continue
        supported = False
        for target in targets:
            target_box = target["bbox_norm1000"]
            target_area = (
                (target_box[2] - target_box[0])
                * (target_box[3] - target_box[1])
                / 1_000_000
            )
            if (
                target_area >= 0.50 * area
                and box_iou(box, target_box) >= 0.50
            ):
                supported = True
                break
        unsupported += not supported
    return min(1.0, unsupported / len(predictions))


def _empty_breakdown(
    mode: str,
    parsed: StrictParsedCompletion,
) -> OCRMultiViewRewardBreakdown:
    return OCRMultiViewRewardBreakdown(
        reward=0.0,
        mode=mode,
        rex_score=0.0,
        slow_score=0.0,
        hybrid_score=0.0,
        ppocr_geometry_score=0.0,
        merged_score=0.0,
        semantic_score=0.0,
        format_reward=parsed.format_reward,
        duplicate_penalty=0.0,
        overgeneration_penalty=0.0,
        invalid_penalty=1.0,
        oversized_box_penalty=0.0,
        prediction_count=len(parsed.instances),
        target_count=0,
    )


def _finalize(
    mode: str,
    parsed: StrictParsedCompletion,
    semantic: float,
    target_count: int,
    active_targets: list[dict[str, Any]],
    rex_score: float = 0.0,
    slow_score: float = 0.0,
    hybrid_score: float = 0.0,
    ppocr_geometry_score: float = 0.0,
    merged_score: float = 0.0,
) -> OCRMultiViewRewardBreakdown:
    target_count = max(target_count, 1)
    prediction_count = len(parsed.instances)
    coverage = min(prediction_count, target_count) / target_count
    format_gate = parsed.format_reward * parsed.format_reward
    base = (
        0.90 * semantic + 0.10 * parsed.format_reward * coverage
    ) * format_gate
    dup = _surface_duplicate_penalty(parsed.instances)
    overgeneration = max(0, prediction_count - target_count) / max(
        prediction_count,
        1,
    )
    invalid = min(
        1.0,
        0.25 * parsed.invalid_line_count
        + 0.50 * parsed.invalid_box_count
        + 0.50 * parsed.outside_content_count,
    )
    oversized = unsupported_oversized_penalty(
        parsed.instances,
        active_targets,
    )
    reward = max(
        0.0,
        min(
            1.0,
            base
            - 0.05 * dup
            - 0.10 * overgeneration
            - 0.05 * invalid
            - 0.08 * oversized,
        ),
    )
    return OCRMultiViewRewardBreakdown(
        reward=reward,
        mode=mode,
        rex_score=rex_score,
        slow_score=slow_score,
        hybrid_score=hybrid_score,
        ppocr_geometry_score=ppocr_geometry_score,
        merged_score=merged_score,
        semantic_score=semantic,
        format_reward=parsed.format_reward,
        duplicate_penalty=dup,
        overgeneration_penalty=overgeneration,
        invalid_penalty=invalid,
        oversized_box_penalty=oversized,
        prediction_count=prediction_count,
        target_count=target_count,
    )


def score_multiteacher(
    completion: str,
    solution: dict[str, Any] | str,
) -> OCRMultiViewRewardBreakdown:
    mode = "multi_teacher"
    parsed = parse_completion_strict(completion)
    references = _references(_as_solution(solution))
    rex = references.get("rex_omni") or []
    slow = references.get("locateanything_slow") or []
    hybrid = references.get("locateanything_hybrid") or []
    ppocr = references.get("ppocrv6") or []
    if not rex or not slow or not hybrid:
        return _empty_breakdown(mode, parsed)

    rex_score = score_text_view(parsed.instances, rex)
    slow_score = score_text_view(parsed.instances, slow)
    hybrid_score = score_text_view(parsed.instances, hybrid)
    high, low = max(rex_score, slow_score), min(rex_score, slow_score)
    primary_score = 0.70 * high + 0.30 * low
    teacher_score = 0.82 * primary_score + 0.18 * hybrid_score
    ppocr_geometry = (
        score_geometry_view(parsed.instances, ppocr)
        if ppocr
        else 0.0
    )
    semantic = (
        0.95 * teacher_score + 0.05 * ppocr_geometry
        if ppocr
        else teacher_score
    )
    target_count = round(statistics.median([len(rex), len(slow), len(hybrid)]))
    active_targets = rex + slow + hybrid + ppocr
    return _finalize(
        mode,
        parsed,
        semantic,
        target_count,
        active_targets,
        rex_score=rex_score,
        slow_score=slow_score,
        hybrid_score=hybrid_score,
        ppocr_geometry_score=ppocr_geometry,
    )


def score_merged_single_teacher(
    completion: str,
    solution: dict[str, Any] | str,
) -> OCRMultiViewRewardBreakdown:
    mode = "merged_single_teacher"
    parsed = parse_completion_strict(completion)
    references = _references(_as_solution(solution))
    merged = references.get("merged_primary_vote") or []
    if not merged:
        return _empty_breakdown(mode, parsed)
    merged_score = score_text_view(parsed.instances, merged)
    return _finalize(
        mode,
        parsed,
        merged_score,
        len(merged),
        merged,
        merged_score=merged_score,
    )


class OCRMultiTeacherRewardV1:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        output = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                output.append(
                    score_multiteacher(str(completion), raw_solution).reward
                )
            except Exception:
                output.append(0.0)
        return output


class OCRMergedSingleTeacherRewardV1:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        output = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                output.append(
                    score_merged_single_teacher(
                        str(completion),
                        raw_solution,
                    ).reward
                )
            except Exception:
                output.append(0.0)
        return output


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftOCRMultiTeacherRewardV1(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRMultiTeacherRewardV1()(
                completions,
                solution=solution,
                **kwargs,
            )

    class SwiftOCRMergedSingleTeacherRewardV1(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRMergedSingleTeacherRewardV1()(
                completions,
                solution=solution,
                **kwargs,
            )

    orms["ocr_multiteacher_reward_v1"] = SwiftOCRMultiTeacherRewardV1
    orms[
        "ocr_merged_single_teacher_reward_v1"
    ] = SwiftOCRMergedSingleTeacherRewardV1
except Exception:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("multi", "merged"),
        required=True,
    )
    parser.add_argument("--completion", required=True)
    parser.add_argument("--solution-json", required=True)
    args = parser.parse_args()
    scorer = (
        score_multiteacher
        if args.mode == "multi"
        else score_merged_single_teacher
    )
    print(
        json.dumps(
            asdict(scorer(args.completion, args.solution_json)),
            ensure_ascii=False,
            indent=2,
        )
    )
