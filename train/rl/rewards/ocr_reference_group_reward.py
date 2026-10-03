"""Granularity-aware multi-reference OCR grounding reward.

This v2 reward keeps the two complete teacher-view scores from v1 and adds a
local group score.  One-to-many groups are evaluated against either complete
segmentation view, avoiding a forced compromise between word and line boxes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from train.rl.rewards.ocr_dual_reference_reward import (
    _as_solution,
    _reference_instances,
    box_iou,
    duplicate_penalty,
    parse_completion,
    score_consensus,
    score_reference,
)


@dataclass(frozen=True)
class OCRReferenceGroupBreakdown:
    reward: float
    ppocr_score: float
    rex_score: float
    dual_reference_score: float
    group_score: float
    format_reward: float
    duplicate_penalty: float
    overgeneration_penalty: float
    invalid_penalty: float
    prediction_count: int
    group_count: int


def _clean_instances(items: Any) -> list[dict[str, Any]]:
    return _reference_instances({"instances": items if isinstance(items, list) else []})


def _group_region(group: dict[str, Any]) -> list[float] | None:
    instances = _clean_instances(group.get("ppocr")) + _clean_instances(group.get("rex"))
    if not instances:
        return None
    boxes = [item["bbox_norm1000"] for item in instances]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _predictions_for_group(
    predictions: list[dict[str, Any]],
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    region = _group_region(group)
    if region is None:
        return []
    output = []
    for prediction in predictions:
        box = prediction["bbox_norm1000"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = region[0] - 15 <= cx <= region[2] + 15 and region[1] - 15 <= cy <= region[3] + 15
        if inside or box_iou(box, region) >= 0.10:
            output.append(prediction)
    return output


def score_reference_groups(
    predictions: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> float:
    if not groups:
        return 0.0
    weighted_score = 0.0
    total_weight = 0.0
    singleton_consensus = []
    singleton_weights = []
    for group in groups:
        group_type = str(group.get("type") or "")
        if group_type == "granularity":
            local_predictions = _predictions_for_group(predictions, group)
            pp = _clean_instances(group.get("ppocr"))
            rex = _clean_instances(group.get("rex"))
            pp_score = score_reference(local_predictions, pp)["score"] if pp else 0.0
            rex_score = score_reference(local_predictions, rex)["score"] if rex else 0.0
            weight = max(len(pp), len(rex), 1)
            weighted_score += weight * max(pp_score, rex_score)
            total_weight += weight
            continue

        pp = _clean_instances(group.get("ppocr"))
        rex = _clean_instances(group.get("rex"))
        texts = list(
            dict.fromkeys(
                str(item["text"]) for item in pp + rex if str(item.get("text") or "")
            )
        )
        boxes = [item["bbox_norm1000"] for item in pp + rex]
        if not texts or not boxes:
            continue
        singleton_consensus.append(
            {
                "text_alternatives": texts,
                "bbox_alternatives_norm1000": boxes,
            }
        )
        singleton_weights.append(0.5 if group_type == "localized_text_conflict" else 1.0)

    if singleton_consensus:
        # score_consensus already enforces global one-to-one assignment and
        # penalizes duplicate/excess predictions through F1 denominators.
        consensus_score = score_consensus(predictions, singleton_consensus)
        weight = sum(singleton_weights)
        weighted_score += weight * consensus_score
        total_weight += weight
    return weighted_score / total_weight if total_weight else 0.0


def score_completion_v2(
    completion: str,
    solution: dict[str, Any] | str,
) -> OCRReferenceGroupBreakdown:
    parsed = parse_completion(completion)
    parsed_solution = _as_solution(solution)
    references = {
        str(reference.get("name")): _reference_instances(reference)
        for reference in parsed_solution.get("references") or []
        if isinstance(reference, dict)
    }
    if "ppocr" not in references or "rex" not in references:
        return OCRReferenceGroupBreakdown(
            reward=0.0,
            ppocr_score=0.0,
            rex_score=0.0,
            dual_reference_score=0.0,
            group_score=0.0,
            format_reward=parsed.format_reward,
            duplicate_penalty=0.0,
            overgeneration_penalty=0.0,
            invalid_penalty=1.0,
            prediction_count=len(parsed.instances),
            group_count=0,
        )
    ppocr = score_reference(parsed.instances, references["ppocr"])["score"]
    rex = score_reference(parsed.instances, references["rex"])["score"]
    high, low = max(ppocr, rex), min(ppocr, rex)
    groups = [
        group for group in parsed_solution.get("reference_groups") or []
        if isinstance(group, dict)
    ]
    group_score = score_reference_groups(parsed.instances, groups)
    has_granularity = any(group.get("type") == "granularity" for group in groups)
    if has_granularity:
        dual = 0.90 * high + 0.10 * low
        semantic = 0.35 * dual + 0.55 * group_score
    else:
        dual = 0.75 * high + 0.25 * low
        semantic = 0.50 * dual + 0.40 * group_score
    dup = duplicate_penalty(parsed.instances)
    target_count = max(len(references["ppocr"]), len(references["rex"]), 1)
    overgeneration = max(0, len(parsed.instances) - target_count) / max(
        len(parsed.instances), 1
    )
    invalid = min(1.0, 0.25 * parsed.invalid_line_count + 0.50 * parsed.invalid_box_count)
    # Format is a gate rather than an additive floor. In particular, a
    # truncated repetitive completion must not earn ~0.06 merely because it
    # opened a <bbox> block. The extra count term catches drifting repetitions
    # whose boxes move enough to evade the pairwise duplicate detector.
    format_gate = parsed.format_reward * parsed.format_reward
    format_coverage = min(len(parsed.instances), target_count) / target_count
    base = (
        semantic + 0.10 * parsed.format_reward * format_coverage
    ) * format_gate
    reward = max(
        0.0,
        min(1.0, base - 0.05 * dup - 0.10 * overgeneration - 0.05 * invalid),
    )
    return OCRReferenceGroupBreakdown(
        reward=reward,
        ppocr_score=ppocr,
        rex_score=rex,
        dual_reference_score=dual,
        group_score=group_score,
        format_reward=parsed.format_reward,
        duplicate_penalty=dup,
        overgeneration_penalty=overgeneration,
        invalid_penalty=invalid,
        prediction_count=len(parsed.instances),
        group_count=len(groups),
    )


class OCRReferenceGroupRewardV2:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        rewards = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                rewards.append(score_completion_v2(str(completion), raw_solution).reward)
            except Exception:
                rewards.append(0.0)
        return rewards


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftOCRReferenceGroupRewardV2(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRReferenceGroupRewardV2()(completions, solution=solution, **kwargs)

    orms["ocr_reference_group_reward_v2"] = SwiftOCRReferenceGroupRewardV2
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
            asdict(score_completion_v2(args.completion, args.solution_json)),
            ensure_ascii=False,
            indent=2,
        )
    )
