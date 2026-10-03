"""GAM-native OCR bbox-text parser and mixed-reference reward.

Expected completion format::

    <|object_ref_start|>text<|object_ref_end|>
    <|box_start|><x1><y1><x2><y2>,...<|box_end|>, ...

The semantic scoring remains identical to the established OCR rewards.  Only
the completion parser and its format gate are GAM-native.  Repeated text in
different boxes is deliberately preserved as separate OCR instances.
"""

from __future__ import annotations

import json
import re
from typing import Any

from train.rl.rewards.ocr_dual_reference_reward import (
    _as_solution,
    _reference_instances,
    duplicate_penalty,
    score_reference,
)
from train.rl.rewards.ocr_mixed_reference_reward import (
    MULTI_TEACHER_V1,
    REFERENCE_GROUP_V2,
    detect_reward_mode,
)
from train.rl.rewards.ocr_multiview_reward import (
    StrictParsedCompletion,
    _empty_breakdown,
    _finalize,
    _references,
    score_geometry_view,
    score_text_view,
)
from train.rl.rewards.ocr_reference_group_reward import (
    OCRReferenceGroupBreakdown,
    score_reference_groups,
)


GAM_OCR_FORMAT = "gam_ocr_bbox_text_v1"
GAM_GROUP_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.S,
)
GAM_BOX_RE = re.compile(
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
)
GAM_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _solution_dict(solution: dict[str, Any] | str) -> dict[str, Any]:
    parsed = json.loads(solution) if isinstance(solution, str) else solution
    if not isinstance(parsed, dict):
        raise TypeError("solution must be a JSON object")
    return parsed


def is_gam_ocr_solution(solution: dict[str, Any] | str) -> bool:
    try:
        return _solution_dict(solution).get("format") == GAM_OCR_FORMAT
    except Exception:
        return False


def _valid_gam_box(box: list[int]) -> bool:
    return (
        len(box) == 4
        and 0 <= box[0] < box[2] <= 999
        and 0 <= box[1] < box[3] <= 999
    )


def parse_gam_ocr_completion(text: str) -> StrictParsedCompletion:
    """Parse GAM OCR groups without collapsing repeated text labels."""
    cleaned = str(text or "").split("<|im_end|>", 1)[0]
    cleaned = GAM_THINK_RE.sub("", cleaned)
    cleaned = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", "", cleaned)
    matches = list(GAM_GROUP_RE.finditer(cleaned))
    if not matches:
        return StrictParsedCompletion([], 0.0, 1, 0, int(bool(cleaned.strip())))

    instances: list[dict[str, Any]] = []
    invalid_groups = 0
    invalid_boxes = 0
    for match in matches:
        label = match.group(1).strip()
        body = match.group(2).strip()
        if not label:
            invalid_groups += 1
            continue
        valid_count = 0
        for box_match in GAM_BOX_RE.finditer(body):
            box = [int(box_match.group(index)) for index in range(1, 5)]
            if not _valid_gam_box(box):
                invalid_boxes += 1
                continue
            instances.append({"bbox_norm1000": box, "text": label})
            valid_count += 1
        residual = GAM_BOX_RE.sub("", body)
        residual = re.sub(r"[\s,]", "", residual)
        if residual or valid_count == 0:
            invalid_groups += 1

    outside = GAM_GROUP_RE.sub("", cleaned)
    outside = re.sub(r"[\s,]", "", outside)
    outside_count = int(bool(outside))
    if instances and invalid_groups == 0 and invalid_boxes == 0 and outside_count == 0:
        format_reward = 1.0
    elif instances:
        format_reward = 0.45
    else:
        format_reward = 0.0
    return StrictParsedCompletion(
        instances=instances,
        format_reward=format_reward,
        invalid_line_count=invalid_groups,
        invalid_box_count=invalid_boxes,
        outside_content_count=outside_count,
    )


def _score_reference_groups_gam(
    parsed: StrictParsedCompletion,
    solution: dict[str, Any] | str,
) -> OCRReferenceGroupBreakdown:
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
        group
        for group in parsed_solution.get("reference_groups") or []
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
    invalid = min(
        1.0,
        0.25 * parsed.invalid_line_count
        + 0.50 * parsed.invalid_box_count
        + 0.50 * parsed.outside_content_count,
    )
    format_gate = parsed.format_reward * parsed.format_reward
    coverage = min(len(parsed.instances), target_count) / target_count
    base = (semantic + 0.10 * parsed.format_reward * coverage) * format_gate
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


def _score_multiteacher_gam(
    parsed: StrictParsedCompletion,
    solution: dict[str, Any] | str,
):
    mode = "multi_teacher"
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
    ppocr_geometry = score_geometry_view(parsed.instances, ppocr) if ppocr else 0.0
    semantic = 0.95 * teacher_score + 0.05 * ppocr_geometry if ppocr else teacher_score
    target_count = round(__import__("statistics").median([len(rex), len(slow), len(hybrid)]))
    return _finalize(
        mode,
        parsed,
        semantic,
        target_count,
        rex + slow + hybrid + ppocr,
        rex_score=rex_score,
        slow_score=slow_score,
        hybrid_score=hybrid_score,
        ppocr_geometry_score=ppocr_geometry,
    )


def score_gam_mixed_reference(
    completion: str,
    solution: dict[str, Any] | str,
):
    """Return the established semantic reward under a GAM-only format gate."""
    if not is_gam_ocr_solution(solution):
        raise ValueError(f"expected solution.format={GAM_OCR_FORMAT!r}")
    parsed = parse_gam_ocr_completion(completion)
    mode = detect_reward_mode(solution)
    if mode == MULTI_TEACHER_V1:
        return _score_multiteacher_gam(parsed, solution)
    if mode == REFERENCE_GROUP_V2:
        return _score_reference_groups_gam(parsed, solution)
    raise ValueError(f"unsupported OCR reward mode: {mode}")


class OCRGAMNativeMixedReferenceRewardV1:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        output = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                output.append(score_gam_mixed_reference(str(completion), raw_solution).reward)
            except Exception:
                output.append(0.0)
        return output


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftOCRGAMNativeMixedReferenceRewardV1(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRGAMNativeMixedReferenceRewardV1()(
                completions, solution=solution, **kwargs
            )

    orms["ocr_gam_native_mixed_reference_reward_v1"] = (
        SwiftOCRGAMNativeMixedReferenceRewardV1
    )
except Exception:
    pass
