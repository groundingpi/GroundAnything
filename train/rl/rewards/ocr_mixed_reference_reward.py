"""Route OCR GRPO rewards by each sample's solution schema.

The mixed training set combines:

* ``reference_groups_v2`` samples scored by ``score_completion_v2``.
* ``multi_teacher_views_v1`` samples scored by ``score_multiteacher``.

Registering one router ORM avoids applying both rewards to every sample, which
would otherwise dilute the valid reward with a zero from the incompatible
scorer.
"""

from __future__ import annotations

import json
from typing import Any

from train.rl.rewards.ocr_multiview_reward import score_multiteacher
from train.rl.rewards.ocr_reference_group_reward import score_completion_v2


REFERENCE_GROUP_V2 = "reference_groups_v2"
MULTI_TEACHER_V1 = "multi_teacher_views_v1"


def _as_solution(solution: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(solution, str):
        solution = json.loads(solution)
    if not isinstance(solution, dict):
        raise TypeError("solution must be a JSON object or encoded JSON object")
    return solution


def detect_reward_mode(solution: dict[str, Any] | str) -> str:
    parsed = _as_solution(solution)
    reference_mode = str(parsed.get("reference_mode") or "")
    names = {
        str(reference.get("name") or "")
        for reference in parsed.get("references") or []
        if isinstance(reference, dict)
    }

    if reference_mode == MULTI_TEACHER_V1 or {
        "rex_omni",
        "locateanything_slow",
        "locateanything_hybrid",
    }.issubset(names):
        return MULTI_TEACHER_V1
    if parsed.get("reference_groups") is not None and {
        "ppocr",
        "rex",
    }.issubset(names):
        return REFERENCE_GROUP_V2
    raise ValueError(
        "unsupported OCR reward solution schema: "
        f"reference_mode={reference_mode!r}, references={sorted(names)!r}"
    )


def score_mixed_reference(
    completion: str,
    solution: dict[str, Any] | str,
) -> float:
    mode = detect_reward_mode(solution)
    if mode == MULTI_TEACHER_V1:
        return score_multiteacher(completion, solution).reward
    return score_completion_v2(completion, solution).reward


class OCRMixedReferenceRewardV1:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [0.0 for _ in completions]
        rewards = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                rewards.append(
                    score_mixed_reference(str(completion), raw_solution)
                )
            except Exception:
                rewards.append(0.0)
        return rewards


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftOCRMixedReferenceRewardV1(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return OCRMixedReferenceRewardV1()(
                completions,
                solution=solution,
                **kwargs,
            )

    orms["ocr_mixed_reference_reward_v1"] = SwiftOCRMixedReferenceRewardV1
except Exception:
    pass

