"""Task-aware reward routing for joint GAM Grounding + OCR GRPO."""

from __future__ import annotations

import json
from typing import Any, Callable

from train.rl.rewards.grounding_bbox_reward import (
    _as_solution as _as_grounding_solution,
    score_completion_original_strict_iou,
    score_completion_rexomni_paper,
)
from train.rl.rewards.ocr_gam_native_reward import GAM_OCR_FORMAT, score_gam_mixed_reference


GROUNDING_FORMAT = "gam_bbox_v1"


def _solution_dict(solution: dict[str, Any] | str) -> dict[str, Any]:
    parsed = json.loads(solution) if isinstance(solution, str) else solution
    if not isinstance(parsed, dict):
        raise TypeError("solution must be a JSON object")
    return parsed


class _JointGroundingReward:
    score_fn: Callable = staticmethod(score_completion_rexomni_paper)

    def __call__(self, completions, solution=None, target_labels=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [None for _ in completions]
        labels = target_labels or kwargs.get("target_label") or [None] * len(completions)
        output: list[float | None] = []
        for completion, raw_solution, sample_labels in zip(completions, solutions, labels):
            try:
                parsed = _solution_dict(raw_solution)
                if parsed.get("format") != GROUNDING_FORMAT:
                    output.append(None)
                    continue
                output.append(
                    self.score_fn(
                        str(completion),
                        _as_grounding_solution(parsed),
                        sample_labels,
                    ).reward
                )
            except Exception:
                output.append(0.0)
        return output


class JointGAMGroundingRexOmniReward(_JointGroundingReward):
    score_fn = staticmethod(score_completion_rexomni_paper)


class JointGAMGroundingStrictIoUReward(_JointGroundingReward):
    score_fn = staticmethod(score_completion_original_strict_iou)


class JointGAMOCRMixedReward:
    def __call__(self, completions, solution=None, **kwargs):
        solutions = solution or kwargs.get("solutions")
        if solutions is None:
            return [None for _ in completions]
        output: list[float | None] = []
        for completion, raw_solution in zip(completions, solutions):
            try:
                parsed = _solution_dict(raw_solution)
                if parsed.get("format") != GAM_OCR_FORMAT:
                    output.append(None)
                    continue
                output.append(score_gam_mixed_reference(str(completion), parsed).reward)
            except Exception:
                output.append(0.0)
        return output


try:
    from swift.rewards import ORM, orms  # type: ignore

    class SwiftJointGAMGroundingRexOmniReward(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return JointGAMGroundingRexOmniReward()(
                completions, solution=solution, **kwargs
            )

    class SwiftJointGAMGroundingStrictIoUReward(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return JointGAMGroundingStrictIoUReward()(
                completions, solution=solution, **kwargs
            )

    class SwiftJointGAMOCRMixedReward(ORM):
        def __call__(self, completions, solution=None, **kwargs):
            return JointGAMOCRMixedReward()(completions, solution=solution, **kwargs)

    orms["joint_gam_grounding_rexomni"] = SwiftJointGAMGroundingRexOmniReward
    orms["joint_gam_grounding_strict_iou"] = SwiftJointGAMGroundingStrictIoUReward
    orms["joint_gam_ocr_mixed"] = SwiftJointGAMOCRMixedReward
except Exception:
    pass
