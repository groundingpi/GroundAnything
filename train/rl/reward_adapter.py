"""Thin adapter around the released joint GAM Grounding/OCR rewards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch




@dataclass(frozen=True)
class RewardResult:
    components: dict[str, float]
    weights: dict[str, float]
    format_valid: bool


class JointGAMRewardAdapter:
    def __init__(self) -> None:
        from train.rl.rewards.joint_gam_reward import (  # type: ignore
            JointGAMGroundingRexOmniReward,
            JointGAMGroundingStrictIoUReward,
            JointGAMOCRMixedReward,
        )

        self.grounding_rex = JointGAMGroundingRexOmniReward()
        self.grounding_iou = JointGAMGroundingStrictIoUReward()
        self.ocr = JointGAMOCRMixedReward()

    @staticmethod
    def _format_valid(response: str) -> bool:
        required = (
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
        )
        return all(token in response for token in required)

    def score(self, response: str, row: dict[str, Any]) -> RewardResult:
        task_type = str(row.get("task_type", ""))
        solution = [row["solution"]]
        if task_type == "grounding_bbox_gam":
            labels = [row.get("target_labels")]
            rex = self.grounding_rex([response], solution=solution, target_labels=labels)[0]
            iou = self.grounding_iou([response], solution=solution, target_labels=labels)[0]
            return RewardResult(
                components={"grounding_rexomni": float(rex or 0.0), "grounding_strict_iou": float(iou or 0.0)},
                weights={"grounding_rexomni": 0.7, "grounding_strict_iou": 0.3},
                format_valid=self._format_valid(response),
            )
        if task_type == "ocr_bbox_text_gam":
            value = self.ocr([response], solution=solution)[0]
            return RewardResult(
                components={"ocr_mixed": float(value or 0.0)},
                weights={"ocr_mixed": 1.0},
                format_valid=self._format_valid(response),
            )
        raise ValueError(f"unsupported GRPO task_type: {task_type!r}")


def gdpo_total_rewards(component_rows: list[dict[str, float]], weights: dict[str, float]) -> torch.Tensor:
    """Match active-column GDPO scaling before final group normalization."""

    if not component_rows:
        raise ValueError("empty reward group")
    keys = tuple(weights)
    if any(set(row) != set(keys) for row in component_rows):
        raise ValueError("reward component routing drift within one prompt group")
    total = torch.zeros(len(component_rows), dtype=torch.float32)
    for key in keys:
        values = torch.tensor([row[key] for row in component_rows], dtype=torch.float32)
        std = values.std(unbiased=False)
        normalized = torch.zeros_like(values) if float(std) == 0.0 else (values - values.mean()) / (std + 1.0e-6)
        total.add_(normalized, alpha=float(weights[key]))
    return total
