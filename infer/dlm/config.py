"""Validated inference configuration for the three GAM DLM decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DLMMode = Literal["hierarchy_step", "hierarchy_dynamic", "speculative"]
AcceptancePolicy = Literal["confidence", "entropy"]


@dataclass(frozen=True)
class DLMInferenceConfig:
    """Per-request DLM decoding policy.

    ``denoise_steps`` is used only by ``hierarchy_step`` and directly controls
    its compute budget. Dynamic mode accepts from raw, pre-filter logits using
    either confidence or DecodeV3-compatible entropy.
    """

    mode: DLMMode = "speculative"
    block_size: int = 32
    sub_block_size: int = 8
    denoise_steps: int = 8
    confidence_threshold: float = 0.9
    acceptance_policy: AcceptancePolicy = "confidence"
    entropy_threshold: float = 0.8
    use_prefix_cache: bool = False
    hierarchy_full_block: bool = False
    enforce_structured_termination: bool = False
    structured_max_coordinate_groups: int = 0
    commit_structural_boundaries: bool = False

    def validate(self, trained_block_size: int) -> "DLMInferenceConfig":
        if self.mode not in {"hierarchy_step", "hierarchy_dynamic", "speculative"}:
            raise ValueError(f"unsupported DLM inference mode: {self.mode!r}")
        if not 2 <= self.block_size <= trained_block_size:
            raise ValueError(
                f"block_size must be in [2, {trained_block_size}], got {self.block_size}"
            )
        if self.mode.startswith("hierarchy_") and (
            self.sub_block_size < 1 or self.sub_block_size > self.block_size
        ):
            raise ValueError(
                f"sub_block_size must be in [1, {self.block_size}], got {self.sub_block_size}"
            )
        if (
            self.mode == "hierarchy_step"
            and not 1 <= self.denoise_steps <= self.block_size - 1
        ):
            raise ValueError(
                f"denoise_steps must be in [1, {self.block_size - 1}], got {self.denoise_steps}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.acceptance_policy not in {"confidence", "entropy"}:
            raise ValueError(
                f"unsupported acceptance_policy: {self.acceptance_policy!r}"
            )
        if self.entropy_threshold < 0.0:
            raise ValueError("entropy_threshold must be non-negative")
        if self.use_prefix_cache and not self.mode.startswith("hierarchy_"):
            raise ValueError("prefix cache is supported only by Hierarchy modes")
        if self.hierarchy_full_block and not self.mode.startswith("hierarchy_"):
            raise ValueError("full-block denoising is supported only by Hierarchy modes")
        if self.enforce_structured_termination and not self.mode.startswith(
            "hierarchy_"
        ):
            raise ValueError("structured termination is supported only by Hierarchy modes")
        if self.structured_max_coordinate_groups not in {0, 1}:
            raise ValueError(
                "structured_max_coordinate_groups currently supports only 0 "
                "(unlimited) or 1 (strict single-box grammar)"
            )
        if (
            self.structured_max_coordinate_groups
            and not self.enforce_structured_termination
        ):
            raise ValueError(
                "structured_max_coordinate_groups requires structured termination"
            )
        if self.commit_structural_boundaries and not self.enforce_structured_termination:
            raise ValueError(
                "commit_structural_boundaries requires structured termination"
            )
        return self
