"""Minimal lossless trajectory representation for fixed B32 DecodeV4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GAMTraceTrajectory:
    output_ids: list[int]
    commit_step: list[int]
    old_logprobs: list[float]
    action_mask: list[int]
    visible_mask: list[int]
    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    policy_version: str = ""
    prompt_id: str = ""
    forced_one_tokens: int = 0
    normal_accept_tokens: int = 0
    num_denoise_forwards: int = 0

    def validate(self, *, block_size: int = 32) -> "GAMTraceTrajectory":
        lengths = {
            len(self.output_ids),
            len(self.commit_step),
            len(self.old_logprobs),
            len(self.action_mask),
            len(self.visible_mask),
        }
        if len(lengths) != 1 or not self.output_ids:
            raise ValueError(f"trajectory arrays must be non-empty and aligned: {lengths}")
        seen_invisible = False
        diffusion_steps: list[int] = []
        for index, (step, action, visible) in enumerate(
            zip(self.commit_step, self.action_mask, self.visible_mask, strict=True)
        ):
            if action not in {0, 1} or visible not in {0, 1}:
                raise ValueError("action_mask and visible_mask must be binary")
            if seen_invisible and visible:
                raise ValueError("visible_mask must be one contiguous prefix")
            seen_invisible |= not bool(visible)
            if index % block_size == 0:
                if step != -1 or action != 0:
                    raise ValueError(f"causal anchor contract drift at output index {index}")
                if self.old_logprobs[index] != 0.0:
                    raise ValueError("causal anchor old_logprobs must be zero")
            else:
                if step < 0:
                    raise ValueError(f"diffusion token lacks commit step at output index {index}")
                if visible and action != 1:
                    raise ValueError("visible diffusion tokens must be trainable actions")
                diffusion_steps.append(step)
                if not visible and action:
                    raise ValueError("EOS-tail diffusion tokens cannot be actions")
        if diffusion_steps:
            unique = set(diffusion_steps)
            if unique != set(range(1, max(unique) + 1)):
                raise ValueError("diffusion commit step IDs must be positive and contiguous")
            previous_block_max = 0
            for block_start in range(0, len(self.output_ids), block_size):
                block = self.commit_step[block_start + 1 : block_start + block_size]
                if not block:
                    continue
                if min(block) <= previous_block_max:
                    raise ValueError("diffusion commit steps cannot cross physical B32 blocks")
                previous_block_max = max(block)
        return self

    @property
    def action_count(self) -> int:
        return sum(self.action_mask)

    @property
    def visible_output_ids(self) -> list[int]:
        return [token for token, visible in zip(self.output_ids, self.visible_mask, strict=True) if visible]

    @property
    def forced_one_ratio(self) -> float:
        return self.forced_one_tokens / max(self.action_count, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
