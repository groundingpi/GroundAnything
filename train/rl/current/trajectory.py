"""Serializable causal rollout state for exact old/current policy ratios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CausalTrajectory:
    completion_ids: list[int]
    response_mask: list[int]
    old_logprobs: list[float]
    prompt_id: str
    policy_version: str
    stop_reason: str

    def validate(self, eos_token_id: int) -> "CausalTrajectory":
        widths = {len(self.completion_ids), len(self.response_mask), len(self.old_logprobs)}
        if len(widths) != 1 or not self.completion_ids:
            raise ValueError("causal trajectory vectors must be non-empty and aligned")
        if any(value != 1 for value in self.response_mask):
            raise ValueError("unpadded causal trajectory response_mask must contain only ones")
        eos_positions = [
            index for index, token in enumerate(self.completion_ids) if token == int(eos_token_id)
        ]
        if eos_positions and eos_positions[0] != len(self.completion_ids) - 1:
            raise ValueError("tokens after EOS are forbidden in DLM RL V2")
        expected = "eos" if eos_positions else "length"
        if self.stop_reason != expected:
            raise ValueError(f"causal stop reason drift: {self.stop_reason!r} != {expected!r}")
        return self
