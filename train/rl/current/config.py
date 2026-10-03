"""Fail-closed contract from docs/DLMRLV2.md."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CausalJustGRPOConfig:
    training_budget: str = "conversion_pilot"
    algorithm: str = "grpo"
    policy_mode: str = "causal"
    rollout_mode: str = "causal"
    inference_mode: str = "DecodeV4"
    num_generations: int = 8
    policy_updates_per_rollout: int = 1
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_completion_tokens: int = 1024
    clip_epsilon: float = 0.2
    reference_kl_beta: float = 0.0
    learning_rate: float = 2.0e-6
    scheduler: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    total_optimizer_steps: int = 125
    save_steps: int = 25
    seed: int = 20260812
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    deepspeed_zero_stage: int = 1
    freeze_vision_encoder: bool = True
    freeze_projector: bool = False

    def validate(self) -> "CausalJustGRPOConfig":
        fixed: dict[str, Any] = {
            "algorithm": "grpo",
            "policy_mode": "causal",
            "rollout_mode": "causal",
            "inference_mode": "DecodeV4",
            "num_generations": 8,
            "policy_updates_per_rollout": 1,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 0,
            "max_completion_tokens": 1024,
            "clip_epsilon": 0.2,
            "reference_kl_beta": 0.0,
            "learning_rate": 2.0e-6,
            "scheduler": "constant",
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "per_device_prompt_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "deepspeed_zero_stage": 1,
            "freeze_vision_encoder": True,
            "freeze_projector": False,
        }
        for name, expected in fixed.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"DLM RL V2 contract drift: {name}={actual!r}, expected {expected!r}")
        budget_steps = {
            "conversion_pilot": 125,
            "full_dataset_epoch": 3653,
        }
        expected_steps = budget_steps.get(self.training_budget)
        if expected_steps is None:
            raise ValueError(f"unknown DLM RL V2 training budget: {self.training_budget!r}")
        if self.total_optimizer_steps != expected_steps:
            raise ValueError(
                "DLM RL V2 optimizer-step budget drift: "
                f"budget={self.training_budget} steps={self.total_optimizer_steps}, "
                f"expected={expected_steps}"
            )
        if self.save_steps <= 0 or self.total_optimizer_steps % self.save_steps:
            raise ValueError("save_steps must be a positive divisor of total_optimizer_steps")
        return self


def load_config(path: str | Path, **overrides: Any) -> CausalJustGRPOConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(payload.get("causal_justgrpo", payload))
    values.update({key: value for key, value in overrides.items() if value is not None})
    known = {field.name for field in fields(CausalJustGRPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown DLM RL V2 settings: {unknown}")
    return CausalJustGRPOConfig(**values).validate()
