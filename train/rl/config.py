"""Strict configuration contract for GAM-TraceRL."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TraceRLConfig:
    method: str = "TraceRL"
    num_generations: int = 8
    policy_epochs_per_rollout: int = 1
    block_size: int = 32
    causal_anchor_tokens: int = 1
    diffusion_tokens_per_block: int = 31
    sub_block_size: int = 4
    token_shift: int = 1
    entropy_threshold: float = 0.8
    legacy_force_fill_all: bool = False
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 4096
    repetition_penalty: float = 0.0
    position_penalty: float = 0.0
    enable_eos_early_cut: bool = False
    expected_cardinality: int | None = None
    single_target_rule: bool = False
    task_profile: str | None = None
    clip_epsilon: float = 0.2
    learning_rate: float = 1.0e-6
    max_grad_norm: float = 1.0
    reference_kl_beta: float = 0.0
    value_model: bool = False
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    save_steps: int = 50
    seed: int = 20260812

    def validate(self) -> "TraceRLConfig":
        required = {
            "method": "TraceRL",
            "num_generations": 8,
            "policy_epochs_per_rollout": 1,
            "block_size": 32,
            "causal_anchor_tokens": 1,
            "diffusion_tokens_per_block": 31,
            "sub_block_size": 4,
            "token_shift": 1,
            "legacy_force_fill_all": False,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 0.0,
            "position_penalty": 0.0,
            "enable_eos_early_cut": False,
            "expected_cardinality": None,
            "single_target_rule": False,
            "task_profile": None,
            "reference_kl_beta": 0.0,
            "value_model": False,
        }
        for name, expected in required.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"fixed TraceRL contract drift: {name}={actual!r}, expected {expected!r}")
        if self.entropy_threshold != 0.8:
            raise ValueError("raw entropy threshold must remain 0.8")
        if self.temperature not in {0.3, 0.5, 0.7}:
            raise ValueError("temperature is restricted to 0.5 and the one-shot 0.3/0.7 fallback")
        if self.max_new_tokens != 4096:
            raise ValueError("TraceRL max_new_tokens must remain 4096")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning rate and max_grad_norm must be positive")
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        return self


def load_config(path: str | Path, **overrides: Any) -> TraceRLConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(payload.get("tracerl", payload))
    values.update({key: value for key, value in overrides.items() if value is not None})
    known = {field.name for field in fields(TraceRLConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown TraceRL settings: {unknown}")
    return TraceRLConfig(**values).validate()
