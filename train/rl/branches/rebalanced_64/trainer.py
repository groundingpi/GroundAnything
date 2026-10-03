#!/usr/bin/env python3
"""Causal-JustGRPO RLV2 on eight H800 nodes (64 ranks).

The released RLV2 loop is intentionally reused byte-for-byte for model
forward, rollout, reward, advantage and optimizer updates.  This adapter only
changes the distributed generation topology and loads the already audited
``rl-rebalanced`` mixture.  It is a separate import path so
the historical 24/56-rank V1/V2 runners are not modified.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

from dataclasses import fields, replace
from datetime import timedelta
import math
from pathlib import Path
from typing import Any

import torch.distributed as dist
import yaml

from train.rl.current import trainer as legacy
from train.rl.current.config import CausalJustGRPOConfig
from train.rl.branches.rebalanced.data import EXPECTED_ROWS, RebalancedRLV2Mixture
from train.rl.branches.rebalanced.trainer import (
    TelemetryRewardAdapter,
    telemetry_append_jsonl,
    telemetry_gather_reward_group,
)


EXPECTED_WORLD_SIZE = 64
GROUP_SIZE = 8
GROUPS_PER_STEP = EXPECTED_WORLD_SIZE // GROUP_SIZE
# One deterministic full pass over the 17,700-row manifest.  The last step
# contains four rows and four deterministic modulo-wrap rows, exactly as the
# established 56-rank adapter handles its ceil-rounded tail.
TOTAL_STEPS = math.ceil(EXPECTED_ROWS / GROUPS_PER_STEP)
SAVE_STEPS = 281

_released_validate = CausalJustGRPOConfig.validate
_released_deepspeed_config = legacy.deepspeed_config


def build_generation_group(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    """Create eight independent 8-rank rollout/reward groups."""

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != EXPECTED_WORLD_SIZE or group_size != GROUP_SIZE:
        raise RuntimeError(
            "DLM RL V2 H80064 requires world64/group8, "
            f"got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    # All ranks execute new_group in the same order; this is required by NCCL.
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create DLM RL V2 H80064 generation group")
    return own_group, group_index, rank % group_size, world_size // group_size


def validate_multiroute_h80064(self: CausalJustGRPOConfig) -> CausalJustGRPOConfig:
    """Keep every released RLV2 hyperparameter; derive only 64-way steps."""

    # The released validator encodes the historical 24-rank budget (3653).
    # Validate immutable algorithmic fields through that probe, then validate
    # the topology-derived one-epoch budget here.
    _released_validate(replace(self, total_optimizer_steps=3653, save_steps=281))
    if self.total_optimizer_steps != TOTAL_STEPS:
        raise ValueError(
            "DLM RL V2 H80064 optimizer-step contract drift: "
            f"{self.total_optimizer_steps} != {TOTAL_STEPS}"
        )
    if self.save_steps <= 0:
        raise ValueError("DLM RL V2 H80064 save_steps must be positive")
    return self


def load_config_h80064(path: str | Path, **overrides: Any) -> CausalJustGRPOConfig:
    """Load the audited RLV2 YAML without invoking the 24-rank loader first."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(payload.get("causal_justgrpo", payload))
    values.update({key: value for key, value in overrides.items() if value is not None})
    known = {field.name for field in fields(CausalJustGRPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown DLM RL V2 H80064 settings: {unknown}")
    base = CausalJustGRPOConfig(**values)
    return replace(base, total_optimizer_steps=TOTAL_STEPS, save_steps=SAVE_STEPS).validate()


def deepspeed_config(config: CausalJustGRPOConfig) -> dict[str, Any]:
    """Use released ZeRO-1/BF16 settings with the actual 64-rank batch."""

    payload = _released_deepspeed_config(config)
    payload["train_batch_size"] = EXPECTED_WORLD_SIZE
    return payload


def main() -> None:
    # Process-local monkeypatches only.  No V1/V2/V3 source file is edited and
    # no global configuration is changed on disk.
    legacy.CorrectedGroundingOCRMixture = RebalancedRLV2Mixture
    legacy.JointGAMRewardAdapter = TelemetryRewardAdapter
    legacy.build_generation_group = build_generation_group
    legacy.deepspeed_config = deepspeed_config
    legacy.load_config = load_config_h80064
    legacy.gather_reward_group = telemetry_gather_reward_group
    legacy.append_jsonl = telemetry_append_jsonl
    CausalJustGRPOConfig.validate = validate_multiroute_h80064
    legacy.main()


if __name__ == "__main__":
    main()

