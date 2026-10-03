"""Causal-JustGRPO V2 adapter for 7x8 H800."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import math
import os

import torch.distributed as dist

from train.rl.corrected_data import CorrectedGroundingOCRMixture
from train.rl import current as _pkg  # noqa: F401
from train.rl.current import trainer as legacy
from train.rl.current.config import CausalJustGRPOConfig


EXPECTED_WORLD_SIZE = 56
GROUP_SIZE = 8
_legacy_ds_config = legacy.deepspeed_config
_legacy_load_config = legacy.load_config
_legacy_validate = CausalJustGRPOConfig.validate


def build_generation_group(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != EXPECTED_WORLD_SIZE or group_size != GROUP_SIZE or world_size % group_size:
        raise RuntimeError(
            f"DLM RL V2 H80056 requires world56/group8, got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create DLM RL V2 H80056 generation group")
    return own_group, group_index, rank % group_size, world_size // group_size


def _validate_56(self: CausalJustGRPOConfig) -> CausalJustGRPOConfig:
    """Retain every V2 fixed hyperparameter while allowing 56-way epoch steps."""
    # The released validator encodes the 24-rank step count (3653).  Validate
    # all fixed fields through it using the released count, then return the
    # actual 56-rank count supplied by the adapter.
    probe = replace(self, total_optimizer_steps=3653, save_steps=281)
    _legacy_validate(probe)
    expected = math.ceil(10957 / 7)  # corrected data: 6557 + 4400 rows
    if self.total_optimizer_steps != expected:
        raise ValueError(
            f"DLM RL V2 H80056 step contract drift: {self.total_optimizer_steps} != {expected}"
        )
    if self.save_steps <= 0 or expected % self.save_steps:
        raise ValueError("DLM RL V2 H80056 save_steps must divide 1566")
    return self


def load_config_56(path, **overrides):
    base = _legacy_load_config(path, **overrides)
    # 10957 rows / (56 ranks / 8 samples per reward group) = 1566 steps.
    return replace(base, total_optimizer_steps=1566, save_steps=261).validate()


def deepspeed_config(config):
    payload = _legacy_ds_config(config)
    payload["train_batch_size"] = EXPECTED_WORLD_SIZE
    return payload


def main() -> None:
    legacy.CorrectedGroundingOCRMixture = CorrectedGroundingOCRMixture
    legacy.build_generation_group = build_generation_group
    legacy.deepspeed_config = deepspeed_config
    legacy.load_config = load_config_56
    CausalJustGRPOConfig.validate = _validate_56
    legacy.main()


if __name__ == "__main__":
    main()
