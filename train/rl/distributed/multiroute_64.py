"""RLV2 multi-route control adapter for eight-node (64-GPU) H800.

This is deliberately a new topology adapter.  The released 24-GPU V2 code and
the 56-GPU adapter are left untouched.  It uses the same 17.7K manifest,
reward adapter and causal-JustGRPO loop; only the generation topology and the
data-derived optimizer budget differ.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import math
from dataclasses import fields
from pathlib import Path

import yaml

import torch.distributed as dist

from train.rl.current import trainer as legacy
from train.rl.current.config import CausalJustGRPOConfig
from train.rl.shared.causal_data import EXPECTED_ROWS, MultirouteRLV2Mixture
from train.rl.shared.reward import MultiRouteGAMRewardAdapter
from train.rl.distributed.causal import deepspeed_config as _legacy_ds_config


EXPECTED_WORLD_SIZE = 64
GROUP_SIZE = 8
GROUPS_PER_STEP = EXPECTED_WORLD_SIZE // GROUP_SIZE
TOTAL_STEPS = math.ceil(EXPECTED_ROWS / GROUPS_PER_STEP)
SAVE_STEPS = 281

_released_validate = CausalJustGRPOConfig.validate
_released_load_config = CausalJustGRPOConfig  # type marker for static readers


def build_generation_group(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != EXPECTED_WORLD_SIZE or group_size != GROUP_SIZE:
        raise RuntimeError(
            f"DLM RL V2 H80064 requires world64/group8, got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create DLM RL V2 H80064 generation group")
    return own_group, group_index, rank % group_size, world_size // group_size


def validate_multiroute_h80064(self: CausalJustGRPOConfig) -> CausalJustGRPOConfig:
    """Validate all released V2 dynamics while deriving the 64-GPU budget.

    The released validator requires the 24-GPU historical step count and a
    divisor save cadence.  Validate those immutable algorithmic fields with a
    probe, then apply the actual 64-GPU epoch count.  ``SAVE_STEPS`` need not
    divide the ceil-rounded epoch: the loop always saves the final step.
    """

    _released_validate(replace(self, total_optimizer_steps=3653, save_steps=281))
    if self.total_optimizer_steps != TOTAL_STEPS:
        raise ValueError(
            f"RLV2 H80064 step contract drift: {self.total_optimizer_steps} != {TOTAL_STEPS}"
        )
    if self.save_steps <= 0:
        raise ValueError("RLV2 H80064 save_steps must be positive")
    return self


def load_config_h80064(path, **overrides):
    # Parse without calling the released loader: ``main`` installs the
    # topology validator on the dataclass before this function runs, and the
    # released loader would therefore validate the 56-GPU budget (3653 steps)
    # before we can replace it with the 64-GPU budget (2213 steps).
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(payload.get("causal_justgrpo", payload))
    values.update({key: value for key, value in overrides.items() if value is not None})
    known = {field.name for field in fields(CausalJustGRPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown DLM RL V2 settings: {unknown}")
    base = CausalJustGRPOConfig(**values)
    return replace(base, total_optimizer_steps=TOTAL_STEPS, save_steps=SAVE_STEPS).validate()


def deepspeed_config(config):
    payload = _legacy_ds_config(config)
    payload["train_batch_size"] = EXPECTED_WORLD_SIZE
    return payload


def main() -> None:
    # Isolate the RLV2 control data/reward from the released V2/V1 imports.
    legacy.DATA_KIND = "multiroute"
    legacy.CorrectedGroundingOCRMixture = MultirouteRLV2Mixture
    legacy.JointGAMRewardAdapter = MultiRouteGAMRewardAdapter
    legacy.build_generation_group = build_generation_group
    legacy.deepspeed_config = deepspeed_config
    legacy.load_config = load_config_h80064
    CausalJustGRPOConfig.validate = validate_multiroute_h80064
    legacy.main()


if __name__ == "__main__":
    main()
