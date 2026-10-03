"""Isolated RLV2 control: same 17.7K data/rewards as RLV3 on 56 H800."""

from __future__ import annotations

from dataclasses import fields, replace
import math
from pathlib import Path
from typing import Any

import yaml

from train.rl.distributed.causal import build_generation_group, deepspeed_config
from train.rl.current import trainer as legacy
from train.rl.current.config import CausalJustGRPOConfig
from train.rl.shared.causal_data import EXPECTED_ROWS, MultirouteRLV2Mixture
from train.rl.shared.reward import MultiRouteGAMRewardAdapter


GROUPS_PER_STEP = 7
TOTAL_STEPS = math.ceil(EXPECTED_ROWS / GROUPS_PER_STEP)
SAVE_STEPS = 281
assert TOTAL_STEPS == 2529 and TOTAL_STEPS % SAVE_STEPS == 0
_released_validate = CausalJustGRPOConfig.validate


def validate_multiroute(self: CausalJustGRPOConfig) -> CausalJustGRPOConfig:
    # Reuse the released V2 validator for every algorithmic field, substituting
    # only the topology/data-derived optimizer budget.
    _released_validate(replace(self, total_optimizer_steps=3653, save_steps=281))
    if self.total_optimizer_steps != TOTAL_STEPS or self.save_steps != SAVE_STEPS:
        raise ValueError(
            f"RLV2 17.7K budget drift: steps/save={self.total_optimizer_steps}/"
            f"{self.save_steps}, expected={TOTAL_STEPS}/{SAVE_STEPS}"
        )
    return self


def load_config_multiroute(path: str | Path, **overrides: Any) -> CausalJustGRPOConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(payload.get("causal_justgrpo", payload))
    values.update({key: value for key, value in overrides.items() if value is not None})
    values["total_optimizer_steps"] = TOTAL_STEPS
    values["save_steps"] = SAVE_STEPS
    known = {field.name for field in fields(CausalJustGRPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown RLV2 multi-route settings: {unknown}")
    return validate_multiroute(CausalJustGRPOConfig(**values))


def main() -> None:
    legacy.DATA_KIND = "multiroute"
    legacy.CorrectedGroundingOCRMixture = MultirouteRLV2Mixture
    legacy.JointGAMRewardAdapter = MultiRouteGAMRewardAdapter
    legacy.build_generation_group = build_generation_group
    legacy.deepspeed_config = deepspeed_config
    legacy.load_config = load_config_multiroute
    CausalJustGRPOConfig.validate = validate_multiroute
    legacy.main()


if __name__ == "__main__":
    main()
