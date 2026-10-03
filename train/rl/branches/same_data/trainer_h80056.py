"""TraceRL V1 on the exact 17.7K multi-route RLV2 data contract.

This adapter intentionally changes only the dataset/reward routing and the
56-rank process topology.  The TraceRL rollout, replay, loss, optimizer and
fixed DecodeV4 scheduler remain the released V1 implementation.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

import os
from pathlib import Path

import torch.distributed as dist

from train.rl.branches.trace import trainer as legacy
from train.rl.distributed.trace import build_process_groups, deepspeed_config
from train.rl.shared.causal_data import MultirouteRLV2Mixture
from train.rl.shared.reward import MultiRouteGAMRewardAdapter


class AuditedMultirouteV1Mixture(MultirouteRLV2Mixture):
    """Persist the same deterministic order audit used by the RLV2 run."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        target = os.environ.get("GAM_RL_V1_MULTI_DATA_AUDIT_PATH", "")
        if target and dist.is_initialized() and dist.get_rank() == 0:
            self.write_audit(Path(target))


def main() -> None:
    # The approved RLV2 control uses the same immutable rows and reward
    # adapter.  Only the policy optimization algorithm remains TraceRL V1.
    legacy.GroundingOCRMixture = AuditedMultirouteV1Mixture
    legacy.JointGAMRewardAdapter = MultiRouteGAMRewardAdapter
    legacy.build_process_groups = build_process_groups
    legacy.deepspeed_config = deepspeed_config
    legacy.main()


if __name__ == "__main__":
    main()
