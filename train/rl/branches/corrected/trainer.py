"""Isolated 24-H800 adapter for TraceRL V1.

The legacy V1 modules are not edited.  This entrypoint replaces only the
topology/batch contract and the user-requested data sampler before invoking the
audited V1 trainer.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

from datetime import timedelta

import torch.distributed as dist

from train.rl.corrected_data import CorrectedGroundingOCRMixture
from train.rl.branches.trace import trainer as legacy


_legacy_deepspeed_config = legacy.deepspeed_config


def build_process_groups(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 24 or group_size != 8 or world_size % group_size:
        raise RuntimeError(
            f"corrected TraceRL V1 requires world24/group8, got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create corrected TraceRL V1 process group")
    return own_group, group_index, rank % group_size, world_size // group_size


def deepspeed_config(config):
    payload = _legacy_deepspeed_config(config)
    payload["train_batch_size"] = 24
    return payload


def main() -> None:
    legacy.GroundingOCRMixture = CorrectedGroundingOCRMixture
    legacy.build_process_groups = build_process_groups
    legacy.deepspeed_config = deepspeed_config
    legacy.main()


if __name__ == "__main__":
    main()
