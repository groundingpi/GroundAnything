"""TraceRL V1 adapter for 7x8 H800, preserving the legacy trainer logic."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path

import torch.distributed as dist

from train.rl.corrected_data import CorrectedGroundingOCRMixture
from train.rl.branches.trace import trainer as legacy
from train.rl.resume import ResumeCheckpoint


EXPECTED_WORLD_SIZE = 56
GROUP_SIZE = 8
_legacy_ds_config = legacy.deepspeed_config
_legacy_validate_resume = legacy.validate_resume_checkpoint


def build_process_groups(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != EXPECTED_WORLD_SIZE or group_size != GROUP_SIZE or world_size % group_size:
        raise RuntimeError(
            f"TraceRL V1 H80056 requires world56/group8, got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create TraceRL V1 H80056 process group")
    return own_group, group_index, rank % group_size, world_size // group_size


def deepspeed_config(config):
    payload = _legacy_ds_config(config)
    payload["train_batch_size"] = EXPECTED_WORLD_SIZE
    # This is consulted only when the initial V1 handoff uses the converted
    # Universal checkpoint.  Normal 56-rank checkpoints remain legacy DS
    # checkpoints and therefore keep the original save/load behavior.
    if os.environ.get("GAM_RL_LOAD_UNIVERSAL", "0") == "1":
        payload["checkpoint"] = {"load_universal": True}
    return payload


def validate_resume_checkpoint(path: Path, expected_world_size: int) -> ResumeCheckpoint:
    """Validate the sidecar for a Universal checkpoint handoff.

    The old V1 checkpoint has 24 rigid ZeRO shards.  DeepSpeed's Universal
    loader performs the safe 24->56 repartition; rejecting it by the old
    rank-coverage check would make the requested scale-up impossible.
    """
    if os.environ.get("GAM_RL_LOAD_UNIVERSAL", "0") != "1":
        return _legacy_validate_resume(path, expected_world_size)
    root = path.resolve(strict=True)
    state_source = Path(os.environ["GAM_RL_UNIVERSAL_RESUME_STATE_SOURCE"]).resolve(strict=True)
    state = json.loads(state_source.read_text(encoding="utf-8"))
    step = int(state.get("optimizer_step", -1))
    if step <= 0 or int(state.get("metrics", {}).get("optimizer_step", step)) != step:
        raise ValueError(f"invalid Universal V1 resume state: {state_source}")
    tag_file = root / "latest_universal"
    if not tag_file.is_file():
        raise ValueError(f"missing Universal checkpoint tag: {tag_file}")
    tag = tag_file.read_text(encoding="utf-8").strip()
    if not tag or "/" in tag or tag in {".", ".."}:
        raise ValueError(f"unsafe Universal checkpoint tag: {tag!r}")
    tagged = root / tag
    if not (tagged / "zero" / "optimizer_state.pt").is_file():
        raise ValueError(f"Universal optimizer state is incomplete: {tagged}")
    if not (tagged / "mp_rank_00_model_states.pt").is_file():
        raise ValueError(f"Universal model state is incomplete: {tagged}")
    return ResumeCheckpoint(root, step, tag, state)


def main() -> None:
    # Keep the corrected sampler and all V1 loss/rollout code exactly as-is.
    legacy.GroundingOCRMixture = CorrectedGroundingOCRMixture
    legacy.build_process_groups = build_process_groups
    legacy.deepspeed_config = deepspeed_config
    legacy.validate_resume_checkpoint = validate_resume_checkpoint
    legacy.main()


if __name__ == "__main__":
    main()
