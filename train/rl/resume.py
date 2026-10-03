"""Fail-closed validation for resumable TraceRL DeepSpeed checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
_OPTIMIZER_PATTERN = re.compile(
    r"bf16_zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt"
)


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: Path
    optimizer_step: int
    tag: str
    trainer_state: dict[str, Any]


def validate_resume_checkpoint(path: Path, expected_world_size: int) -> ResumeCheckpoint:
    """Validate every artifact needed for an exact ZeRO-1 optimizer resume.

    This intentionally checks structure and rank coverage without deserializing
    the ~60 GiB optimizer payload. DeepSpeed performs tensor/schema validation
    collectively when ``load_checkpoint`` is called by all ranks.
    """

    checkpoint = path.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint is not a directory: {checkpoint}")
    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"invalid resume checkpoint name: {checkpoint.name}")
    path_step = int(match.group(1))

    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file() or trainer_state_path.stat().st_size <= 0:
        raise ValueError(f"missing trainer state: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    state_step = int(trainer_state.get("optimizer_step", -1))
    metric_step = int(trainer_state.get("metrics", {}).get("optimizer_step", state_step))
    if state_step <= 0 or state_step != path_step or metric_step != state_step:
        raise ValueError(
            "resume optimizer-step mismatch: "
            f"path={path_step} state={state_step} metrics={metric_step}"
        )

    latest = checkpoint / "latest"
    if not latest.is_file():
        raise ValueError(f"missing DeepSpeed latest tag: {latest}")
    tag = latest.read_text(encoding="utf-8").strip()
    if not tag or "/" in tag or tag in {".", ".."}:
        raise ValueError(f"unsafe DeepSpeed tag: {tag!r}")
    tagged = checkpoint / tag
    model_state = tagged / "mp_rank_00_model_states.pt"
    exported_model = checkpoint / "model.safetensors"
    for artifact in (model_state, exported_model):
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"missing resume artifact: {artifact}")

    ranks: list[int] = []
    for shard in tagged.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"):
        match = _OPTIMIZER_PATTERN.fullmatch(shard.name)
        if match is None or shard.stat().st_size <= 0:
            raise ValueError(f"invalid optimizer shard: {shard}")
        ranks.append(int(match.group(1)))
    expected_ranks = list(range(expected_world_size))
    if sorted(ranks) != expected_ranks:
        raise ValueError(
            f"ZeRO-1 optimizer rank coverage mismatch: got={sorted(ranks)} "
            f"expected={expected_ranks}"
        )
    return ResumeCheckpoint(checkpoint, state_step, tag, trainer_state)


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject optimizer resumes under different training dynamics."""

    # ``max_runtime_hours`` was an operational wall-clock kill switch, not a
    # training-dynamics setting.  It has been removed from TraceRL; tolerate it
    # only in checkpoints created before that removal so exact optimizer-state
    # resume remains possible.
    saved = {key: value for key, value in saved.items() if key != "max_runtime_hours"}
    current = {key: value for key, value in current.items() if key != "max_runtime_hours"}
    if saved != current:
        changed = sorted(
            key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
        )
        raise ValueError(f"resume config drift is forbidden: changed={changed}")
