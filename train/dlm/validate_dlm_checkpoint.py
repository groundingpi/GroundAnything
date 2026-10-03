#!/usr/bin/env python3
"""Fail-closed structural validation for a full DeepSpeed DLM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--require-complete-run", action="store_true")
    parser.add_argument(
        "--allow-archive-name",
        action="store_true",
        help="Allow an epoch archive directory such as 2.0; global_step and DeepSpeed markers remain strict.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    require(checkpoint.is_dir(), f"checkpoint is not a directory: {checkpoint}")
    if not args.allow_archive_name:
        require(
            checkpoint.name == f"checkpoint-{args.expected_step}",
            f"checkpoint step/name drift: {checkpoint.name}",
        )
    state_path = checkpoint / "trainer_state.json"
    model_path = checkpoint / "model.safetensors"
    training_args = checkpoint / "training_args.bin"
    for required in (state_path, model_path, training_args):
        require(required.is_file() and required.stat().st_size > 0, f"missing checkpoint file: {required}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(int(state["global_step"]) == args.expected_step, "trainer global_step drift")
    if args.require_complete_run:
        require(int(state["global_step"]) == int(state["max_steps"]), "checkpoint is not the completed run")

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    require(keys, "wrapper model has no tensors")
    require(all(key.startswith("base_model.") for key in keys), "checkpoint is not a DLM wrapper state dict")

    rng_files = sorted(checkpoint.glob("rng_state*.pth"))
    require(len(rng_files) == args.expected_world_size, f"RNG shard count drift: {len(rng_files)}")
    # Transformers uses an unranked RNG file for a one-process training run.
    if args.expected_world_size == 1 and (checkpoint / "rng_state.pth").is_file():
        expected_rng = {"rng_state.pth"}
    else:
        expected_rng = {f"rng_state_{rank}.pth" for rank in range(args.expected_world_size)}
    require({path.name for path in rng_files} == expected_rng, "RNG rank coverage drift")

    global_step_dir = checkpoint / f"global_step{args.expected_step}"
    require(global_step_dir.is_dir(), f"missing DeepSpeed state directory: {global_step_dir}")
    optimizer_files = sorted(global_step_dir.glob("*_optim_states.pt"))
    require(
        len(optimizer_files) == args.expected_world_size,
        f"optimizer shard count drift: {len(optimizer_files)}",
    )
    model_state_files = sorted(global_step_dir.glob("*model_states.pt"))
    require(len(model_state_files) == 1, f"DeepSpeed model-state count drift: {len(model_state_files)}")
    require(all(path.stat().st_size > 0 for path in optimizer_files + model_state_files), "empty DeepSpeed shard")
    latest = checkpoint / "latest"
    require(latest.is_file(), "missing DeepSpeed latest marker")
    require(latest.read_text(encoding="utf-8").strip() == f"global_step{args.expected_step}", "latest marker drift")

    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint": str(checkpoint),
                "global_step": args.expected_step,
                "max_steps": int(state["max_steps"]),
                "wrapper_tensors": len(keys),
                "model_bytes": model_path.stat().st_size,
                "rng_shards": len(rng_files),
                "optimizer_shards": len(optimizer_files),
                "deepspeed_model_state_files": len(model_state_files),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
