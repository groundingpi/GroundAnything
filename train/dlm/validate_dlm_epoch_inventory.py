#!/usr/bin/env python3
"""Validate full checkpoints at requested data epochs and emit an inventory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--target-epochs", type=float, nargs="+", required=True)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    output = args.output.resolve(strict=True)
    validator = args.validator.resolve(strict=True)
    checkpoints = sorted(
        output.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[1])
    )
    if not checkpoints:
        raise RuntimeError("formal DLM run produced no checkpoints")

    candidates: list[tuple[Path, int, float, int]] = []
    for path in checkpoints:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        candidates.append(
            (path, int(state["global_step"]), float(state["epoch"]), int(state["max_steps"]))
        )
    max_steps = candidates[-1][3]
    if any(item[3] != max_steps for item in candidates):
        raise RuntimeError("max_steps drift across checkpoints")

    selected: dict[str, dict[str, object]] = {}
    used_steps: set[int] = set()
    for target_epoch in args.target_epochs:
        path, step, actual_epoch, _ = min(
            candidates, key=lambda item: abs(item[2] - target_epoch)
        )
        delta = abs(actual_epoch - target_epoch)
        if delta > args.tolerance:
            raise RuntimeError(
                f"no checkpoint close to epoch {target_epoch}: closest={actual_epoch:.8f}"
            )
        if step in used_steps:
            raise RuntimeError(f"checkpoint step {step} selected for multiple target epochs")
        command = [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(path),
            "--expected-step",
            str(step),
            "--expected-world-size",
            str(args.world_size),
        ]
        if target_epoch == max(args.target_epochs):
            command.append("--require-complete-run")
        subprocess.run(command, check=True)
        key = f"{target_epoch:.1f}"
        selected[key] = {
            "target_epoch": target_epoch,
            "actual_epoch": actual_epoch,
            "global_step": step,
            "checkpoint": str(path),
        }
        used_steps.add(step)

    final_key = f"{max(args.target_epochs):.1f}"
    inventory: dict[str, object] = {
        "status": "PASS",
        "world_size": args.world_size,
        "max_steps": max_steps,
        "target_epochs": args.target_epochs,
        "checkpoints": selected,
        "final_checkpoint": selected[final_key]["checkpoint"],
    }
    atomic_json(output / "checkpoint-inventory.json", inventory)
    print(json.dumps(inventory, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
