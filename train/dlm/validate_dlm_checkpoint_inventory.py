#!/usr/bin/env python3
"""Validate 0.1-epoch full-checkpoint coverage for a formal DLM run."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    validator = args.validator.resolve(strict=True)
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not checkpoints:
        raise RuntimeError("formal DLM run produced no checkpoints")
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    max_steps = int(state["max_steps"])
    interval = math.ceil(max_steps * 0.1)
    expected = list(range(interval, max_steps, interval)) + [max_steps]
    actual = [int(path.name.rsplit("-", 1)[1]) for path in checkpoints]
    if actual != expected:
        raise RuntimeError(f"0.1-epoch checkpoint coverage drift: expected={expected}, actual={actual}")
    for path, step in zip(checkpoints, actual):
        command = [sys.executable, str(validator), "--checkpoint", str(path),
                   "--expected-step", str(step), "--expected-world-size", str(args.world_size)]
        if step == max_steps:
            command.append("--require-complete-run")
        subprocess.run(command, check=True)
    inventory = {
        "status": "PASS",
        "max_steps": max_steps,
        "save_interval_steps": interval,
        "checkpoint_steps": actual,
        "final_checkpoint": str(checkpoints[-1]),
    }
    (output / "checkpoint-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
