#!/usr/bin/env python3
"""Fail-closed runtime/checkpoint gate for the isolated RLV3 DARE route."""

from __future__ import annotations


import argparse
from collections import Counter
import json
import math
from pathlib import Path


WORLD_SIZE = 56
PARITY_MEAN_MAX = 0.02
PARITY_P99_MAX = 0.05
PARITY_RATIO_MIN = 0.98
PARITY_RATIO_MAX = 1.02


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def audit_runtime(output: Path) -> dict[str, object]:
    root = output / "runtime-audit"
    bindings = [load_json(root / f"gpu-binding-rank-{rank:02d}.json") for rank in range(WORLD_SIZE)]
    models = [load_json(root / f"model-audit-rank-{rank:02d}.json") for rank in range(WORLD_SIZE)]
    if any(row.get("status") != "PASS" for row in bindings + models):
        raise RuntimeError("RLV3 runtime audit contains a non-PASS worker")
    if sorted(int(row["rank"]) for row in bindings) != list(range(WORLD_SIZE)):
        raise RuntimeError("RLV3 GPU-binding rank coverage drift")
    by_node = Counter(str(row["ray_node_id"]) for row in bindings)
    if sorted(by_node.values()) != [8] * 7:
        raise RuntimeError(f"RLV3 Ray topology drift: {dict(by_node)}")
    for node in by_node:
        devices = {
            tuple(row["ray_accelerator_ids"])
            for row in bindings
            if str(row["ray_node_id"]) == node
        }
        if len(devices) != 8 or any(len(item) != 1 for item in devices):
            raise RuntimeError(f"RLV3 GPU collision on node={node}: {devices}")
    for row in models:
        groups = row.get("groups") or {}
        vision = groups.get("vision_encoder") or {}
        projector = groups.get("projector") or {}
        language = groups.get("language") or {}
        if not (
            row.get("policy_mode") == "causal"
            and row.get("freeze_vision_encoder") is True
            and row.get("freeze_projector") is False
            and int(vision.get("total", 0)) > 0
            and int(vision.get("trainable", -1)) == 0
            and int(projector.get("total", 0)) > 0
            and int(projector.get("trainable", -1)) == int(projector.get("total", 0))
            and int(language.get("total", 0)) > 0
            and int(language.get("trainable", -1)) == int(language.get("total", 0))
        ):
            raise RuntimeError(f"RLV3 trainability drift: {row}")
    return {"worker_count": WORLD_SIZE, "node_count": 7, "workers_per_node": dict(by_node)}


def audit_metrics(output: Path, expected_step: int) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row.get("step", -1)) == expected_step]
    if not matches:
        raise RuntimeError(f"RLV3 metrics missing step={expected_step}")
    metrics = {str(key): float(value) for key, value in matches[-1]["metrics"].items()}
    required = {
        "training/global_step",
        "actor/pg_loss",
        "actor/grad_norm",
        "critic/score/mean",
        "gam_rlv3/mean_abs_rollout_old_logp",
        "gam_rlv3/p99_abs_rollout_old_logp",
        "gam_rlv3/ratio_mean",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise RuntimeError(f"RLV3 required metrics missing: {missing}")
    if any(not math.isfinite(metrics[key]) for key in required):
        raise RuntimeError("RLV3 has non-finite reward/loss/gradient/parity metrics")
    if int(metrics["training/global_step"]) != expected_step:
        raise RuntimeError("RLV3 global-step metric drift")
    if metrics["actor/grad_norm"] <= 0:
        raise RuntimeError(f"RLV3 non-positive gradient norm: {metrics['actor/grad_norm']}")
    if not 0.0 <= metrics["critic/score/mean"] <= 1.0 + 1e-6:
        raise RuntimeError(f"RLV3 reward out of range: {metrics['critic/score/mean']}")
    if metrics["gam_rlv3/mean_abs_rollout_old_logp"] > PARITY_MEAN_MAX:
        raise RuntimeError("RLV3 rollout/teacher-forcing mean log-prob parity failed")
    if metrics["gam_rlv3/p99_abs_rollout_old_logp"] > PARITY_P99_MAX:
        raise RuntimeError("RLV3 rollout/teacher-forcing p99 log-prob parity failed")
    if not PARITY_RATIO_MIN <= metrics["gam_rlv3/ratio_mean"] <= PARITY_RATIO_MAX:
        raise RuntimeError("RLV3 rollout/teacher-forcing ratio parity failed")
    return metrics


def audit_checkpoint(output: Path, expected_step: int) -> dict[str, object]:
    from safetensors import safe_open

    checkpoint = output / f"global_step_{expected_step}"
    actor = checkpoint / "actor"
    hf = actor / "huggingface"
    for path in (actor / "config.json", hf / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    weights = sorted(hf.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"RLV3 HF checkpoint has no safetensors: {hf}")
    tensor_count = 0
    total_bytes = 0
    for path in weights:
        total_bytes += path.stat().st_size
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            tensor_count += len(list(stream.keys()))
    if total_bytes < 5_000_000_000 or tensor_count < 100:
        raise RuntimeError(
            f"RLV3 HF checkpoint appears truncated: bytes={total_bytes} tensors={tensor_count}"
        )
    return {
        "checkpoint": str(checkpoint),
        "actor_checkpoint": str(actor),
        "hf_checkpoint": str(hf),
        "weight_files": len(weights),
        "weight_bytes": total_bytes,
        "tensor_count": tensor_count,
    }


def audit_rollout(output: Path, run_kind: str) -> dict[str, object] | None:
    if run_kind != "smoke":
        return None
    path = output / "rollouts/1.jsonl"
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("RLV3 smoke rollout artifact is missing")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("RLV3 smoke rollout artifact is empty")
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    expected_step = 1 if args.run_kind == "smoke" else 317
    payload: dict[str, object] = {
        "status": "PASS",
        "run_kind": args.run_kind,
        "task_id": args.task_id,
        "global_step": expected_step,
        "output": str(output),
        "runtime": audit_runtime(output),
        "metrics": audit_metrics(output, expected_step),
        "checkpoint_audit": audit_checkpoint(output, expected_step),
        "rollout_audit": audit_rollout(output, args.run_kind),
        "parity_thresholds": {
            "mean_abs_max": PARITY_MEAN_MAX,
            "p99_abs_max": PARITY_P99_MAX,
            "ratio_min": PARITY_RATIO_MIN,
            "ratio_max": PARITY_RATIO_MAX,
        },
    }
    checkpoint = payload["checkpoint_audit"]
    assert isinstance(checkpoint, dict)
    payload.update(
        {
            "checkpoint": checkpoint["checkpoint"],
            "actor_checkpoint": checkpoint["actor_checkpoint"],
            "hf_checkpoint": checkpoint["hf_checkpoint"],
        }
    )
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.gate.with_name(f".{args.gate.name}.tmp-{args.task_id}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.gate)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
