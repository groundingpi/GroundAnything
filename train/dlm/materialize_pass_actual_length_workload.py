#!/usr/bin/env python3
"""Attach conservative Qwen3 DLM workload lengths to a passing real-length audit.

This is the no-rejection companion to ``drop_actual_length_rejections.py``.
It preserves the sampling manifest byte-for-byte and only writes derived audit
and length sidecars used for workload-balanced packing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npy(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--selected-lengths", type=Path, required=True)
    parser.add_argument("--actual-length-audit-report", type=Path, required=True)
    parser.add_argument("--output-clean-lengths", type=Path, required=True)
    parser.add_argument("--output-workload-lengths", type=Path, required=True)
    parser.add_argument("--output-actual-audit", type=Path, required=True)
    args = parser.parse_args()

    sampling = args.sampling_manifest.resolve(strict=True)
    lengths_path = args.selected_lengths.resolve(strict=True)
    parent_report_path = args.actual_length_audit_report.resolve(strict=True)
    manifest = json.loads(sampling.read_text(encoding="utf-8"))
    report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    rows = int(manifest["sampled_rows"])
    if report.get("status") != "PASS" or report.get("invalid_rows"):
        raise RuntimeError("materialization requires a passing audit without rejects")
    if report.get("sampling_manifest_sha256") != sha256_file(sampling):
        raise RuntimeError("actual-length audit sampling SHA drift")
    if report.get("selected_lengths_sha256") != sha256_file(lengths_path):
        raise RuntimeError("actual-length audit selected-length SHA drift")

    arrays_path = Path(report["arrays"]).resolve(strict=True)
    if report.get("arrays_sha256") != sha256_file(arrays_path):
        raise RuntimeError("actual-length audit array SHA drift")
    arrays = np.load(arrays_path, allow_pickle=False)
    positions = np.asarray(arrays["positions"], dtype=np.int64)
    cached = np.asarray(arrays["cached_lengths"], dtype=np.int32)
    actual = np.asarray(arrays["actual_lengths"], dtype=np.int32)
    joint = np.asarray(arrays["actual_joint_lengths"], dtype=np.int32)
    if not (len(positions) == len(cached) == len(actual) == len(joint) == int(report["audited_rows"])):
        raise RuntimeError("actual-length audit array row drift")
    if len(np.unique(positions)) != len(positions) or np.any(positions < 0) or np.any(positions >= rows):
        raise RuntimeError("invalid audited positions")

    selected = np.asarray(np.load(lengths_path, allow_pickle=False), dtype=np.int32)
    if len(selected) != rows or not np.array_equal(selected[positions], cached):
        raise RuntimeError("selected-length sidecar no longer matches audited positions")
    clean = selected.copy()
    clean[positions] = actual
    noisy = joint.astype(np.int64) - actual.astype(np.int64)
    exact_workload = actual.astype(np.int64) + 2 * noisy
    if np.any(noisy < 0) or np.any(exact_workload <= 0):
        raise RuntimeError("invalid clean/noisy workload lengths")
    observed_ratio = exact_workload.astype(np.float64) / cached.astype(np.float64)
    scale = float(np.ceil(observed_ratio.max() * 10.0) / 10.0)
    workload = np.ceil(selected.astype(np.float64) * scale).astype(np.int64)
    workload[positions] = exact_workload
    if np.any(workload > np.iinfo(np.int32).max):
        raise RuntimeError("packing workload exceeds int32")

    clean_sha = save_npy(args.output_clean_lengths.resolve(), clean.astype(np.int32))
    workload_sha = save_npy(args.output_workload_lengths.resolve(), workload.astype(np.int32))
    derived = dict(report)
    derived.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_lengths": str(args.output_clean_lengths.resolve()),
            "selected_lengths_sha256": clean_sha,
            "packing_workload_lengths": str(args.output_workload_lengths.resolve()),
            "packing_workload_lengths_sha256": workload_sha,
            "packing_workload_definition": (
                "exact clean+2*noisy for audited rows; conservative audited-ratio envelope "
                "times cached_length outside audit scope"
            ),
            "packing_workload_proxy_scale": scale,
            "parent_actual_length_audit": str(parent_report_path),
            "parent_actual_length_audit_sha256": sha256_file(parent_report_path),
            "derivation": "passing tail+multi-image audit with unchanged sampling manifest",
        }
    )
    atomic_json(args.output_actual_audit.resolve(), derived)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": rows,
                "audited_rows": len(positions),
                "packing_workload_proxy_scale": scale,
                "clean_lengths_sha256": clean_sha,
                "workload_lengths_sha256": workload_sha,
                "actual_audit": str(args.output_actual_audit.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
