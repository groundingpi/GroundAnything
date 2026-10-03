#!/usr/bin/env python3
"""Summarize exact supervised-response lengths by source and route.

The input sidecar is row-aligned with a sampling manifest.  This is an audit
only: it never rewrites selection indices, caches, JSONL, or training tokens.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = (32, 64, 128, 256, 512, 640, 1024)
BUCKETS = (
    ("B01", 1, 32),
    ("B02", 33, 64),
    ("B03-B04", 65, 128),
    ("B05-B08", 129, 256),
    ("B09-B16", 257, 512),
    ("B17-B20", 513, 640),
    ("B21-B32", 641, 1024),
    ("B33+", 1025, None),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0 or not np.all(values > 0):
        raise RuntimeError("response token sidecar contains an empty or non-positive slice")
    quantiles = np.quantile(values, (0.5, 0.9, 0.95, 0.99))
    result: dict[str, Any] = {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "min": int(values.min()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": int(values.max()),
        "mean_response_blocks": float(np.ceil(values / 32.0).mean()),
    }
    for threshold in THRESHOLDS:
        result[f"over_{threshold}_rows"] = int((values > threshold).sum())
        result[f"over_{threshold}_ratio"] = float((values > threshold).mean())
    result["block_buckets"] = {}
    for name, low, high in BUCKETS:
        selected = values >= low
        if high is not None:
            selected &= values <= high
        rows = int(selected.sum())
        result["block_buckets"][name] = {"rows": rows, "ratio": rows / len(values)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--response-tokens", type=Path, required=True)
    parser.add_argument("--response-audit-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-over-32-ratio", type=float, default=0.4)
    args = parser.parse_args()

    manifest_path = args.sampling_manifest.resolve(strict=True)
    response_path = args.response_tokens.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = np.load(response_path, mmap_mode="r", allow_pickle=False)
    total = int(manifest["sampled_rows"])
    if values.ndim != 1 or len(values) != total:
        raise RuntimeError("sampling/response sidecar row drift")

    route_slices: dict[str, list[np.ndarray]] = {}
    sources: list[dict[str, Any]] = []
    offset = 0
    for source in manifest["sources"]:
        rows = int(source["sampled_rows"])
        stop = offset + rows
        source_values = np.asarray(values[offset:stop], dtype=np.int32)
        route = str(source.get("route", "general"))
        sources.append(
            {
                "source_id": str(source["source_id"]),
                "route": route,
                "summary": summarize(source_values),
            }
        )
        route_slices.setdefault(route, []).append(source_values)
        offset = stop
    if offset != total:
        raise RuntimeError("manifest source row conservation failed")

    routes = {
        route: summarize(np.concatenate(parts) if len(parts) > 1 else parts[0])
        for route, parts in sorted(route_slices.items())
    }
    overall = summarize(values)
    status = "PASS" if overall["over_32_ratio"] >= args.minimum_over_32_ratio else "FAIL"
    parent_response_audit = None
    if args.response_audit_report is not None:
        parent_path = args.response_audit_report.resolve(strict=True)
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent.get("status") != "PASS":
            raise RuntimeError("parent response-length audit is not PASS")
        if parent.get("sampling_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("parent response-length audit sampling SHA drift")
        if parent.get("response_tokens_sha256") != sha256_file(response_path):
            raise RuntimeError("parent response-length audit sidecar SHA drift")
        parent_response_audit = {
            "report": str(parent_path),
            "report_sha256": sha256_file(parent_path),
            "tokenizer_path": parent.get("tokenizer_path"),
        }
    report = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "exact enabled assistant body tokens plus supervised im_end; injected empty-thinking prefix excluded",
        "block_size": 32,
        "minimum_over_32_ratio": args.minimum_over_32_ratio,
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": sha256_file(manifest_path),
        "response_tokens": str(response_path),
        "response_tokens_sha256": sha256_file(response_path),
        "parent_response_audit": parent_response_audit,
        "overall": overall,
        "routes": routes,
        "sources": sources,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({"status": status, "overall": overall, "routes": routes}, ensure_ascii=False, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
