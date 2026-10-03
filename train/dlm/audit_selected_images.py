#!/usr/bin/env python3
"""Strictly decode every image referenced by a selected DLM sidecar sample.

The audit is intentionally independent of the cached-length tail audit.  A row
with a short token length can still contain a truncated or otherwise unreadable
image, so formal training must bind both reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from functools import partial

from torch.utils.data import DataLoader, Dataset

from train.dlm.data import IndexedCacheDataset, _load_image, _typed_messages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PositionedShard(Dataset):
    def __init__(self, dataset: IndexedCacheDataset, shard_id: int, num_shards: int) -> None:
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rows = max(0, (len(dataset) - shard_id + num_shards - 1) // num_shards)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> tuple[int, dict[str, Any]]:
        position = self.shard_id + index * self.num_shards
        return position, self.dataset[position]


def image_description(value: Any) -> str:
    if not isinstance(value, dict):
        return f"<invalid-record:{type(value).__name__}>"
    path = value.get("path")
    if path:
        return str(path)
    payload = value.get("bytes")
    size = len(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else None
    return f"<embedded-bytes:{size}>"


def audit_batch(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    decode_images: bool = True,
) -> dict[str, Any]:
    decoded_images = 0
    validated_multimodal_rows = 0
    invalid_rows: list[dict[str, Any]] = []
    for position, row in rows:
        row_errors: list[dict[str, Any]] = []
        images = row.get("images")
        if not isinstance(images, list):
            images = []
            row_errors.append(
                {
                    "image_index": None,
                    "image": "<images-field>",
                    "error_type": "TypeError",
                    "error": f"images is {type(row.get('images')).__name__}, expected list",
                }
            )
        else:
            try:
                _typed_messages(row.get("messages"), len(images))
                validated_multimodal_rows += 1
            except Exception as error:  # noqa: BLE001 - retain every malformed row
                row_errors.append(
                    {
                        "image_index": None,
                        "image": "<messages/images-contract>",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        for image_index, value in enumerate(images if decode_images else []):
            try:
                image = _load_image(value)
                image.close()
                decoded_images += 1
            except Exception as error:  # noqa: BLE001 - audit must retain every bad row
                row_errors.append(
                    {
                        "image_index": image_index,
                        "image": image_description(value),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        if row_errors:
            invalid_rows.append(
                {
                    "position": int(position),
                    "id": str(row.get("id")),
                    "source_id": str(row.get("_dlm_source_id")),
                    "local_index": int(row.get("_dlm_local_index", -1)),
                    "errors": row_errors,
                }
            )
    return {
        "audited_rows": len(rows),
        "decoded_images": decoded_images,
        "validated_multimodal_rows": validated_multimodal_rows,
        "invalid_rows": invalid_rows,
    }


def run_part(args: argparse.Namespace) -> None:
    manifest = args.sampling_manifest.resolve(strict=True)
    dataset = IndexedCacheDataset(manifest)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    shard = PositionedShard(dataset, args.shard_id, args.num_shards)
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=partial(audit_batch, decode_images=not args.skip_image_decode),
    )
    started = time.monotonic()
    audited_rows = 0
    decoded_images = 0
    validated_multimodal_rows = 0
    invalid_rows: list[dict[str, Any]] = []
    for result in loader:
        audited_rows += int(result["audited_rows"])
        decoded_images += int(result["decoded_images"])
        validated_multimodal_rows += int(result["validated_multimodal_rows"])
        invalid_rows.extend(result["invalid_rows"])
        if audited_rows % args.log_every < args.batch_size or audited_rows == len(shard):
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "completed": audited_rows,
                        "rows": len(shard),
                        "invalid_rows": len(invalid_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if audited_rows != len(shard):
        raise RuntimeError(f"shard row conservation failed: {audited_rows} != {len(shard)}")
    report = {
        "schema_version": 1,
        "status": "PASS" if not invalid_rows else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_manifest": str(manifest),
        "sampling_manifest_sha256": sha256_file(manifest),
        "selected_rows": len(dataset),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "audited_rows": audited_rows,
        "decoded_images": decoded_images,
        "image_decode_enabled": not args.skip_image_decode,
        "validated_multimodal_rows": validated_multimodal_rows,
        "invalid_rows": invalid_rows,
        "duration_seconds": time.monotonic() - started,
    }
    part = args.output_dir.resolve() / f"part-{args.shard_id:05d}.json"
    atomic_json(part, report)
    print(json.dumps({"image_audit_part": report, "report": str(part)}, ensure_ascii=False), flush=True)


def merge_parts(args: argparse.Namespace) -> None:
    manifest = args.sampling_manifest.resolve(strict=True)
    manifest_sha = sha256_file(manifest)
    output_dir = args.output_dir.resolve(strict=True)
    reports = []
    for shard_id in range(args.num_shards):
        path = output_dir / f"part-{shard_id:05d}.json"
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    selected_rows = {int(report["selected_rows"]) for report in reports}
    if len(selected_rows) != 1:
        raise RuntimeError("image audit part selected-row drift")
    if any(report["sampling_manifest_sha256"] != manifest_sha for report in reports):
        raise RuntimeError("image audit part sampling SHA drift")
    decode_modes = {bool(report.get("image_decode_enabled", True)) for report in reports}
    if len(decode_modes) != 1:
        raise RuntimeError("image audit part decode-mode drift")
    if [int(report["shard_id"]) for report in reports] != list(range(args.num_shards)):
        raise RuntimeError("image audit shard coverage drift")
    audited_rows = sum(int(report["audited_rows"]) for report in reports)
    selected = selected_rows.pop()
    if audited_rows != selected:
        raise RuntimeError(f"full image audit row conservation failed: {audited_rows} != {selected}")
    invalid_rows = sorted(
        (row for report in reports for row in report["invalid_rows"]),
        key=lambda row: int(row["position"]),
    )
    final = {
        "schema_version": 1,
        "status": "PASS" if not invalid_rows else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_manifest": str(manifest),
        "sampling_manifest_sha256": manifest_sha,
        "selected_rows": selected,
        "audited_rows": audited_rows,
        "decoded_images": sum(int(report["decoded_images"]) for report in reports),
        "image_decode_enabled": decode_modes.pop(),
        "validated_multimodal_rows": sum(
            int(report["validated_multimodal_rows"]) for report in reports
        ),
        "num_shards": args.num_shards,
        "part_reports": [str(output_dir / f"part-{index:05d}.json") for index in range(args.num_shards)],
        "invalid_rows": invalid_rows,
    }
    path = output_dir / "report.json"
    atomic_json(path, final)
    print(json.dumps({"image_integrity_audit": final, "report": str(path)}, ensure_ascii=False), flush=True)
    if invalid_rows:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--skip-image-decode", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_parts(args)
    else:
        if args.shard_id is None:
            raise SystemExit("--shard-id is required unless --merge is set")
        run_part(args)


if __name__ == "__main__":
    main()
