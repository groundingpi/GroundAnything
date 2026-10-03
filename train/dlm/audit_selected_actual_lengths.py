#!/usr/bin/env python3
"""Fail-closed audit of real processor lengths for selected DLM tail rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from train.dlm.data import DLMDataCollator, IndexedCacheDataset
from train.dlm.train_dlm import configure_processor


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


class ActualLengthCollator:
    def __init__(self, processor: Any, model_family: str) -> None:
        # Use generous guards here so the audit can measure, rather than reject,
        # rows that exceed the formal 4096/8192 training limits.
        self.collator = DLMDataCollator(
            processor,
            max_length=1 << 30,
            max_joint_length=1 << 30,
            model_family=model_family,
        )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if len(rows) != 1:
            raise RuntimeError("actual-length audit requires batch_size=1")
        row = rows[0]
        encoded = self.collator._encode(row)
        input_ids = encoded["input_ids"]
        vision_ids = {
            int(self.collator.tokenizer.convert_tokens_to_ids("<|vision_start|>")),
            int(self.collator.tokenizer.convert_tokens_to_ids("<|image_pad|>")),
            int(self.collator.tokenizer.convert_tokens_to_ids("<|video_pad|>")),
        }
        noisy = sum(int(token) not in vision_ids for token in input_ids.tolist())
        return {
            "id": str(row.get("id")),
            "actual_length": int(input_ids.numel()),
            "actual_joint_length": int(input_ids.numel()) + noisy,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--selected-lengths", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("qwen3_5", "groundinganything_qwen3"),
        default="qwen3_5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cached-length", type=int, default=3000)
    parser.add_argument(
        "--include-multi-image",
        action="store_true",
        help="also audit every selected row containing more than one image",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-joint-length", type=int, default=8192)
    parser.add_argument("--image-max-token-num", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    sampling_manifest = args.sampling_manifest.resolve(strict=True)
    selected_lengths_path = args.selected_lengths.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    lengths = np.load(selected_lengths_path, mmap_mode="r", allow_pickle=False)
    dataset = IndexedCacheDataset(sampling_manifest)
    if len(lengths) != len(dataset):
        raise RuntimeError("selected-length row count drift")
    candidate_mask = np.asarray(lengths >= args.min_cached_length, dtype=bool)
    multi_image_rows = 0
    if args.include_multi_image:
        image_counts: list[np.ndarray] = []
        for source in dataset.sources:
            selected = np.asarray(source["indices"], dtype=np.int64)
            counts = pc.fill_null(
                pc.list_value_length(source["dataset"].data.column("images")),
                0,
            ).take(pa.array(selected))
            image_counts.append(
                np.asarray(counts.to_numpy(zero_copy_only=False), dtype=np.int32)
            )
        selected_image_counts = np.concatenate(image_counts)
        if len(selected_image_counts) != len(dataset):
            raise RuntimeError("selected image-count row drift")
        multi_image = selected_image_counts > 1
        multi_image_rows = int(multi_image.sum())
        candidate_mask |= multi_image
    candidates = np.flatnonzero(candidate_mask).astype(np.int64)
    if not len(candidates):
        raise RuntimeError("actual-length audit selected no tail rows")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.model_family == "groundinganything_qwen3",
    )
    configure_processor(processor, args.image_max_token_num, args.model_family)
    loader = DataLoader(
        Subset(dataset, candidates.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=ActualLengthCollator(processor, args.model_family),
    )
    actual = np.empty(len(candidates), dtype=np.int32)
    joint = np.empty(len(candidates), dtype=np.int32)
    ids: list[str] = []
    for offset, result in enumerate(loader):
        actual[offset] = int(result["actual_length"])
        joint[offset] = int(result["actual_joint_length"])
        ids.append(result["id"])
        if (offset + 1) % 250 == 0 or offset + 1 == len(candidates):
            print(
                json.dumps(
                    {"completed": offset + 1, "candidates": len(candidates)},
                    sort_keys=True,
                ),
                flush=True,
            )

    cached = np.asarray(lengths[candidates], dtype=np.int32)
    invalid_mask = (actual > args.max_length) | (joint > args.max_joint_length)
    invalid_positions = candidates[invalid_mask]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "actual_lengths.npz"
    temporary = arrays_path.with_name(f".{arrays_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        positions=candidates,
        cached_lengths=cached,
        actual_lengths=actual,
        actual_joint_lengths=joint,
        invalid_positions=invalid_positions,
    )
    os.replace(temporary, arrays_path)
    invalid_rows = [
        {
            "position": int(candidates[index]),
            "id": ids[index],
            "cached_length": int(cached[index]),
            "actual_length": int(actual[index]),
            "actual_joint_length": int(joint[index]),
        }
        for index in np.flatnonzero(invalid_mask)
    ]
    report = {
        "status": "PASS" if not invalid_rows else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_manifest": str(sampling_manifest),
        "sampling_manifest_sha256": sha256_file(sampling_manifest),
        "selected_lengths": str(selected_lengths_path),
        "selected_lengths_sha256": sha256_file(selected_lengths_path),
        "model_path": str(model_path),
        "model_family": args.model_family,
        "model_chat_template_sha256": sha256_file(model_path / "chat_template.jinja"),
        "model_tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "min_cached_length": args.min_cached_length,
        "candidate_rule": (
            f"cached_length>={args.min_cached_length} OR image_count>1"
            if args.include_multi_image
            else f"cached_length>={args.min_cached_length}"
        ),
        "include_multi_image": bool(args.include_multi_image),
        "selected_multi_image_rows": multi_image_rows,
        "max_length": args.max_length,
        "max_joint_length": args.max_joint_length,
        "selected_rows": len(dataset),
        "audited_rows": len(candidates),
        "cached_min": int(cached.min()),
        "cached_max": int(cached.max()),
        "actual_min": int(actual.min()),
        "actual_max": int(actual.max()),
        "actual_joint_max": int(joint.max()),
        "delta_min": int((actual - cached).min()),
        "delta_max": int((actual - cached).max()),
        "invalid_rows": invalid_rows,
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if invalid_rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
