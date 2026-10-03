#!/usr/bin/env python3
"""Exact supervised-response token audit for a DLM sampling manifest.

This is a read-only sidecar audit.  It mirrors the GroundAnything/Qwen3 assistant
normalization and label contract, but deliberately avoids image decoding and
full prompt tokenization because only supervised assistant tokens determine
how many B32 response blocks are trained.  Work is split into deterministic
contiguous global ranges so multiple workers can scan one manifest without
overlap.  Rank zero merges the parts into one row-aligned NumPy sidecar.
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

import numpy as np


HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
BUCKET_NAMES = ("1-32", "33-64", "65-128", "129-256", "257-512", "513-1024", ">=1025")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def normalize_assistant_content(content: str) -> str:
    stripped = str(content).strip()
    if "</think>" in stripped and "<think>" in stripped:
        before, _, after = stripped.partition("</think>")
        reasoning = before.rstrip("\n").rsplit("<think>", 1)[-1].lstrip("\n").strip()
        rest = after.lstrip("\n")
        stripped = f"<think>\n{reasoning}\n</think>\n\n{rest}"
    if not stripped.startswith(("<think>", NON_THINKING_PREFIX)):
        stripped = NON_THINKING_PREFIX + stripped
    return stripped.replace("<image>", IMAGE_TOKENS)


def bucket_counts(values: np.ndarray) -> dict[str, int]:
    edges = ((1, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, None))
    result: dict[str, int] = {}
    for name, (low, high) in zip(BUCKET_NAMES, edges, strict=True):
        mask = values >= low
        if high is not None:
            mask &= values <= high
        result[name] = int(mask.sum())
    return result


def summarize(values: np.ndarray) -> dict[str, Any]:
    require(values.ndim == 1 and len(values) > 0, "response-length part is empty")
    require(bool(np.all(values > 0)), "sample contains no supervised assistant tokens")
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "min": int(values.min()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": int(values.max()),
        "over_32_rows": int((values > 32).sum()),
        "over_32_ratio": float((values > 32).mean()),
        "over_64_rows": int((values > 64).sum()),
        "over_64_ratio": float((values > 64).mean()),
        "over_128_rows": int((values > 128).sum()),
        "over_128_ratio": float((values > 128).mean()),
        "buckets": bucket_counts(values),
    }


def response_lengths(tokenizer: Any, messages_batch: list[list[dict[str, Any]]]) -> np.ndarray:
    header_length = len(tokenizer.encode(HEADER, add_special_tokens=False))
    ignored_prefix_length = len(tokenizer.encode(NON_THINKING_PREFIX, add_special_tokens=False))
    texts: list[str] = []
    owners: list[tuple[int, bool]] = []
    for row_index, messages in enumerate(messages_batch):
        for message in messages:
            if message.get("role") != "assistant" or message.get("loss") == 0:
                continue
            content = normalize_assistant_content(str(message.get("content", "")))
            texts.append(HEADER + content + IM_END)
            owners.append((row_index, content.startswith(NON_THINKING_PREFIX)))
    totals = np.zeros(len(messages_batch), dtype=np.int32)
    if texts:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        for length, (row_index, has_ignored_prefix) in zip(encoded["length"], owners, strict=True):
            supervised = int(length) - header_length
            if has_ignored_prefix:
                supervised -= ignored_prefix_length
            require(supervised > 0, f"non-positive supervised response length at local row {row_index}")
            totals[row_index] += supervised
    require(bool(np.all(totals > 0)), "one or more rows contain no supervised assistant tokens")
    return totals


def partition(total: int, rank: int, world_size: int) -> tuple[int, int]:
    return total * rank // world_size, total * (rank + 1) // world_size


def run_part(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    manifest_path = args.sampling_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = int(manifest["sampled_rows"])
    start, stop = partition(total, args.rank, args.world_size)
    require(stop > start, f"empty rank partition: rank={args.rank} world={args.world_size}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    output = np.empty(stop - start, dtype=np.int32)
    source_summaries: list[dict[str, Any]] = []
    cursor = 0
    global_offset = 0
    manifest_root = manifest_path.parent
    started = time.time()
    for source in manifest["sources"]:
        source_rows = int(source["sampled_rows"])
        source_start, source_stop = global_offset, global_offset + source_rows
        global_offset = source_stop
        overlap_start = max(start, source_start)
        overlap_stop = min(stop, source_stop)
        if overlap_stop <= overlap_start:
            continue
        selected = np.load(manifest_root / source["local_indices"], mmap_mode="r", allow_pickle=False)
        local_start = overlap_start - source_start
        local_stop = overlap_stop - source_start
        physical = np.asarray(selected[local_start:local_stop], dtype=np.int64)
        dataset = load_from_disk(source["cache_path"], keep_in_memory=False)
        part_values = np.empty(len(physical), dtype=np.int32)
        for batch_start in range(0, len(physical), args.batch_rows):
            batch_stop = min(batch_start + args.batch_rows, len(physical))
            rows = dataset[physical[batch_start:batch_stop].tolist()]
            part_values[batch_start:batch_stop] = response_lengths(tokenizer, rows["messages"])
        output[cursor : cursor + len(part_values)] = part_values
        cursor += len(part_values)
        source_summaries.append(
            {
                "source_id": source["source_id"],
                "route": source.get("route", "general"),
                "global_start": overlap_start,
                "global_stop": overlap_stop,
                "summary": summarize(part_values),
            }
        )
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "source_id": source["source_id"],
                    "completed_rows": cursor,
                    "rank_rows": stop - start,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    require(global_offset == total, "manifest source row conservation failed")
    require(cursor == stop - start, "rank response-length row conservation failed")
    part_path = args.output_dir / f"part-{args.rank:03d}.npy"
    part_sha = atomic_npy(part_path, output)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "rank": args.rank,
        "world_size": args.world_size,
        "global_start": start,
        "global_stop": stop,
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_path": str(args.tokenizer_path.resolve(strict=True)),
        "part": str(part_path),
        "part_sha256": part_sha,
        "summary": summarize(output),
        "sources": source_summaries,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / f"receipt-{args.rank:03d}.json", receipt)
    return receipt


def merge(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.rank != 0:
        return None
    deadline = time.monotonic() + args.merge_timeout
    receipt_paths = [args.output_dir / f"receipt-{rank:03d}.json" for rank in range(args.world_size)]
    while not all(path.is_file() and path.stat().st_size > 0 for path in receipt_paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in receipt_paths if not path.is_file() or path.stat().st_size == 0]
            raise TimeoutError(f"response-length receipts timed out: {missing}")
        time.sleep(5)
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
    manifest_path = args.sampling_manifest.resolve(strict=True)
    manifest_sha = sha256_file(manifest_path)
    expected_start = 0
    total = int(json.loads(manifest_path.read_text(encoding="utf-8"))["sampled_rows"])
    for rank, receipt in enumerate(receipts):
        require(receipt["status"] == "PASS" and int(receipt["rank"]) == rank, f"invalid receipt rank {rank}")
        require(receipt["sampling_manifest_sha256"] == manifest_sha, "sampling manifest SHA drift across ranks")
        require(int(receipt["global_start"]) == expected_start, "rank ranges are not contiguous")
        expected_start = int(receipt["global_stop"])
        require(sha256_file(Path(receipt["part"])) == receipt["part_sha256"], f"part SHA drift: rank {rank}")
    require(expected_start == total, "merged response-length row conservation failed")
    final_path = args.output_dir / "response_tokens.npy"
    temporary = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    merged = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(total,))
    offset = 0
    for receipt in receipts:
        values = np.load(receipt["part"], mmap_mode="r", allow_pickle=False)
        merged[offset : offset + len(values)] = values
        offset += len(values)
    merged.flush()
    del merged
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, final_path)
    values = np.load(final_path, mmap_mode="r", allow_pickle=False)
    report = {
        "schema_version": 1,
        "status": "PASS" if float((values > 32).mean()) >= args.minimum_over_32_ratio else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "sum of all enabled assistant body tokens plus each supervised im_end; injected empty-thinking prefix excluded",
        "block_size": 32,
        "minimum_over_32_ratio": args.minimum_over_32_ratio,
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": manifest_sha,
        "tokenizer_path": str(args.tokenizer_path.resolve(strict=True)),
        "response_tokens": str(final_path),
        "response_tokens_sha256": sha256_file(final_path),
        "summary": summarize(values),
        "rank_receipts": [str(path) for path in receipt_paths],
    }
    atomic_json(args.output_dir / "report.json", report)
    require(report["status"] == "PASS", f"multi-block ratio gate failed: {report['summary']['over_32_ratio']:.6f}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--minimum-over-32-ratio", type=float, default=0.4)
    parser.add_argument("--merge-timeout", type=int, default=21600)
    args = parser.parse_args()
    require(args.world_size > 0 and 0 <= args.rank < args.world_size, "invalid rank/world size")
    require(args.batch_rows > 0, "batch rows must be positive")
    require(0.0 <= args.minimum_over_32_ratio <= 1.0, "invalid ratio gate")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = run_part(args)
    report = merge(args)
    print(json.dumps({"receipt": receipt, "report": report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
