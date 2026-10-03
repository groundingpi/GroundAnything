#!/usr/bin/env python3
"""Build standalone fixed-microbatch packing orders without touching VLM caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path

import numpy as np

from train.dlm.data import IndexedCacheDataset


BS2_ALGORITHM = "dlm-bs2-long-short-global-rank-bucket-pcg64dxsm-v2"
FIXED_PACK_ALGORITHM = "dlm-fixed-pack-quantile-zigzag-global-rank-bucket-pcg64dxsm-v3"
BALANCED_PACK_ALGORITHM = "dlm-fixed-pack-quantile-balanced-global-rank-bucket-pcg64dxsm-v4"


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
        np.save(handle, values.astype(np.int32, copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--pack-size", type=int, default=2)
    parser.add_argument(
        "--packing-strategy",
        choices=("zigzag", "balanced"),
        default="zigzag",
        help="Fixed-pack construction for pressure and formal manifests; legacy routes default to zigzag.",
    )
    parser.add_argument("--selected-lengths", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.world_size < 1:
        raise ValueError("world-size must be positive")
    # Large values are useful for explicit capacity probes.  This only builds
    # an index ordering sidecar; admission is still decided by the full-topology
    # GPU pressure test before any value is used for formal training.
    if not 2 <= args.pack_size <= 64:
        raise ValueError("pack-size must be in [2, 64]")
    sampling_manifest = args.sampling_manifest.resolve(strict=True)
    ordering_lengths_path: Path | None = None
    ordering_lengths_sha256: str | None = None
    if args.selected_lengths is None:
        dataset = IndexedCacheDataset(sampling_manifest)
        lengths = dataset.cached_lengths()
    else:
        ordering_lengths_path = args.selected_lengths.resolve(strict=True)
        ordering_lengths_sha256 = sha256_file(ordering_lengths_path)
        lengths = np.load(ordering_lengths_path, mmap_mode="r", allow_pickle=False)
        sampling = json.loads(sampling_manifest.read_text(encoding="utf-8"))
        if len(lengths) != int(sampling["sampled_rows"]):
            raise RuntimeError("precomputed selected-length row count drift")
    if len(lengths) <= 0:
        raise RuntimeError("DLM packing requires a positive row count")

    order = np.argsort(lengths, kind="stable")
    tail_count = len(order) % args.pack_size
    if args.pack_size == 2:
        # BS2 keeps the long/short pairing contract.  A sampling manifest may
        # legitimately contain one odd row (the 499,939-row General pool is
        # such a case); retain it as an explicit tail row instead of silently
        # dropping it or failing after an otherwise deterministic build.
        if tail_count not in (0, 1):
            raise RuntimeError("DLM BS2 packing supports at most one tail row")
        if tail_count:
            middle = len(order) // 2
            tail_rows = order[middle : middle + tail_count].copy()
            order = np.concatenate((order[:middle], order[middle + tail_count :]))
        else:
            tail_rows = np.empty(0, dtype=order.dtype)
        half = len(order) // 2
        packs = np.stack((order[:half], order[: half - 1 : -1]), axis=1)
        algorithm = BS2_ALGORITHM
    else:
        # Each pack receives one sample from every length quantile.  Alternating
        # quantile direction makes the sums substantially flatter than random
        # grouping while retaining a deterministic, index-only data sidecar.
        if tail_count:
            middle = len(order) // 2 - tail_count // 2
            tail_rows = order[middle : middle + tail_count].copy()
            order = np.concatenate((order[:middle], order[middle + tail_count :]))
        else:
            tail_rows = np.empty(0, dtype=order.dtype)
        bands = order.reshape(args.pack_size, -1).copy()
        if args.packing_strategy == "zigzag":
            bands[1::2] = bands[1::2, ::-1]
            packs = bands.transpose(1, 0).copy()
            algorithm = FIXED_PACK_ALGORITHM
        else:
            # Seed one pack per longest row, then place every remaining row in
            # the currently lightest non-full pack.  Cardinality stays fixed,
            # hence optimizer/global-batch semantics are unchanged; unlike a
            # coarse quantile zigzag this prevents multiple tail rows from
            # coinciding in the same optimizer step.
            pack_count = len(order) // args.pack_size
            descending = order[::-1]
            packs = np.empty((pack_count, args.pack_size), dtype=order.dtype)
            packs[:, 0] = descending[:pack_count]
            heap = [
                (int(lengths[row]), pack_index, 1)
                for pack_index, row in enumerate(packs[:, 0])
            ]
            heapq.heapify(heap)
            for row in descending[pack_count:]:
                load, pack_index, column = heapq.heappop(heap)
                packs[pack_index, column] = row
                column += 1
                if column < args.pack_size:
                    heapq.heappush(
                        heap,
                        (load + int(lengths[row]), pack_index, column),
                    )
            if heap:
                raise RuntimeError("balanced fixed-pack heap did not drain")
            algorithm = BALANCED_PACK_ALGORITHM
    pack_sums = lengths[packs].sum(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_orders = []
    rank_bucket_audit: dict[str, int] = {}
    pack_length_order = np.argsort(pack_sums, kind="stable")
    full_groups = len(packs) // args.world_size
    remainder_count = len(packs) - full_groups * args.world_size
    for epoch in range(args.epochs):
        rng = np.random.Generator(np.random.PCG64DXSM(args.seed + epoch))
        # Rotate the non-divisible tail across epochs instead of permanently
        # dropping the same longest packs under distributed drop_last.
        if remainder_count:
            remainder = np.sort(
                rng.choice(len(packs), size=remainder_count, replace=False)
            )
            keep = np.ones(len(packs), dtype=bool)
            keep[remainder] = False
            grouped_order = pack_length_order[keep[pack_length_order]]
        else:
            remainder = np.empty(0, dtype=np.int64)
            grouped_order = pack_length_order
        grouped = grouped_order.reshape(full_groups, args.world_size)
        # Keep rank-local packs in each distributed microstep close in token
        # count. Shuffle global steps and rank assignment; never mix samples.
        epoch_grouped = grouped[rng.permutation(full_groups)].copy()
        within_group_keys = rng.random(epoch_grouped.shape)
        within_group_order = np.argsort(within_group_keys, axis=1, kind="stable")
        epoch_grouped = np.take_along_axis(epoch_grouped, within_group_order, axis=1)
        pack_order = epoch_grouped.reshape(-1)
        if len(remainder):
            pack_order = np.concatenate((pack_order, rng.permutation(remainder)))
        epoch_packs = packs[pack_order].copy()
        within_pack_keys = rng.random(epoch_packs.shape)
        within_pack_order = np.argsort(within_pack_keys, axis=1, kind="stable")
        epoch_packs = np.take_along_axis(epoch_packs, within_pack_order, axis=1)
        if epoch == 0 and full_groups:
            grouped_sums = lengths[
                epoch_packs[: full_groups * args.world_size]
            ].sum(axis=1).reshape(full_groups, args.world_size)
            spreads = grouped_sums.max(axis=1) - grouped_sums.min(axis=1)
            rank_bucket_audit = {
                "epoch0_within_step_spread_p50": int(np.percentile(spreads, 50)),
                "epoch0_within_step_spread_p95": int(np.percentile(spreads, 95)),
                "epoch0_within_step_spread_p99": int(np.percentile(spreads, 99)),
                "epoch0_within_step_spread_max": int(spreads.max()),
            }
        values = np.concatenate((epoch_packs.reshape(-1), rng.permutation(tail_rows)))
        if len(values) != len(lengths) or len(np.unique(values)) != len(values):
            raise RuntimeError(f"packing epoch {epoch} lost or duplicated rows")
        path = args.output_dir / f"epoch-{epoch:03d}.npy"
        epoch_orders.append({"epoch": epoch, "path": path.name, "sha256": save_npy(path, values)})

    percentile = lambda value: int(np.percentile(pack_sums, value))
    pack_sum_audit = {
        "individual_min": int(lengths.min()),
        "individual_max": int(lengths.max()),
        "pack_sum_min": int(pack_sums.min()),
        "pack_sum_p50": percentile(50),
        "pack_sum_p95": percentile(95),
        "pack_sum_p99": percentile(99),
        "pack_sum_max": int(pack_sums.max()),
    }
    if args.pack_size == 2:
        pack_sum_audit.update(
            {
                "pair_sum_min": pack_sum_audit["pack_sum_min"],
                "pair_sum_p50": pack_sum_audit["pack_sum_p50"],
                "pair_sum_p95": pack_sum_audit["pack_sum_p95"],
                "pair_sum_p99": pack_sum_audit["pack_sum_p99"],
                "pair_sum_max": pack_sum_audit["pack_sum_max"],
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "materialization": "DLM-only index packing; VLM jsonl and Arrow caches remain immutable",
        "sampling_manifest": str(sampling_manifest),
        "sampling_manifest_sha256": sha256_file(sampling_manifest),
        "ordering_lengths": str(ordering_lengths_path) if ordering_lengths_path else None,
        "ordering_lengths_sha256": ordering_lengths_sha256,
        "rows": len(lengths),
        "packs": len(packs),
        "pack_size": args.pack_size,
        "packing_strategy": args.packing_strategy if args.pack_size > 2 else "long-short",
        "seed": args.seed,
        "world_size": args.world_size,
        "global_rank_bucketing": {
            "full_global_microbatches": full_groups,
            "tail_packs": int(remainder_count),
            "tail_rows": int(len(tail_rows)),
            "contract": "each full consecutive world_size packs has neighboring pack-sums",
            "tail_contract": "non-divisible packs are deterministically rotated across epochs",
            **rank_bucket_audit,
        },
        "epoch_orders": epoch_orders,
        "pack_sum_audit": pack_sum_audit,
    }
    if args.pack_size == 2:
        manifest["pairs"] = len(packs)
        manifest["pair_sum_audit"] = pack_sum_audit
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", **pack_sum_audit, "rows": len(lengths)}), flush=True)


if __name__ == "__main__":
    main()
