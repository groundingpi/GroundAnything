#!/usr/bin/env python3
"""Extract selected Arrow length columns in parallel for DLM packing."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(args: tuple[dict, Path]) -> tuple[int, np.ndarray, int]:
    source, manifest_dir = args
    cache_path = Path(source["cache_path"])
    state = json.loads((cache_path / "state.json").read_text(encoding="utf-8"))
    chunks: list[np.ndarray] = []
    rows = 0
    for entry in state["_data_files"]:
        arrow_path = cache_path / entry["filename"]
        with pa.memory_map(str(arrow_path), "r") as mapped:
            reader = ipc.open_stream(mapped)
            for batch in reader:
                column_index = batch.schema.get_field_index("length")
                if column_index < 0:
                    raise RuntimeError(f"missing length column: {arrow_path}")
                values = batch.column(column_index).to_numpy(zero_copy_only=False)
                chunks.append(np.asarray(values, dtype=np.int32))
                rows += batch.num_rows
    if rows != int(source["rows"]):
        raise RuntimeError(f"source row drift: {source['source_id']}: {rows} != {source['rows']}")
    full_lengths = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    selected = np.load(
        manifest_dir / source["local_indices"],
        mmap_mode="r",
        allow_pickle=False,
    )
    selected_lengths = full_lengths[np.asarray(selected, dtype=np.int64)]
    if len(selected_lengths) != int(source["sampled_rows"]):
        raise RuntimeError(f"selected length drift: {source['source_id']}")
    return int(source["position"]), selected_lengths, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    sampling_manifest = args.sampling_manifest.resolve(strict=True)
    manifest = json.loads(sampling_manifest.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    results: list[np.ndarray | None] = [None] * len(sources)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(read_source, (source, sampling_manifest.parent)): source["source_id"]
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), 1):
            position, values, rows = future.result()
            results[position] = values
            print(
                json.dumps(
                    {
                        "completed_sources": completed,
                        "total_sources": len(sources),
                        "source_id": futures[future],
                        "rows": rows,
                        "selected_rows": len(values),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if any(values is None for values in results):
        raise RuntimeError("selected length extraction incomplete")
    selected_lengths = np.concatenate(results).astype(np.int32, copy=False)
    if len(selected_lengths) != int(manifest["sampled_rows"]):
        raise RuntimeError("selected length row conservation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, selected_lengths, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(selected_lengths),
                "min": int(selected_lengths.min()),
                "max": int(selected_lengths.max()),
                "sha256": sha256_file(args.output),
                "sampling_manifest_sha256": sha256_file(sampling_manifest),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
