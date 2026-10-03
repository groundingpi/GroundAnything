#!/usr/bin/env python3
"""Build immutable index-only sampling manifests for DLM Direct Conversion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import yaml


ALGORITHM = "numpy-pcg64dxsm-global-without-replacement-v1"
FILTERED_ALGORITHM = "numpy-pcg64dxsm-global-eligible-without-replacement-v1"


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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npy(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.int64, copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML root must be a mapping: {path}")
    return value


def load_sources(config: Path, exclude_source: set[str]) -> list[dict[str, Any]]:
    value = load_yaml(config)
    datasets = value.get("datasets")
    require(isinstance(datasets, list) and datasets, f"empty datasets: {config}")
    sources: list[dict[str, Any]] = []
    offset = 0
    for entry in datasets:
        source_id = str(entry["source_id"])
        if source_id in exclude_source:
            continue
        cache_path = Path(str(entry["path"])).resolve(strict=True)
        manifest_path = Path(str(entry["manifest"])).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("num_rows")
        require(type(rows) is int and rows > 0, f"invalid num_rows: {manifest_path}")
        require(manifest.get("source_id") == source_id, f"source_id drift: {manifest_path}")
        require(manifest.get("cache_path") == str(cache_path), f"cache_path drift: {manifest_path}")
        state_path = cache_path / "state.json"
        require(state_path.is_file(), f"missing state.json: {cache_path}")
        sources.append(
            {
                "position": len(sources),
                "name": str(entry["name"]),
                "source_id": source_id,
                "cache_path": str(cache_path),
                "cache_manifest": str(manifest_path),
                "cache_manifest_sha256": sha256_file(manifest_path),
                "state_json_sha256": sha256_file(state_path),
                "rows": rows,
                "global_start": offset,
                "global_stop": offset + rows,
            }
        )
        offset += rows
    require(sources, "all sources were excluded")
    return sources


def eligible_local_indices(source: dict[str, Any], max_cached_length: int) -> tuple[int, np.ndarray]:
    cache_path = Path(source["cache_path"])
    state = json.loads((cache_path / "state.json").read_text(encoding="utf-8"))
    eligible_chunks: list[np.ndarray] = []
    row_offset = 0
    for entry in state["_data_files"]:
        arrow_path = cache_path / entry["filename"]
        with pa.memory_map(str(arrow_path), "r") as mapped:
            reader = ipc.open_stream(mapped)
            for batch in reader:
                column_index = batch.schema.get_field_index("length")
                require(column_index >= 0, f"missing length column: {arrow_path}")
                lengths = np.asarray(
                    batch.column(column_index).to_numpy(zero_copy_only=False),
                    dtype=np.int64,
                )
                local = np.flatnonzero(lengths <= max_cached_length).astype(np.int64)
                if len(local):
                    eligible_chunks.append(local + row_offset)
                row_offset += batch.num_rows
    require(row_offset == int(source["rows"]), f"source row drift: {source['source_id']}")
    eligible = (
        np.concatenate(eligible_chunks)
        if eligible_chunks
        else np.empty(0, dtype=np.int64)
    )
    return int(source["position"]), eligible


def filtered_population(
    sources: list[dict[str, Any]],
    max_cached_length: int,
    workers: int,
) -> np.ndarray:
    results: list[np.ndarray | None] = [None] * len(sources)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(eligible_local_indices, source, max_cached_length): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), 1):
            position, local = future.result()
            source = futures[future]
            source["eligible_rows"] = int(len(local))
            source["filtered_rows"] = int(source["rows"] - len(local))
            results[position] = local + int(source["global_start"])
            print(
                json.dumps(
                    {
                        "completed_sources": completed,
                        "total_sources": len(sources),
                        "source_id": source["source_id"],
                        "rows": source["rows"],
                        "eligible_rows": len(local),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    require(not any(value is None for value in results), "eligible population scan incomplete")
    return np.concatenate(results)


def build(
    config: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    stage: str,
    exclude_source: set[str],
    max_cached_length: int | None = None,
    workers: int = 12,
) -> dict[str, Any]:
    sources = load_sources(config, exclude_source)
    raw_population = sum(source["rows"] for source in sources)
    eligible_global: np.ndarray | None = None
    if max_cached_length is not None:
        require(max_cached_length > 0, "max_cached_length must be positive")
        eligible_global = filtered_population(sources, max_cached_length, workers)
        population = len(eligible_global)
    else:
        population = raw_population
    require(0 < sample_size <= population, "sample_size is outside population")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    sampled_positions = rng.choice(population, size=sample_size, replace=False)
    indices = (
        sampled_positions
        if eligible_global is None
        else eligible_global[sampled_positions]
    )
    indices.sort()
    require(len(indices) == 1 or bool(np.all(indices[1:] > indices[:-1])), "duplicate sample indices")

    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "global_indices.npy"
    global_sha256 = save_npy(global_path, indices)
    selected_total = 0
    for source in sources:
        left = int(np.searchsorted(indices, source["global_start"], side="left"))
        right = int(np.searchsorted(indices, source["global_stop"], side="left"))
        local = indices[left:right] - source["global_start"]
        local_path = output_dir / f"source-{source['position']:03d}.npy"
        source["sampled_rows"] = int(len(local))
        sampling_denominator = int(source.get("eligible_rows", source["rows"]))
        source["sampling_ratio"] = len(local) / sampling_denominator
        source["local_indices"] = local_path.name
        source["local_indices_sha256"] = save_npy(local_path, local)
        selected_total += len(local)
    require(selected_total == sample_size, "per-source row conservation failed")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "algorithm": FILTERED_ALGORITHM if max_cached_length is not None else ALGORITHM,
        "seed": seed,
        "source_config": str(config.resolve(strict=True)),
        "source_config_sha256": sha256_file(config),
        "excluded_source_ids": sorted(exclude_source),
        "population_rows": population,
        "raw_population_rows": raw_population,
        "max_cached_length": max_cached_length,
        "sampled_rows": sample_size,
        "global_indices": global_path.name,
        "global_indices_sha256": global_sha256,
        "materialization": "index-only; source jsonl and Arrow caches are immutable",
        "sources": sources,
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--stage", choices=("general", "specialist"), required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--max-cached-length", type=int)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.config,
        args.output_dir,
        args.sample_size,
        args.seed,
        args.stage,
        set(args.exclude_source),
        args.max_cached_length,
        args.workers,
    )
    print(
        json.dumps(
            {
                "stage": manifest["stage"],
                "population_rows": manifest["population_rows"],
                "sampled_rows": manifest["sampled_rows"],
                "sources": len(manifest["sources"]),
                "global_indices_sha256": manifest["global_indices_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
