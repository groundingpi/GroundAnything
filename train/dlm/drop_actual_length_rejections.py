#!/usr/bin/env python3
"""Drop real-length rejects and materialize Qwen3 DLM workload lengths.

The immutable Arrow/jsonl caches are never modified.  This rewrites only the
index sidecar selected for DLM, derives new fail-closed image/length audits,
and builds a packing proxy matching the actual clean + two noisy streams.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


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


def save_npz(path: Path, **values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--selected-lengths", type=Path, required=True)
    parser.add_argument("--actual-length-audit-report", type=Path, required=True)
    parser.add_argument("--image-audit-report", type=Path)
    parser.add_argument("--completed-source-checkpoint", type=Path)
    parser.add_argument("--source-recipe", type=Path)
    parser.add_argument("--source-recipe-sha256")
    parser.add_argument("--output-sampling-dir", type=Path, required=True)
    parser.add_argument("--output-clean-lengths", type=Path, required=True)
    parser.add_argument("--output-workload-lengths", type=Path, required=True)
    parser.add_argument("--output-actual-audit", type=Path, required=True)
    parser.add_argument("--output-image-audit", type=Path, required=True)
    args = parser.parse_args()

    sampling_path = args.sampling_manifest.resolve(strict=True)
    lengths_path = args.selected_lengths.resolve(strict=True)
    actual_report_path = args.actual_length_audit_report.resolve(strict=True)
    strict_image_audit = args.image_audit_report is not None
    if strict_image_audit:
        if any((args.completed_source_checkpoint, args.source_recipe, args.source_recipe_sha256)):
            raise RuntimeError("strict image audit and completed-source provenance are mutually exclusive")
        image_report_path = args.image_audit_report.resolve(strict=True)
        image_report = json.loads(image_report_path.read_text(encoding="utf-8"))
        source_epoch_provenance = None
    else:
        if not all((args.completed_source_checkpoint, args.source_recipe, args.source_recipe_sha256)):
            raise RuntimeError("completed-source provenance requires checkpoint, recipe and recipe SHA")
        checkpoint = args.completed_source_checkpoint.resolve(strict=True)
        source_recipe = args.source_recipe.resolve(strict=True)
        if sha256_file(source_recipe) != str(args.source_recipe_sha256):
            raise RuntimeError("completed-source recipe SHA drift")
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        global_step = int(trainer_state.get("global_step", -1))
        max_steps = int(trainer_state.get("max_steps", -1))
        epoch = float(trainer_state.get("epoch", -1.0))
        if global_step <= 0 or global_step != max_steps or epoch < 1.0:
            raise RuntimeError("completed-source checkpoint is not a full final epoch")
        if not (checkpoint / "model.safetensors.index.json").is_file():
            raise RuntimeError("completed-source checkpoint lacks model weights")
        source_epoch_provenance = {
            "method": "completed_source_epoch_provenance",
            "completed_checkpoint": str(checkpoint),
            "global_step": global_step,
            "max_steps": max_steps,
            "epoch": epoch,
            "source_recipe": str(source_recipe),
            "source_recipe_sha256": str(args.source_recipe_sha256),
        }
        image_report_path = None
        image_report = None
    manifest = json.loads(sampling_path.read_text(encoding="utf-8"))
    actual_report = json.loads(actual_report_path.read_text(encoding="utf-8"))
    old_rows = int(manifest["sampled_rows"])

    if actual_report.get("status") != "FAIL" or not actual_report.get("invalid_rows"):
        raise RuntimeError("drop requires a failed actual-length audit with explicit rejects")
    if actual_report.get("sampling_manifest_sha256") != sha256_file(sampling_path):
        raise RuntimeError("actual-length audit sampling SHA drift")
    if actual_report.get("selected_lengths_sha256") != sha256_file(lengths_path):
        raise RuntimeError("actual-length audit selected-length SHA drift")
    arrays_path = Path(actual_report["arrays"]).resolve(strict=True)
    if actual_report.get("arrays_sha256") != sha256_file(arrays_path):
        raise RuntimeError("actual-length audit array SHA drift")
    if strict_image_audit:
        if image_report.get("status") != "PASS" or image_report.get("invalid_rows"):
            raise RuntimeError("source sample lacks a passing full image audit")
        if image_report.get("image_decode_enabled") is not True:
            raise RuntimeError("source image audit did not decode images")
        if image_report.get("sampling_manifest_sha256") != sha256_file(sampling_path):
            raise RuntimeError("image audit sampling SHA drift")
        if int(image_report.get("audited_rows", -1)) != old_rows:
            raise RuntimeError("image audit is not full coverage")
    else:
        normalized_path = Path(manifest["source_config"]).resolve(strict=True)
        if manifest.get("source_config_sha256") != sha256_file(normalized_path):
            raise RuntimeError("normalized source recipe SHA drift")
        normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
        normalized_meta = normalized.get("meta", {})
        if Path(normalized_meta["source_recipe"]).resolve() != Path(source_epoch_provenance["source_recipe"]):
            raise RuntimeError("normalized recipe source path drift")
        if normalized_meta.get("source_recipe_sha256") != source_epoch_provenance["source_recipe_sha256"]:
            raise RuntimeError("normalized recipe source SHA drift")
        if int(normalized_meta.get("effective_rows", -1)) != old_rows:
            raise RuntimeError("normalized recipe effective-row drift")

    arrays = np.load(arrays_path, allow_pickle=False)
    audited_positions = np.asarray(arrays["positions"], dtype=np.int64)
    cached_audited = np.asarray(arrays["cached_lengths"], dtype=np.int32)
    actual = np.asarray(arrays["actual_lengths"], dtype=np.int32)
    joint = np.asarray(arrays["actual_joint_lengths"], dtype=np.int32)
    invalid_positions = np.asarray(arrays["invalid_positions"], dtype=np.int64)
    report_positions = np.asarray(
        sorted({int(row["position"]) for row in actual_report["invalid_rows"]}),
        dtype=np.int64,
    )
    if not (
        len(audited_positions)
        == len(cached_audited)
        == len(actual)
        == len(joint)
        == int(actual_report["audited_rows"])
    ):
        raise RuntimeError("actual-length audit array row drift")
    if not np.array_equal(np.sort(invalid_positions), report_positions):
        raise RuntimeError("actual-length reject report/array drift")
    if len(np.unique(audited_positions)) != len(audited_positions):
        raise RuntimeError("duplicated audited positions")
    if np.any(audited_positions < 0) or np.any(audited_positions >= old_rows):
        raise RuntimeError("audited position out of range")

    old_lengths = np.asarray(np.load(lengths_path, allow_pickle=False), dtype=np.int32)
    if len(old_lengths) != old_rows:
        raise RuntimeError("selected-length row count drift")
    if not np.array_equal(old_lengths[audited_positions], cached_audited):
        raise RuntimeError("cached lengths no longer match audited positions")

    keep = np.ones(old_rows, dtype=bool)
    keep[invalid_positions] = False
    new_rows = int(keep.sum())
    output_sampling_dir = args.output_sampling_dir.resolve()
    if output_sampling_dir.exists():
        raise RuntimeError(f"output sampling directory already exists: {output_sampling_dir}")
    output_sampling_dir.mkdir(parents=True)

    # Reindex the immutable sidecar source by source.
    source_start = 0
    deleted_details: list[dict] = []
    detail_by_position = {
        int(row["position"]): dict(row) for row in actual_report["invalid_rows"]
    }
    for source in manifest["sources"]:
        source_rows = int(source["sampled_rows"])
        source_stop = source_start + source_rows
        rejected = invalid_positions[
            (invalid_positions >= source_start) & (invalid_positions < source_stop)
        ]
        local_positions = rejected - source_start
        source_path = sampling_path.parent / source["local_indices"]
        source_indices = np.asarray(np.load(source_path, allow_pickle=False), dtype=np.int64)
        if len(source_indices) != source_rows:
            raise RuntimeError(f"source selected-index drift: {source['source_id']}")
        target_path = output_sampling_dir / source["local_indices"]
        if len(local_positions):
            source_keep = np.ones(source_rows, dtype=bool)
            source_keep[local_positions] = False
            source_indices = source_indices[source_keep]
            source["local_indices_sha256"] = save_npy(target_path, source_indices)
            source["sampled_rows"] = int(len(source_indices))
            if "sampled_unique_physical_rows" in source:
                unique, exposures = np.unique(source_indices, return_counts=True)
                source["sampled_unique_physical_rows"] = int(len(unique))
                source["sampled_physical_duplicates"] = int(len(source_indices) - len(unique))
                source["max_physical_exposure"] = int(exposures.max()) if len(exposures) else 0
            if "eligible_rows" in source and int(source["eligible_rows"]) > 0:
                source["sampling_ratio"] = len(source_indices) / int(source["eligible_rows"])
            elif "eligible_effective_rows" in source and int(source["eligible_effective_rows"]) > 0:
                source["sampling_ratio"] = len(source_indices) / int(source["eligible_effective_rows"])
            for old_position, selected_local in zip(rejected.tolist(), local_positions.tolist()):
                detail = detail_by_position[int(old_position)]
                detail.update(
                    {
                        "source_id": str(source["source_id"]),
                        "local_index": int(np.load(source_path, mmap_mode="r", allow_pickle=False)[selected_local]),
                    }
                )
                deleted_details.append(detail)
        else:
            os.link(source_path, target_path)
        source_start = source_stop
    if source_start != old_rows or len(deleted_details) != len(invalid_positions):
        raise RuntimeError("source deletion conservation failed")

    global_name = manifest.get("global_indices")
    if global_name:
        old_global_path = sampling_path.parent / global_name
        old_global = np.asarray(np.load(old_global_path, allow_pickle=False), dtype=np.int64)
        if len(old_global) != old_rows:
            raise RuntimeError("global-index row count drift")
        manifest["global_indices_sha256"] = save_npy(
            output_sampling_dir / global_name,
            old_global[keep],
        )

    manifest["sampled_rows"] = new_rows
    manifest["sampled_unique_physical_rows"] = sum(
        int(source.get("sampled_unique_physical_rows", source["sampled_rows"]))
        for source in manifest["sources"]
    )
    manifest["sampled_physical_duplicates"] = (
        new_rows - int(manifest["sampled_unique_physical_rows"])
    )
    manifest["unique_fraction"] = int(manifest["sampled_unique_physical_rows"]) / new_rows
    manifest["max_physical_exposure"] = max(
        int(source.get("max_physical_exposure", 1)) for source in manifest["sources"]
    )
    # Specialist manifests carry an explicit route on every source, while the
    # historical General manifest predates that field.  Treat a missing/empty
    # route as ``general`` instead of emitting a bogus ``None: 0`` bucket or
    # failing while summarizing deleted rows.  This changes provenance only;
    # the selected index arrays above are untouched.
    def source_route(source: dict) -> str:
        return str(source.get("route") or "general")

    manifest["route_selected_rows"] = {
        route: sum(
            int(source["sampled_rows"])
            for source in manifest["sources"]
            if source_route(source) == route
        )
        for route in sorted({source_route(source) for source in manifest["sources"]})
    }
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["algorithm"] = f"{manifest['algorithm']}+actual-length-rejection-deletion-v2"
    manifest["parent_sampling_manifest"] = str(sampling_path)
    manifest["parent_sampling_manifest_sha256"] = sha256_file(sampling_path)
    manifest["failed_actual_length_audit"] = str(actual_report_path)
    manifest["failed_actual_length_audit_sha256"] = sha256_file(actual_report_path)
    manifest["deleted_actual_length_rejections"] = sorted(
        deleted_details,
        key=lambda row: int(row["position"]),
    )
    manifest["deleted_actual_length_rejections_by_route"] = {
        route: sum(
            1
            for row in deleted_details
            if next(
                source_route(source)
                for source in manifest["sources"]
                if source["source_id"] == row["source_id"]
            ) == route
        )
        for route in manifest["route_selected_rows"]
    }
    output_manifest = output_sampling_dir / "manifest.json"
    atomic_json(output_manifest, manifest)

    # Cache length is a reasonable clean-length proxy outside the audited tail.
    # For every audited row use the exact Qwen3/K3 clean length.  DLM executes
    # one clean stream plus two complementary noisy text streams, so use
    # clean + 2*text as the packing workload proxy.
    clean_proxy = old_lengths.copy()
    clean_proxy[audited_positions] = actual
    # For rows outside the real-length audit, use the conservative upper
    # envelope observed in that audit rather than the old cache length alone.
    # This is for ordering only; it never changes training tokens or loss.
    observed_ratio = (
        actual.astype(np.float64)
        + 2 * (joint.astype(np.float64) - actual.astype(np.float64))
    ) / cached_audited.astype(np.float64)
    workload_proxy_scale = float(np.ceil(observed_ratio.max() * 10.0) / 10.0)
    workload_proxy = np.ceil(
        old_lengths.astype(np.float64) * workload_proxy_scale
    ).astype(np.int64)
    noisy = joint.astype(np.int64) - actual.astype(np.int64)
    exact_workload = actual.astype(np.int64) + 2 * noisy
    if np.any(exact_workload <= 0) or np.any(exact_workload > np.iinfo(np.int32).max):
        raise RuntimeError("invalid exact DLM workload length")
    workload_proxy[audited_positions] = exact_workload
    clean_sha = save_npy(args.output_clean_lengths.resolve(), clean_proxy[keep].astype(np.int32))
    workload_sha = save_npy(
        args.output_workload_lengths.resolve(), workload_proxy[keep].astype(np.int32)
    )

    audited_keep = ~np.isin(audited_positions, invalid_positions, assume_unique=False)
    kept_old_positions = audited_positions[audited_keep]
    shift = np.searchsorted(invalid_positions, kept_old_positions, side="left")
    new_positions = kept_old_positions - shift
    new_cached = cached_audited[audited_keep]
    new_actual = actual[audited_keep]
    new_joint = joint[audited_keep]
    new_exact_workload = exact_workload[audited_keep].astype(np.int32)
    output_actual_dir = args.output_actual_audit.resolve()
    output_arrays = output_actual_dir / "actual_lengths.npz"
    arrays_sha = save_npz(
        output_arrays,
        positions=new_positions.astype(np.int64),
        cached_lengths=new_cached,
        actual_lengths=new_actual,
        actual_joint_lengths=new_joint,
        packing_workload_lengths=new_exact_workload,
        invalid_positions=np.empty(0, dtype=np.int64),
    )
    derived_actual = dict(actual_report)
    derived_actual.update(
        {
            "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_manifest": str(output_manifest),
            "sampling_manifest_sha256": sha256_file(output_manifest),
            "selected_lengths": str(args.output_clean_lengths.resolve()),
            "selected_lengths_sha256": clean_sha,
            "packing_workload_lengths": str(args.output_workload_lengths.resolve()),
            "packing_workload_lengths_sha256": workload_sha,
            "packing_workload_definition": (
                "exact clean+2*noisy for audited rows; conservative audited-ratio envelope "
                "times cached_length outside audit scope"
            ),
            "packing_workload_proxy_scale": workload_proxy_scale,
            "selected_rows": new_rows,
            "audited_rows": int(len(new_positions)),
            "cached_min": int(new_cached.min()),
            "cached_max": int(new_cached.max()),
            "actual_min": int(new_actual.min()),
            "actual_max": int(new_actual.max()),
            "actual_joint_max": int(new_joint.max()),
            "delta_min": int((new_actual - new_cached).min()),
            "delta_max": int((new_actual - new_cached).max()),
            "invalid_rows": [],
            "arrays": str(output_arrays),
            "arrays_sha256": arrays_sha,
            "derivation": "parent tail+multi-image audit minus every explicit real-length reject",
            "parent_actual_length_audit": str(actual_report_path),
            "parent_actual_length_audit_sha256": sha256_file(actual_report_path),
            "parent_selected_rows": old_rows,
            "deleted_rows": int(len(invalid_positions)),
            "deleted_rejections": manifest["deleted_actual_length_rejections"],
        }
    )
    # The exact multi-image count in the parent is no longer the count in the
    # derived sample; the candidate rule and full audited position set remain.
    derived_actual.pop("selected_multi_image_rows", None)
    atomic_json(output_actual_dir / "report.json", derived_actual)

    if strict_image_audit:
        derived_image = {
            "schema_version": 1,
            "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_manifest": str(output_manifest),
            "sampling_manifest_sha256": sha256_file(output_manifest),
            "selected_rows": new_rows,
            "audited_rows": new_rows,
            "decoded_images": int(image_report["decoded_images"]),
            "image_decode_enabled": True,
            "validated_multimodal_rows": new_rows,
            "invalid_rows": [],
            "derivation": "full parent image audit with actual-length rejects removed",
            "parent_image_audit": str(image_report_path),
            "parent_image_audit_sha256": sha256_file(image_report_path),
            "parent_selected_rows": old_rows,
            "deleted_rows": int(len(invalid_positions)),
        }
    else:
        derived_image = {
            "schema_version": 1,
            "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_manifest": str(output_manifest),
            "sampling_manifest_sha256": sha256_file(output_manifest),
            "selected_rows": new_rows,
            "image_decode_enabled": False,
            "invalid_rows": [],
            "deleted_rows": int(len(invalid_positions)),
            "claim": "prior full-epoch source-loader completion; not a new strict image decode audit",
            **source_epoch_provenance,
        }
    atomic_json(args.output_image_audit.resolve(), derived_image)
    print(
        json.dumps(
            {
                "status": "PASS",
                "old_rows": old_rows,
                "deleted_rows": int(len(invalid_positions)),
                "new_rows": new_rows,
                "sampling_manifest": str(output_manifest),
                "clean_lengths_sha256": clean_sha,
                "workload_lengths_sha256": workload_sha,
                "actual_audit": str(output_actual_dir / "report.json"),
                "image_audit": str(args.output_image_audit.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
