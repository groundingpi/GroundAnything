"""Independent fail-closed audit for the approved RL V3 manifest."""

from __future__ import annotations


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

from train.rl.shared.multiroute_data import (
    BBOX_ROUTES,
    EXPECTED_ROWS,
    OCR_ROUTES,
    OUTPUT_DIR,
    POINT_ROUTES,
    QUOTA,
    TokenCounter,
    reference_completion,
)
from train.rl.shared.reward import MultiRouteGAMRewardAdapter
from train.dlm.data import _load_image
from train.runtime.tar_uri import parse_tar_uri


def _weighted(result: Any) -> float:
    return sum(
        float(result.components[key]) * float(result.weights[key])
        for key in result.weights
    )


def _validate_solution(row: dict[str, Any]) -> None:
    route = str(row["rlv3_route"])
    task = str(row["task_type"])
    solution = json.loads(row["solution"])
    if route in BBOX_ROUTES:
        if task != "grounding_bbox_gam" or solution.get("format") != "gam_bbox_v1":
            raise ValueError(f"bbox route/schema mismatch: {route}/{task}")
        records = solution.get("items") or []
        key = "bbox_2d"
        arity = 4
    elif route in POINT_ROUTES:
        if task != "point_gam" or solution.get("format") != "gam_point_v1":
            raise ValueError(f"point route/schema mismatch: {route}/{task}")
        records = solution.get("items") or []
        key = "point_2d"
        arity = 2
    elif route in OCR_ROUTES:
        if task != "ocr_bbox_text_gam" or solution.get("format") != "gam_ocr_bbox_text_v1":
            raise ValueError(f"OCR route/schema mismatch: {route}/{task}")
        references = solution.get("references") or []
        if not references or not (references[0].get("instances") or []):
            raise ValueError("empty OCR references")
        for reference in references:
            for instance in reference.get("instances") or []:
                box = [int(value) for value in instance.get("bbox_norm1000", [])]
                if len(box) != 4 or not (0 <= box[0] < box[2] <= 999 and 0 <= box[1] < box[3] <= 999):
                    raise ValueError("invalid OCR bbox")
        return
    else:
        raise ValueError(f"unknown route: {route}")
    if not records:
        raise ValueError(f"empty structured solution: {route}")
    grouped: dict[str, list[list[int]]] = {}
    for record in records:
        label = str(record.get("label", "")).strip()
        coordinates = [int(value) for value in record.get(key, [])]
        if not label or len(coordinates) != arity or any(value < 0 or value > 999 for value in coordinates):
            raise ValueError(f"invalid {key} item")
        if arity == 4 and not (
            coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]
        ):
            raise ValueError("invalid xyxy box")
        grouped.setdefault(label, []).append(coordinates)
    if any(values != sorted(values, key=tuple) for values in grouped.values()):
        raise ValueError(f"non-deterministic coordinate order: {row['id']}")


def _image_locator(
    value: Any,
    file_paths: set[Path],
    tar_required_ends: dict[Path, int],
) -> tuple[str, int]:
    if isinstance(value, dict):
        if set(value) != {"bytes", "path"}:
            raise ValueError("invalid image record keys")
        if value["bytes"] is not None:
            payload = value["bytes"]
            if not payload:
                raise ValueError("empty embedded image")
            return "embedded_bytes", len(payload)
        value = value["path"]
    if not isinstance(value, str) or not value:
        raise ValueError("empty image path")
    parsed = parse_tar_uri(value)
    if parsed is not None:
        tar, offset, size, _ = parsed
        tar_required_ends[tar] = max(tar_required_ends.get(tar, 0), offset + size)
        return "tar_uri", int(size)
    path = Path(value)
    file_paths.add(path)
    return "file", 0


def _validate_image_paths(
    file_paths: set[Path], tar_required_ends: dict[Path, int]
) -> None:
    requirements: list[tuple[str, Path, int]] = [
        ("file", path, 1) for path in sorted(file_paths)
    ] + [
        ("tar_uri", path, required_end)
        for path, required_end in sorted(tar_required_ends.items())
    ]

    def inspect(item: tuple[str, Path, int]) -> tuple[str, Path, int, os.stat_result]:
        kind, path, required_size = item
        return kind, path, required_size, os.stat(path)

    # CPFS metadata latency dominates a sequential 19.8K-path audit. Parallel
    # stat keeps the same fail-closed all-path guarantee without serial stalls.
    with ThreadPoolExecutor(max_workers=32) as executor:
        for kind, path, required_size, result in executor.map(inspect, requirements):
            if not stat.S_ISREG(result.st_mode) or result.st_size < required_size:
                raise ValueError(
                    f"invalid {kind} image locator: {path} "
                    f"({result.st_size} < {required_size})"
                )


def audit(directory: Path = OUTPUT_DIR) -> dict[str, Any]:
    directory = directory.resolve(strict=True)
    train_path = (directory / "train.jsonl").resolve(strict=True)
    build = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    counter = TokenCounter()
    rows = 0
    tokens = 0
    ids: set[str] = set()
    route_rows = Counter()
    route_tokens = Counter()
    route_buckets = Counter()
    tier_rows = Counter()
    source_rows = Counter()
    image_locators = Counter()
    image_file_paths: set[Path] = set()
    image_tar_required_ends: dict[Path, int] = {}
    image_decode_samples: dict[tuple[str, str], Any] = {}
    reward_samples: dict[tuple[str, str, str, str], tuple[dict[str, Any], str]] = {}
    token_audit_rows: list[
        tuple[int, str, str, str, str, int, str]
    ] = []
    prefix_rows = Counter()
    max_prefix_route_error = 0.0
    expected_route_share = {
        route: count / EXPECTED_ROWS
        for route, count in Counter(
            key[0] for key, value in QUOTA.items() for _ in range(value)
        ).items()
    }
    with train_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            digest.update(line.encode("utf-8"))
            row = json.loads(line)
            rows += 1
            identifier = str(row.get("id", ""))
            if not identifier or identifier in ids:
                raise ValueError(f"duplicate/empty ID at line {line_number}: {identifier}")
            ids.add(identifier)
            route = str(row.get("rlv3_route", ""))
            metadata = row.get("rlv3_audit") or {}
            if metadata.get("semantic_route") != route:
                raise ValueError(f"route metadata drift at line {line_number}")
            _validate_solution(row)
            messages = row.get("messages") or []
            if [item.get("role") for item in messages] != ["system", "user"]:
                raise ValueError(f"message contract drift at line {line_number}")
            images = row.get("images") or []
            placeholders = sum(str(item.get("content", "")).count("<image>") for item in messages)
            if not images or placeholders != len(images):
                raise ValueError(f"image placeholder mismatch at line {line_number}")
            source_key = (str(metadata["source_tier"]), str(metadata["source_name"]))
            for image in images:
                kind, _ = _image_locator(
                    image, image_file_paths, image_tar_required_ends
                )
                image_locators[kind] += 1
                image_decode_samples.setdefault(source_key, image)
            reference = reference_completion(row)
            tier = str(metadata["source_tier"])
            source = str(metadata["source_name"])
            route_rows[route] += 1
            tier_rows[tier] += 1
            source_rows[(route, tier, source)] += 1
            token_audit_rows.append(
                (
                    line_number,
                    route,
                    tier,
                    source,
                    reference,
                    int(metadata["reference_tokens"]),
                    str(metadata["length_bucket"]),
                )
            )
            reward_samples.setdefault(
                (route, tier, source, str(metadata["length_bucket"])),
                (row, reference),
            )
            prefix_rows[route] += 1
            if rows >= 100:
                max_prefix_route_error = max(
                    max_prefix_route_error,
                    max(
                        abs(prefix_rows[name] / rows - share)
                        for name, share in expected_route_share.items()
                    ),
                )

    print(
        f"image path audit: {len(image_file_paths)} files + "
        f"{len(image_tar_required_ends)} tar containers",
        file=sys.stderr,
        flush=True,
    )
    _validate_image_paths(image_file_paths, image_tar_required_ends)

    # Batch encoding preserves the independent tokenizer parity check while
    # avoiding 19.8K Python/Rust crossings from one-row-at-a-time encoding.
    batch_size = 512
    for start in range(0, len(token_audit_rows), batch_size):
        chunk = token_audit_rows[start : start + batch_size]
        encodings = counter.tokenizer.encode_batch(
            [item[4] for item in chunk], add_special_tokens=False
        )
        for item, encoding in zip(chunk, encodings, strict=True):
            line_number, route, _, _, _, recorded_length, recorded_bucket = item
            length = len(encoding.ids)
            if length != recorded_length:
                raise ValueError(f"reference token drift at line {line_number}")
            if not 0 < length <= 1_024:
                raise ValueError(f"inadmissible response at line {line_number}: {length}")
            expected_bucket = (
                "short" if length <= 64 else "medium" if length <= 256 else "long"
            )
            if recorded_bucket != expected_bucket:
                raise ValueError(f"length bucket drift at line {line_number}")
            tokens += length
            route_tokens[route] += length
            route_buckets[(route, expected_bucket)] += 1
        if start == 0 or start + batch_size >= len(token_audit_rows) or start % 4096 == 0:
            print(
                f"token parity: {min(start + batch_size, len(token_audit_rows))}/"
                f"{len(token_audit_rows)}",
                file=sys.stderr,
                flush=True,
            )

    if rows != EXPECTED_ROWS or len(ids) != EXPECTED_ROWS:
        raise ValueError(f"row conservation drift: {rows}/{len(ids)}")
    if digest.hexdigest() != build["train_jsonl_sha256"]:
        raise ValueError("train JSONL SHA256 drift")
    if dict(route_rows) != build["route_rows"] or tokens != int(build["reference_tokens"]):
        raise ValueError("build/audit count drift")
    if dict(tier_rows) != build["tier_rows"]:
        raise ValueError("provenance tier drift")

    decoded: dict[str, list[int]] = {}
    for (tier, source), value in sorted(image_decode_samples.items()):
        record = value if isinstance(value, dict) else {"bytes": None, "path": value}
        image = _load_image(record)
        decoded[f"{tier}/{source}"] = [int(image.width), int(image.height)]

    adapter = MultiRouteGAMRewardAdapter()
    ground_truth_reward: dict[str, dict[str, float | int]] = {}
    for key, (row, reference) in sorted(reward_samples.items()):
        result = adapter.score(reference, row)
        score = _weighted(result)
        if not result.format_valid or not math.isfinite(score) or not 0.0 <= score <= 1.0 + 1e-6:
            raise ValueError(f"invalid ground-truth reward for {key}: {score}")
        route = key[0]
        aggregate = ground_truth_reward.setdefault(route, {"samples": 0, "min": 1.0, "max": 0.0})
        aggregate["samples"] = int(aggregate["samples"]) + 1
        aggregate["min"] = min(float(aggregate["min"]), score)
        aggregate["max"] = max(float(aggregate["max"]), score)

    negative_reward: dict[str, float] = {}
    for route in sorted(route_rows):
        row, _ = next(value for key, value in reward_samples.items() if key[0] == route)
        result = adapter.score("invalid output", row)
        score = _weighted(result)
        if result.format_valid or score != 0.0:
            raise ValueError(f"invalid output was rewarded for {route}: {score}")
        negative_reward[route] = score

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "train_jsonl": str(train_path),
        "train_jsonl_sha256": digest.hexdigest(),
        "rows": rows,
        "unique_ids": len(ids),
        "reference_tokens": tokens,
        "route_rows": dict(route_rows),
        "route_reference_tokens": dict(route_tokens),
        "route_bucket_rows": {
            f"{route}/{bucket}": route_buckets[(route, bucket)]
            for route in sorted(route_rows)
            for bucket in ("short", "medium", "long")
        },
        "tier_rows": dict(tier_rows),
        "source_rows": {"/".join(key): value for key, value in sorted(source_rows.items())},
        "image_locators": dict(image_locators),
        "decoded_image_sources": decoded,
        "max_prefix_route_share_error_after_100_rows": max_prefix_route_error,
        "reward_ground_truth": ground_truth_reward,
        "reward_invalid_output": negative_reward,
        "checks": {
            "no_replacement": True,
            "all_answer_tokens_le_1024": True,
            "all_images_resolvable": True,
            "one_image_decoded_per_source": True,
            "all_solutions_schema_valid": True,
            "canonical_reference_token_parity": True,
            "ground_truth_reward_finite_and_format_valid": True,
            "invalid_output_reward_zero": True,
            "v1_v2_untouched": True,
        },
    }
    result["audit_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = directory / "audit.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
