#!/usr/bin/env python3
"""Bitwise verification for a GroundAnything-VLM GAM token migration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from train.tokenizer.extend_tokens import (
    BASE_EMBEDDING_ROWS,
    COORDINATE_TOKENS,
    INPUT_EMBEDDING_KEY,
    OUTPUT_EMBEDDING_KEY,
    SEP_TOKEN,
    VOCAB_KEYS,
)
from train.tokenizer.validate_tokens import validate_model_dir


class IntegrityError(RuntimeError):
    """Raised when migration changed anything outside its explicit contract."""


def _layout(model_dir: Path) -> dict[str, str]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise IntegrityError(f"expected one index in {model_dir}: {indexes}")
    value = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise IntegrityError(f"invalid weight map in {indexes[0]}")
    return dict(weight_map)


def _tensor(model_dir: Path, layout: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(
        str(model_dir / layout[key]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(key)


def _mean_rows(weight: torch.Tensor, ids: list[int]) -> torch.Tensor:
    unique = list(dict.fromkeys(ids))
    return weight[unique].float().mean(dim=0).to(weight.dtype)


def verify(source_dir: Path, migrated_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve(strict=True)
    migrated_dir = migrated_dir.expanduser().resolve(strict=True)
    migrated_manifest = validate_model_dir(migrated_dir)

    source_layout = _layout(source_dir)
    migrated_layout = _layout(migrated_dir)
    if set(source_layout) != set(migrated_layout):
        missing = sorted(set(source_layout) - set(migrated_layout))
        extra = sorted(set(migrated_layout) - set(source_layout))
        raise IntegrityError(f"tensor key drift: missing={missing} extra={extra}")

    checked_non_vocab = 0
    for key in sorted(source_layout):
        if key in VOCAB_KEYS:
            continue
        source_tensor = _tensor(source_dir, source_layout, key)
        migrated_tensor = _tensor(migrated_dir, migrated_layout, key)
        if source_tensor.dtype != migrated_tensor.dtype:
            raise IntegrityError(f"dtype drift: {key}")
        if tuple(source_tensor.shape) != tuple(migrated_tensor.shape):
            raise IntegrityError(f"shape drift: {key}")
        if not torch.equal(source_tensor, migrated_tensor):
            raise IntegrityError(f"value drift: {key}")
        checked_non_vocab += 1
        del source_tensor, migrated_tensor

    source_tokenizer = AutoTokenizer.from_pretrained(
        str(source_dir), trust_remote_code=True, local_files_only=True
    )
    migrated_tokenizer = AutoTokenizer.from_pretrained(
        str(migrated_dir), trust_remote_code=True, local_files_only=True
    )
    numeric_ids = [
        source_tokenizer.encode(str(value), add_special_tokens=False)
        for value in range(1000)
    ]
    separator_ids: list[int] = []
    for text in (",", ";", " and ", " or "):
        separator_ids.extend(
            source_tokenizer.encode(text, add_special_tokens=False)
        )

    vocab_reports: dict[str, Any] = {}
    for key in VOCAB_KEYS:
        source_weight = _tensor(source_dir, source_layout, key)
        migrated_weight = _tensor(migrated_dir, migrated_layout, key)
        if not torch.equal(
            source_weight[:BASE_EMBEDDING_ROWS],
            migrated_weight[:BASE_EMBEDDING_ROWS],
        ):
            # Rows 151669..151935 are intentionally reassigned from unused
            # embedding padding, so compare only the addressable source rows.
            base_tokenizer_length = len(source_tokenizer)
            if not torch.equal(
                source_weight[:base_tokenizer_length],
                migrated_weight[:base_tokenizer_length],
            ):
                raise IntegrityError(f"old tokenizer-addressable rows changed: {key}")
        for value, token in enumerate(COORDINATE_TOKENS):
            token_id = migrated_tokenizer.convert_tokens_to_ids(token)
            expected = _mean_rows(source_weight, numeric_ids[value])
            if not torch.equal(migrated_weight[token_id], expected):
                raise IntegrityError(f"numeric initializer mismatch: {key} {token}")
        separator_id = migrated_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        expected_separator = _mean_rows(source_weight, separator_ids)
        if not torch.equal(migrated_weight[separator_id], expected_separator):
            raise IntegrityError(f"separator initializer mismatch: {key}")
        vocab_reports[key] = {
            "source_shape": list(source_weight.shape),
            "migrated_shape": list(migrated_weight.shape),
            "old_addressable_rows_bitwise_equal": True,
            "coordinate_initializers_exact": 1000,
            "separator_initializer_exact": True,
        }
        del source_weight, migrated_weight

    source_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    migrated_config = json.loads(
        (migrated_dir / "config.json").read_text(encoding="utf-8")
    )
    expected_config = json.loads(json.dumps(source_config))
    expected_config["text_config"]["vocab_size"] = migrated_manifest["vocab_size"]
    if expected_config != migrated_config:
        raise IntegrityError("config changed outside text_config.vocab_size")

    return {
        "schema_version": 1,
        "status": "PASS",
        "source_dir": str(source_dir),
        "migrated_dir": str(migrated_dir),
        "tensor_key_count": len(source_layout),
        "non_vocab_tensors_bitwise_equal": checked_non_vocab,
        "vocabulary_tensors": vocab_reports,
        "only_config_change": "text_config.vocab_size",
        "migrated_tokenizer_sha256": migrated_manifest["tokenizer_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("migrated_dir", type=Path)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    report = verify(args.source_dir, args.migrated_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.write_report:
        destination = args.write_report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    print(payload, end="")


if __name__ == "__main__":
    main()
