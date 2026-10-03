#!/usr/bin/env python3
"""Atomically append GAM coordinate tokens to a GroundAnything-VLM checkpoint.

The GroundAnything checkpoint uses an untied Qwen3 input embedding and LM head.  Both
vocabulary tensors are therefore expanded and initialized independently from
the corresponding Qwen3 numeric-token rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import shutil
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


COORDINATE_TOKENS = tuple(f"<{index}>" for index in range(1000))
SEP_TOKEN = "</c>"
INPUT_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
OUTPUT_EMBEDDING_KEY = "lm_head.weight"
VOCAB_KEYS = (INPUT_EMBEDDING_KEY, OUTPUT_EMBEDDING_KEY)
BASE_TOKENIZER_LENGTH = 151669
BASE_EMBEDDING_ROWS = 151936
TARGET_TOKENIZER_LENGTH = 152670
TARGET_EMBEDDING_ROWS = 152670
HIDDEN_SIZE = 2560
FIXED_TOKEN_IDS = {
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
}


class MigrationError(RuntimeError):
    """Raised when the copied checkpoint violates the migration contract."""


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_layout(model_dir: Path) -> tuple[dict[str, str], Path]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise MigrationError(f"expected one safetensors index, found {indexes}")
    index = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise MigrationError("safetensors index has no weight_map")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise MigrationError("safetensors weight_map must be string -> string")
    return dict(weight_map), indexes[0]


def _mean_rows(weight: torch.Tensor, ids: list[int]) -> torch.Tensor:
    unique = list(dict.fromkeys(ids))
    if not unique or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= BASE_TOKENIZER_LENGTH
        for index in unique
    ):
        raise MigrationError(f"invalid initializer token ids: {unique}")
    return weight[unique].float().mean(dim=0).to(weight.dtype)


def _validate_base_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if config.get("model_type") != "groundinganything_vlm":
        raise MigrationError(f"unexpected model_type: {config.get('model_type')}")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3":
        raise MigrationError("config.text_config must be Qwen3")
    if int(text_config.get("vocab_size", -1)) != BASE_EMBEDDING_ROWS:
        raise MigrationError("unexpected base text vocab size")
    if int(text_config.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise MigrationError("unexpected Qwen3 hidden size")
    if config.get("tie_word_embeddings") is not False:
        raise MigrationError("top-level embeddings must be untied")
    if text_config.get("tie_word_embeddings") is not False:
        raise MigrationError("text embeddings must be untied")
    return config


def migrate(model_dir: Path, *, keep_backup: bool) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve(strict=True)
    if (model_dir / "gam_tokenizer_manifest.json").exists():
        raise MigrationError("model already has gam_tokenizer_manifest.json")

    config = _validate_base_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    if len(tokenizer) != BASE_TOKENIZER_LENGTH:
        raise MigrationError(f"unexpected base tokenizer length: {len(tokenizer)}")
    for token, expected in FIXED_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise MigrationError(f"fixed token id drift: {token} {actual} != {expected}")
    collisions = [
        token for token in (*COORDINATE_TOKENS, SEP_TOKEN) if token in tokenizer.get_vocab()
    ]
    if collisions:
        raise MigrationError(f"new-token collision: {collisions[:8]}")

    numeric_source_ids = [
        tokenizer.encode(str(value), add_special_tokens=False) for value in range(1000)
    ]
    separator_source_ids: list[int] = []
    for text in (",", ";", " and ", " or "):
        separator_source_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    existing_additional = list(tokenizer.additional_special_tokens)
    added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": existing_additional
            + list(COORDINATE_TOKENS)
            + [SEP_TOKEN]
        },
        replace_additional_special_tokens=True,
    )
    if added != 1001 or len(tokenizer) != TARGET_TOKENIZER_LENGTH:
        raise MigrationError(
            f"tokenizer append mismatch: added={added} len={len(tokenizer)}"
        )
    coordinate_ids = [tokenizer.convert_tokens_to_ids(t) for t in COORDINATE_TOKENS]
    expected_ids = list(range(BASE_TOKENIZER_LENGTH, BASE_TOKENIZER_LENGTH + 1000))
    if coordinate_ids != expected_ids:
        raise MigrationError("coordinate IDs are not the expected contiguous range")
    separator_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    if separator_id != TARGET_TOKENIZER_LENGTH - 1:
        raise MigrationError(f"unexpected separator ID: {separator_id}")

    weight_map, index_path = _weight_layout(model_dir)
    vocab_shards = {weight_map.get(key) for key in VOCAB_KEYS}
    if None in vocab_shards or len(vocab_shards) != 1:
        raise MigrationError(f"vocabulary tensors must share one shard: {vocab_shards}")
    vocab_shard_name = next(iter(vocab_shards))
    assert isinstance(vocab_shard_name, str)
    vocab_shard = model_dir / vocab_shard_name
    with safe_open(str(vocab_shard), framework="pt", device="cpu") as handle:
        for key in VOCAB_KEYS:
            if key not in handle.keys():
                raise MigrationError(f"missing vocabulary tensor: {key}")
            shape = tuple(handle.get_slice(key).get_shape())
            dtype = str(handle.get_slice(key).get_dtype())
            if shape != (BASE_EMBEDDING_ROWS, HIDDEN_SIZE) or dtype != "BF16":
                raise MigrationError(f"unexpected {key} contract: {shape} {dtype}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = model_dir.parent / f".{model_dir.name}.gam-expand-{stamp}-{os.getpid()}"
    backup = model_dir.parents[1] / "temp" / f"{model_dir.name}.pre-special-token-{stamp}"
    if staging.exists() or backup.exists():
        raise MigrationError(f"staging/backup collision: {staging} / {backup}")
    staging.mkdir(parents=True)

    try:
        shard_names = set(weight_map.values())
        for item in model_dir.iterdir():
            if item.name in shard_names or item.name == index_path.name:
                continue
            destination = staging / item.name
            if item.is_file() and not item.is_symlink():
                shutil.copy2(item, destination)
            elif item.is_dir() and not item.is_symlink():
                shutil.copytree(item, destination)
            else:
                raise MigrationError(f"unsupported model entry: {item}")

        source_hashes = {
            name: _sha256_file(model_dir / name) for name in sorted(shard_names)
        }
        for shard_name in sorted(shard_names):
            source_shard = model_dir / shard_name
            destination = staging / shard_name
            if shard_name != vocab_shard_name:
                os.link(source_shard, destination)
                continue

            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(source_shard), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for key in handle.keys():
                    tensors[key] = handle.get_tensor(key)

            for key in VOCAB_KEYS:
                old_weight = tensors[key]
                new_weight = torch.empty(
                    (TARGET_EMBEDDING_ROWS, HIDDEN_SIZE), dtype=old_weight.dtype
                )
                new_weight[:BASE_EMBEDDING_ROWS].copy_(old_weight)
                for value, token_id in enumerate(coordinate_ids):
                    new_weight[token_id].copy_(
                        _mean_rows(old_weight, numeric_source_ids[value])
                    )
                new_weight[separator_id].copy_(
                    _mean_rows(old_weight, separator_source_ids)
                )
                tensors[key] = new_weight

            temporary = destination.with_suffix(destination.suffix + ".tmp")
            save_file(tensors, str(temporary), metadata=metadata)
            os.replace(temporary, destination)
            del tensors

        index = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = index.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("total_size"), int):
            raise MigrationError("index metadata.total_size is missing")
        metadata["total_size"] += (
            TARGET_EMBEDDING_ROWS - BASE_EMBEDDING_ROWS
        ) * HIDDEN_SIZE * 2 * len(VOCAB_KEYS)
        _atomic_json(staging / index_path.name, index)

        config["text_config"]["vocab_size"] = TARGET_EMBEDDING_ROWS
        _atomic_json(staging / "config.json", config)
        tokenizer.save_pretrained(str(staging))
        if (model_dir / "chat_template.jinja").is_file():
            shutil.copy2(
                model_dir / "chat_template.jinja", staging / "chat_template.jinja"
            )

        from train.tokenizer.validate_tokens import validate_model_dir

        manifest = validate_model_dir(staging)
        manifest["migration"] = {
            "schema_version": 1,
            "base_tokenizer_length": BASE_TOKENIZER_LENGTH,
            "base_embedding_rows": BASE_EMBEDDING_ROWS,
            "target_tokenizer_length": TARGET_TOKENIZER_LENGTH,
            "target_embedding_rows": TARGET_EMBEDDING_ROWS,
            "hidden_size": HIDDEN_SIZE,
            "new_token_count": 1001,
            "padding_rows_reassigned": BASE_EMBEDDING_ROWS - BASE_TOKENIZER_LENGTH,
            "rows_appended_per_vocab_tensor": TARGET_EMBEDDING_ROWS
            - BASE_EMBEDDING_ROWS,
            "source_shard_sha256s": source_hashes,
        }
        _atomic_json(staging / "gam_tokenizer_manifest.json", manifest)

        backup.parent.mkdir(parents=True, exist_ok=True)
        model_dir.rename(backup)
        try:
            staging.rename(model_dir)
        except BaseException:
            backup.rename(model_dir)
            raise

        final_manifest = validate_model_dir(model_dir)
        final_manifest["migration"] = manifest["migration"]
        if keep_backup:
            final_manifest["pre_migration_backup"] = str(backup)
        else:
            shutil.rmtree(backup)
        _atomic_json(model_dir / "gam_tokenizer_manifest.json", final_manifest)
        return final_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--delete-backup-after-validation", action="store_true")
    args = parser.parse_args()
    result = migrate(
        args.model_dir,
        keep_backup=not args.delete_backup_after_validation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
