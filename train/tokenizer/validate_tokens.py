#!/usr/bin/env python3
"""Validate the GroundAnything-VLM GAM tokenizer and both vocabulary tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safetensors import safe_open
from transformers import AutoTokenizer

from train.tokenizer.extend_tokens import (
    COORDINATE_TOKENS,
    FIXED_TOKEN_IDS,
    HIDDEN_SIZE,
    INPUT_EMBEDDING_KEY,
    OUTPUT_EMBEDDING_KEY,
    SEP_TOKEN,
    TARGET_EMBEDDING_ROWS,
    TARGET_TOKENIZER_LENGTH,
)


class ValidationError(RuntimeError):
    """Raised when a migrated GroundAnything model violates its hard contract."""


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValidationError(f"expected one safetensors index: {indexes}")
    index = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    result: dict[str, tuple[int, ...]] = {}
    for key in (INPUT_EMBEDDING_KEY, OUTPUT_EMBEDDING_KEY):
        filename = weight_map.get(key)
        if not isinstance(filename, str):
            raise ValidationError(f"index lacks {key}")
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise ValidationError(f"shard lacks {key}")
            shape = tuple(handle.get_slice(key).get_shape())
            dtype = str(handle.get_slice(key).get_dtype())
            if dtype != "BF16":
                raise ValidationError(f"unexpected {key} dtype: {dtype}")
            result[key] = shape
    return result


def validate_model_dir(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve(strict=True)
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "modeling_groundinganything.py",
        "processing_groundinganything.py",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise ValidationError(f"missing GroundAnything assets: {missing}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if config.get("model_type") != "groundinganything_vlm":
        raise ValidationError("top-level model_type is not groundinganything_vlm")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3":
        raise ValidationError("text_config is not Qwen3")
    if int(text_config.get("vocab_size", -1)) != TARGET_EMBEDDING_ROWS:
        raise ValidationError("text vocab size does not match migrated contract")
    if int(text_config.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise ValidationError("text hidden size drift")
    if config.get("tie_word_embeddings") is not False:
        raise ValidationError("top-level model unexpectedly tied")
    if text_config.get("tie_word_embeddings") is not False:
        raise ValidationError("Qwen3 model unexpectedly tied")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    if len(tokenizer) != TARGET_TOKENIZER_LENGTH:
        raise ValidationError(f"tokenizer length drift: {len(tokenizer)}")
    for token, expected in FIXED_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValidationError(f"fixed token ID drift: {token} {actual} != {expected}")

    coordinate_ids: list[int] = []
    for token in COORDINATE_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != token:
            raise ValidationError(f"coordinate token is not atomic: {token} -> {ids}")
        coordinate_ids.append(ids[0])
    expected_coordinate_ids = list(range(coordinate_ids[0], coordinate_ids[0] + 1000))
    if coordinate_ids != expected_coordinate_ids:
        raise ValidationError("coordinate IDs are not ordered and contiguous")
    sep_ids = tokenizer.encode(SEP_TOKEN, add_special_tokens=False)
    if len(sep_ids) != 1 or tokenizer.decode(
        sep_ids, skip_special_tokens=False
    ) != SEP_TOKEN:
        raise ValidationError(f"separator is not atomic: {sep_ids}")

    shapes = _weight_shapes(model_dir)
    expected_shape = (TARGET_EMBEDDING_ROWS, HIDDEN_SIZE)
    for key, shape in shapes.items():
        if shape != expected_shape:
            raise ValidationError(f"unexpected {key} shape: {shape}")

    file_hashes = {
        filename: _sha256_file(model_dir / filename)
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "config.json",
        )
    }
    return {
        "schema_version": 1,
        "manifest_type": "gam_groundinganything_vlm_tokenizer_manifest",
        "model_path": str(model_dir),
        "model_type": "groundinganything_vlm",
        "text_model_type": "qwen3",
        "weight_sharing": "untied",
        "tokenizer_sha256": file_hashes["tokenizer.json"],
        "tokenizer_file_sha256s": file_hashes,
        "vocab_size": TARGET_EMBEDDING_ROWS,
        "tokenizer_length": TARGET_TOKENIZER_LENGTH,
        "input_embedding_shape": list(shapes[INPUT_EMBEDDING_KEY]),
        "output_embedding_shape": list(shapes[OUTPUT_EMBEDDING_KEY]),
        "coordinate_token_count": 1000,
        "coordinate_id_start": coordinate_ids[0],
        "coordinate_id_end": coordinate_ids[-1],
        "separator_token": SEP_TOKEN,
        "separator_token_id": sep_ids[0],
        "fixed_token_ids": FIXED_TOKEN_IDS,
        "decode_requires_skip_special_tokens_false": True,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    manifest = validate_model_dir(args.model_dir)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.write_manifest:
        destination = args.write_manifest.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    print(payload, end="")


if __name__ == "__main__":
    main()
