#!/usr/bin/env python3
"""Shared, dependency-light helpers for GAM launch-time validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SCHEMA_VERSION = 1
PACKING_FINGERPRINT_SCHEMA_VERSION = 2
MODEL_MANIFEST_TYPE = "gam_tokenizer_manifest"
CACHE_MANIFEST_TYPE = "gam_cache_manifest"
SOURCE_MANIFEST_TYPE = "gam_cleaned_source_manifest"
MODEL_MANIFEST_BASENAME = "gam_tokenizer_manifest.json"
CACHE_MANIFEST_BASENAME = "cache_manifest.json"
SOURCE_MANIFEST_BASENAME = "source_manifest.json"
TOKENIZER_HASH_ALGORITHM = "sha256:tokenizer.json-bytes"

DATA_POOL_NAME = "special_token_data_V1"

DEFAULT_DATA_ROOT = Path(
    f"data/train/{DATA_POOL_NAME}"
)

FIXED_QWEN35_TOKEN_IDS: Dict[str, int] = {
    "<|object_ref_start|>": 248047,
    "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    # The vendored-free Docker swift Qwen3.5 template uses these IDs directly.
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
}

REQUIRED_TOKENIZER_AUX_FILES: Tuple[str, ...] = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)

EXPECTED_CACHE_COLUMNS: Tuple[str, ...] = ("id", "messages", "images", "length")
EXPECTED_MESSAGE_COLUMNS: Tuple[str, ...] = ("role", "content", "loss")
EXPECTED_IMAGE_COLUMNS: Tuple[str, ...] = ("bytes", "path")

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DATA_POOL_RE = re.compile(
    r"^(?:special_token_data_V[1-9][0-9]*|general_data_V1)$"
)


class ValidationError(RuntimeError):
    """Raised when a launch-time invariant is violated."""


@dataclass(frozen=True)
class PathPolicy:
    gam_root: Path
    data_root: Path

    @classmethod
    def default(cls) -> "PathPolicy":
        # Resolve defaults relative to the repository, independently of cwd.
        gam_root = Path(__file__).resolve().parents[2]
        return cls(gam_root=gam_root.resolve(), data_root=(gam_root / DEFAULT_DATA_ROOT).resolve())

    @property
    def model_path(self) -> Path:
        return self.gam_root / "weights" / "base_model"

    @property
    def output_root(self) -> Path:
        return self.gam_root / "outputs" / "train"

    @property
    def train_root(self) -> Path:
        return self.gam_root / "train"


def derive_source_data_policy(
    manifest_path: os.PathLike[str] | str, policy: PathPolicy
) -> Tuple[PathPolicy, str]:
    """Derive a sibling versioned pool from a published source manifest."""

    manifest = canonical_path(manifest_path)
    value = load_json(manifest)
    pool = value.get("data_pool")
    require(isinstance(pool, str) and DATA_POOL_RE.fullmatch(pool) is not None,
            "source manifest.data_pool 不是合法 versioned GAM pool")
    data_root = manifest.parent.parent
    pool_directory_matches = data_root.name == pool
    if pool == "general_data_V1":
        # General Data revisions are immutable sibling publications.  Keep the
        # semantic data_pool stable while allowing only explicit r2+ directory
        # names; this does not relax any special_token_data path contract.
        pool_directory_matches = (
            pool_directory_matches
            or re.fullmatch(r"general_data_V1_r[2-9][0-9]*", data_root.name)
            is not None
        )
    require(
        pool_directory_matches,
        "source manifest 所在 pool 目录与 data_pool 不一致",
    )
    configured_root = canonical_path(policy.data_root, strict=False)
    require(data_root.parent == configured_root.parent,
            f"source manifest 必须位于固定 GAM data base: {configured_root.parent}")
    return PathPolicy(gam_root=policy.gam_root, data_root=data_root), pool


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def is_strict_int(value: Any) -> bool:
    """Return true for JSON/YAML integers, never for Python booleans."""
    return type(value) is int


def canonical_path(path: os.PathLike[str] | str, *, strict: bool = True) -> Path:
    p = Path(path).expanduser()
    try:
        return p.resolve(strict=strict)
    except FileNotFoundError as exc:
        raise ValidationError(f"路径不存在: {p}") from exc


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_within(path: Path, root: Path, label: str, *, allow_equal: bool = False) -> None:
    ok = path == root if allow_equal else False
    require(ok or is_within(path, root), f"{label} 必须位于 {root} 内，实际为 {path}")


def load_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    p = canonical_path(path)
    require(p.is_file(), f"JSON manifest 不是文件: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"无法读取 JSON: {p}: {exc}") from exc
    require(isinstance(obj, dict), f"JSON 顶层必须为 object: {p}")
    return obj


def atomic_write_json(path: os.PathLike[str] | str, obj: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} 必须是 64 位小写 SHA256")
    return value


def tokenizer_id_map(tokenizer_json: os.PathLike[str] | str) -> Dict[str, int]:
    data = load_json(tokenizer_json)
    result: Dict[str, int] = {}
    vocab = data.get("model", {}).get("vocab", {})
    require(isinstance(vocab, dict), "tokenizer.json 缺少 model.vocab")
    for token, token_id in vocab.items():
        if isinstance(token, str) and is_strict_int(token_id):
            result[token] = token_id
    added = data.get("added_tokens", [])
    require(isinstance(added, list), "tokenizer.json added_tokens 必须为 list")
    for item in added:
        require(isinstance(item, dict), "tokenizer.json added_tokens 元素必须为 object")
        token, token_id = item.get("content"), item.get("id")
        require(isinstance(token, str) and is_strict_int(token_id),
                "tokenizer.json added_tokens 缺少 content/id")
        prior = result.get(token)
        require(prior is None or prior == token_id, f"tokenizer token ID 冲突: {token}")
        result[token] = token_id
    require(result, "tokenizer.json 未解析出任何 token")
    return result


def tokenizer_added_token_map(tokenizer_json: os.PathLike[str] | str) -> Dict[str, Dict[str, Any]]:
    """Read the authoritative atomic-token entries from tokenizer.json."""
    data = load_json(tokenizer_json)
    added = data.get("added_tokens", [])
    require(isinstance(added, list), "tokenizer.json added_tokens 必须为 list")
    result: Dict[str, Dict[str, Any]] = {}
    for item in added:
        require(isinstance(item, dict), "tokenizer.json added_tokens 元素必须为 object")
        token, token_id = item.get("content"), item.get("id")
        require(isinstance(token, str) and is_strict_int(token_id),
                "tokenizer.json added_tokens 缺少 content/id")
        require(token not in result, f"tokenizer.json added token 重复: {token}")
        result[token] = item
    return result


def extract_text_vocab_size(config: Mapping[str, Any]) -> Optional[int]:
    candidates = [config.get("vocab_size")]
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        candidates.insert(0, text_cfg.get("vocab_size"))
    for value in candidates:
        if is_strict_int(value) and value > 0:
            return value
    return None


def get_feature_child(feature: Any) -> Dict[str, Any]:
    """Return the child mapping for HF Sequence/List-like feature JSON."""
    if not isinstance(feature, dict):
        return {}
    child = feature.get("feature")
    if isinstance(child, dict):
        return child
    # Compatibility with a few datasets releases that serialize List as `_type` + `feature` only;
    # explicit fallback keys are intentionally not accepted to keep the schema strict.
    return {}


def feature_dtype(feature: Any) -> Optional[str]:
    return feature.get("dtype") if isinstance(feature, dict) else None


def validate_cache_features(features: Any, label: str = "cache features") -> None:
    require(isinstance(features, dict), f"{label} 必须为 object")
    require(set(features) == set(EXPECTED_CACHE_COLUMNS),
            f"{label} 顶层字段必须恰为 {EXPECTED_CACHE_COLUMNS}，实际为 {tuple(features)}")
    require(feature_dtype(features["id"]) == "string", f"{label}.id 必须为 string")
    require(feature_dtype(features["length"]) == "int64", f"{label}.length 必须为 int64")

    messages = get_feature_child(features["messages"])
    require(set(messages) == set(EXPECTED_MESSAGE_COLUMNS),
            f"{label}.messages 字段必须恰为 {EXPECTED_MESSAGE_COLUMNS}，实际为 {tuple(messages)}")
    require(feature_dtype(messages["role"]) == "string", f"{label}.messages.role 必须为 string")
    require(feature_dtype(messages["content"]) == "string", f"{label}.messages.content 必须为 string")
    require(feature_dtype(messages["loss"]) == "float64",
            f"{label}.messages.loss 必须为 float64")

    images = get_feature_child(features["images"])
    require(set(images) == set(EXPECTED_IMAGE_COLUMNS),
            f"{label}.images 字段必须恰为 {EXPECTED_IMAGE_COLUMNS}，实际为 {tuple(images)}")
    require(feature_dtype(images["bytes"]) == "binary", f"{label}.images.bytes 必须为 binary，禁止 null")
    require(feature_dtype(images["path"]) == "string", f"{label}.images.path 必须为 string")


def dataset_info_features(cache_train_path: Path) -> Dict[str, Any]:
    info_path = cache_train_path / "dataset_info.json"
    info = load_json(info_path)
    features = info.get("features")
    validate_cache_features(features, str(info_path))
    return features


def validate_hf_cache_files(cache_train_path: Path) -> List[Path]:
    """Validate HF cache metadata and return Arrow paths in state order."""
    require(cache_train_path.is_dir() and not cache_train_path.is_symlink(),
            f"cache train 必须是非符号链接目录: {cache_train_path}")
    state_path = cache_train_path / "state.json"
    dataset_info_path = cache_train_path / "dataset_info.json"
    for metadata_path in (state_path, dataset_info_path):
        require(metadata_path.is_file() and not metadata_path.is_symlink(),
                f"cache metadata 必须是非符号链接普通文件: {metadata_path}")
    state = load_json(state_path)
    data_files = state.get("_data_files")
    require(isinstance(data_files, list) and data_files, f"cache state 未声明 data files: {cache_train_path}")
    arrow_paths: List[Path] = []
    seen_filenames: set[str] = set()
    for item in data_files:
        require(isinstance(item, dict) and isinstance(item.get("filename"), str),
                f"cache state data file 项非法: {cache_train_path}")
        filename = item["filename"]
        require(Path(filename).name == filename
                and "/" not in filename and "\\" not in filename
                and filename.endswith(".arrow"),
                f"cache state data filename 非法: {filename!r}")
        require(filename not in seen_filenames,
                f"cache state data filename 重复: {filename!r}")
        seen_filenames.add(filename)
        data_file = cache_train_path / filename
        require(data_file.is_file() and not data_file.is_symlink()
                and data_file.stat().st_size > 0,
                f"cache Arrow 必须是非符号链接且非空: {data_file}")
        arrow_paths.append(data_file)
    actual_arrow_names = {
        path.name for path in cache_train_path.glob("*.arrow") if path.is_file()
    }
    require(actual_arrow_names == seen_filenames,
            "cache train 中 Arrow 文件集合与 state.json._data_files 不一致")
    entries = list(cache_train_path.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in entries),
            "cache train 只允许非符号链接普通文件")
    expected_names = seen_filenames | {"state.json", "dataset_info.json"}
    require({path.name for path in entries} == expected_names,
            "cache train 文件集合必须恰为 state/dataset_info 与 state 声明的 Arrow")
    dataset_info_features(cache_train_path)
    return arrow_paths


def validate_cleaned_source_manifest(
    manifest_path: os.PathLike[str] | str,
    data_root: os.PathLike[str] | str,
    *,
    verify_shard_content: bool,
    expected_data_pool: str = DATA_POOL_NAME,
) -> Dict[str, Any]:
    """Validate one atomically published, ordered cleaned-source shard set."""
    manifest_file = canonical_path(manifest_path)
    root = canonical_path(data_root)
    require(manifest_file.name == SOURCE_MANIFEST_BASENAME,
            f"source manifest 文件名必须为 {SOURCE_MANIFEST_BASENAME}")
    require_within(manifest_file, root, "source manifest")
    source_dir = manifest_file.parent
    manifest = load_json(manifest_file)
    schema_version = manifest.get("schema_version")
    require(is_strict_int(schema_version) and schema_version == SCHEMA_VERSION,
            f"source manifest.schema_version 必须为 {SCHEMA_VERSION}")
    require(manifest.get("manifest_type") == SOURCE_MANIFEST_TYPE,
            f"source manifest.manifest_type 必须为 {SOURCE_MANIFEST_TYPE}")
    require(manifest.get("data_pool") == expected_data_pool,
            f"source manifest.data_pool 必须为 {expected_data_pool}")
    source_id = manifest.get("source_id")
    require(isinstance(source_id, str) and SOURCE_ID_RE.fullmatch(source_id) is not None,
            "source manifest.source_id 非法")
    require(source_dir.name == source_id,
            "source manifest.source_id 必须与对应来源目录名一致")
    output_records = manifest.get("output_records")
    require(is_strict_int(output_records) and output_records > 0,
            "source manifest.output_records 必须为正整数")
    raw_shards = manifest.get("shards")
    require(isinstance(raw_shards, list) and raw_shards,
            "source manifest.shards 必须为非空 ordered list")

    shard_paths = []
    normalized_shards = []
    seen_paths = set()
    combined_hash = hashlib.sha256()
    total_rows = 0
    for index, item in enumerate(raw_shards):
        label = f"source manifest.shards[{index}]"
        require(isinstance(item, dict), f"{label} 必须为 object")
        require(set(item) == {"path", "rows", "bytes", "sha256"},
                f"{label} 字段必须恰为 path/rows/bytes/sha256")
        relative = item.get("path")
        expected_name = f"part-{index:05d}.jsonl"
        require(isinstance(relative, str) and relative == expected_name,
                f"{label}.path 必须按顺序为 {expected_name}")
        require(relative not in seen_paths, f"source shard path 重复: {relative}")
        seen_paths.add(relative)
        rows = item.get("rows")
        byte_count = item.get("bytes")
        digest = validate_sha256(item.get("sha256"), f"{label}.sha256")
        require(is_strict_int(rows) and rows > 0, f"{label}.rows 必须为正整数")
        require(is_strict_int(byte_count) and byte_count > 0,
                f"{label}.bytes 必须为正整数")
        shard_path = canonical_path(source_dir / relative)
        require(shard_path.parent == source_dir,
                f"{label}.path 必须直接位于对应 source 输出目录")
        require(shard_path.is_file() and shard_path.stat().st_size == byte_count,
                f"{label}.bytes 与文件不一致: {shard_path}")

        if verify_shard_content:
            shard_hash = hashlib.sha256()
            observed_rows = 0
            last_byte = b""
            with shard_path.open("rb") as handle:
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    shard_hash.update(chunk)
                    combined_hash.update(chunk)
                    observed_rows += chunk.count(b"\n")
                    last_byte = chunk[-1:]
            require(last_byte == b"\n", f"{label} 最后一行必须以换行符结束")
            require(shard_hash.hexdigest() == digest,
                    f"{label}.sha256 与文件原始字节不一致")
            require(observed_rows == rows,
                    f"{label}.rows={rows}，实际 JSONL 行数={observed_rows}")

        total_rows += rows
        shard_paths.append(str(shard_path))
        normalized_shards.append(dict(item))

    require(total_rows == output_records,
            "source manifest.output_records 与 sum(shards[].rows) 不一致")
    raw_concat_hash = validate_sha256(
        manifest.get("raw_concatenated_sha256"),
        "source manifest.raw_concatenated_sha256",
    )
    if verify_shard_content:
        require(combined_hash.hexdigest() == raw_concat_hash,
                "source manifest.raw_concatenated_sha256 与 ordered shards 不一致")

    summary_relative = manifest.get("summary_path")
    require(summary_relative == "summary.json", "source manifest.summary_path 必须为 summary.json")
    summary_path = canonical_path(source_dir / summary_relative)
    require(summary_path.parent == source_dir, "summary.json 必须直接位于 source 输出目录")
    summary_sha256 = validate_sha256(
        manifest.get("summary_sha256"), "source manifest.summary_sha256"
    )
    require(sha256_file(summary_path) == summary_sha256,
            "source manifest.summary_sha256 与 summary.json 不一致")
    summary = load_json(summary_path)
    require(summary.get("data_pool") == expected_data_pool,
            f"summary.json data_pool 必须为 {expected_data_pool}")
    require(summary.get("source") == source_id, "summary.json source 与 source manifest 不一致")
    require(summary.get("output_records") == output_records,
            "summary.json output_records 与 source manifest 不一致")
    require(summary.get("shards") == normalized_shards,
            "summary.json shards 与 source manifest 不一致")

    rejects = manifest.get("rejects")
    require(isinstance(rejects, dict), "source manifest.rejects 必须为 object")
    require(set(rejects) == {"path", "rows", "bytes", "sha256"},
            "source manifest.rejects 字段必须恰为 path/rows/bytes/sha256")
    require(rejects.get("path") == "rejects.jsonl",
            "source manifest.rejects.path 必须为 rejects.jsonl")
    reject_rows = rejects.get("rows")
    reject_bytes = rejects.get("bytes")
    reject_sha256 = validate_sha256(
        rejects.get("sha256"), "source manifest.rejects.sha256"
    )
    require(is_strict_int(reject_rows) and reject_rows >= 0,
            "source manifest.rejects.rows 必须为非负整数")
    require(is_strict_int(reject_bytes) and reject_bytes >= 0,
            "source manifest.rejects.bytes 必须为非负整数")
    reject_path = canonical_path(source_dir / "rejects.jsonl")
    require(reject_path.parent == source_dir,
            "rejects.jsonl 必须直接位于 source 输出目录")
    require(reject_path.is_file() and reject_path.stat().st_size == reject_bytes,
            "source manifest.rejects.bytes 与文件不一致")
    if verify_shard_content:
        reject_hash = hashlib.sha256()
        observed_reject_rows = 0
        last_byte = b""
        with reject_path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                reject_hash.update(chunk)
                observed_reject_rows += chunk.count(b"\n")
                last_byte = chunk[-1:]
        require(reject_hash.hexdigest() == reject_sha256,
                "source manifest.rejects.sha256 与文件原始字节不一致")
        require(observed_reject_rows == reject_rows,
                "source manifest.rejects.rows 与文件行数不一致")
        require(reject_rows == 0 or last_byte == b"\n",
                "rejects.jsonl 最后一行必须以换行符结束")
    require(summary.get("rejects") == rejects,
            "summary.json rejects 与 source manifest 不一致")
    require(summary.get("rejected_records") == reject_rows,
            "summary.json rejected_records 与 rejects.rows 不一致")

    success_path = source_dir / "_SUCCESS"
    success = load_json(success_path)
    success_schema = success.get("schema_version")
    require(is_strict_int(success_schema) and success_schema == SCHEMA_VERSION,
            f"source _SUCCESS.schema_version 必须为 {SCHEMA_VERSION}")
    manifest_sha256 = sha256_file(manifest_file)
    require(success.get("manifest_sha256") == manifest_sha256,
            "source _SUCCESS.manifest_sha256 与 source_manifest.json 不一致")
    require(success.get("output_records") == output_records,
            "source _SUCCESS.output_records 与 source manifest 不一致")
    require(success.get("data_pool") == expected_data_pool,
            f"source _SUCCESS.data_pool 必须为 {expected_data_pool}")

    return {
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "source_dir": str(source_dir),
        "source_id": source_id,
        "data_pool": expected_data_pool,
        "output_records": output_records,
        "raw_concatenated_sha256": raw_concat_hash,
        "shards": normalized_shards,
        "shard_paths": shard_paths,
        "rejects": dict(rejects),
        "reject_path": str(reject_path),
    }


def stable_token_id_sha256(items: Mapping[str, int]) -> str:
    payload = json.dumps(dict(sorted(items.items())), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def packing_dataset_sha256(
    cache_manifest_paths: Iterable[os.PathLike[str] | str],
    max_length: int,
    packing_length: int,
    packing_num_proc: int,
    row_selector_sha256: Optional[str] = None,
) -> str:
    """Fingerprint ordered immutable cache manifests and packing geometry."""
    require(is_strict_int(max_length) and max_length > 0, "max_length 必须为正整数")
    require(is_strict_int(packing_length) and packing_length > 0,
            "packing_length 必须为正整数")
    require(is_strict_int(packing_num_proc) and packing_num_proc > 0,
            "packing_num_proc 必须为正整数")
    if row_selector_sha256 is not None:
        validate_sha256(row_selector_sha256, "packing row selector sha256")
    manifest_hashes = []
    for raw_path in cache_manifest_paths:
        path = canonical_path(raw_path)
        require(path.is_file(), f"cache manifest 不是文件: {path}")
        manifest_hashes.append(sha256_file(path))
    require(bool(manifest_hashes), "packing dataset fingerprint 至少需要一个 cache manifest")
    payload = json.dumps(
        {
            "schema_version": PACKING_FINGERPRINT_SCHEMA_VERSION,
            "algorithm": "gam_compact_packing",
            "cache_manifest_sha256s": manifest_hashes,
            "max_length": max_length,
            "packing_length": packing_length,
            "packing_num_proc": packing_num_proc,
            "row_selector_sha256": row_selector_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deep_find_sensitive_keys(value: Any, prefix: str = "") -> Iterable[str]:
    sensitive = re.compile(r"(^|_)(api_?key|access_?token|password|secret|credential)($|_)", re.I)
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            if sensitive.search(key_str):
                yield child_prefix
            yield from deep_find_sensitive_keys(child, child_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from deep_find_sensitive_keys(child, f"{prefix}[{idx}]")
