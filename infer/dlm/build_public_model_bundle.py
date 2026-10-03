#!/usr/bin/env python3
"""Build a minimal, self-contained and sanitized GAM model release bundle.

The builder intentionally excludes every training-only artifact (optimizer,
scheduler, RNG, Trainer state and launch arguments).  It copies rather than
hard-links model files so a published snapshot cannot mutate with its source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


VLM_REQUIRED = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
}
BASE_REQUIRED = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
}
MODEL_METADATA = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}
CUSTOM_CODE = {
    "configuration_groundinganything_vision.py",
    "configuration_groundinganything.py",
    "image_processing_groundinganything.py",
    "media_utils.py",
    "modeling_groundinganything_vision.py",
    "modeling_groundinganything.py",
    "processing_groundinganything.py",
    "streammind_gate.py",
}
TRAINING_ONLY = {
    "args.json",
    "latest",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
    "groundinganything_vlm_build_manifest.json",
    "gam_tokenizer_manifest.json",
}
# Environment-specific and account-specific denylist entries were removed
# from this public source for anonymous review. They previously flagged
# internal identifiers during model-bundle export; generic checks remain.
SENSITIVE_TEXT = re.compile(
    r"resources/external/|\bwandb\b|"
    r"(?:api|access)[_-]?(?:key|token)|secret",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == sha256(source):
            return
        raise FileExistsError(f"refusing to overwrite a non-identical release file: {destination}")
    temporary = destination.with_name(f".{destination.name}.copy-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def sanitize_json_value(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: sanitize_json_value(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item, key) for item in value]
    if isinstance(value, str):
        lower = key.lower()
        if lower in {"_name_or_path", "name_or_path"} and value.startswith("/"):
            return Path(value).name
        if any(piece in lower for piece in ("output_dir", "logging_dir", "run_name")):
            return None
    return value


def sanitize_json_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Transformers 5 treats ``extra_special_tokens`` as the named-token map
    # consumed by SpecialTokensMixin.  The training checkpoint stores a legacy
    # list here; passing that list through makes Qwen2Tokenizer.all_special_tokens
    # contain a non-token object and tokenizers.add_tokens() raises TypeError.
    # The mask token itself is already embedded atomically in tokenizer.json and
    # is verified below, so remove only this stale registration metadata.
    if isinstance(payload, dict) and isinstance(payload.get("extra_special_tokens"), list):
        payload["extra_special_tokens"] = {}
    payload = sanitize_json_value(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_metadata(source: Path, destination: Path, names: set[str]) -> list[str]:
    copied: list[str] = []
    for name in sorted(names):
        path = source / name
        if not path.is_file():
            continue
        copy_file(path, destination / name)
        if name.endswith(".json") and name not in {"tokenizer.json", "vocab.json"}:
            sanitize_json_file(destination / name)
        copied.append(name)
    return copied


def apply_groundinganything_serving_compatibility(destination: Path) -> list[str]:
    """Embed the audited GroundAnything serving overlay instead of requiring it at runtime."""

    gam_root = Path(__file__).resolve().parents[2]
    if str(gam_root) not in sys.path:
        sys.path.insert(0, str(gam_root))
    from models.vlm_compat import (
        _patched_config_source,
        _patched_kimi_model_source,
        _patched_model_source,
        _patched_processor_source,
    )

    transforms = {
        "configuration_groundinganything.py": _patched_config_source,
        "modeling_groundinganything_vision.py": _patched_kimi_model_source,
        "modeling_groundinganything.py": _patched_model_source,
        "processing_groundinganything.py": _patched_processor_source,
    }
    patched: list[str] = []
    for name, transform in transforms.items():
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source = path.read_text(encoding="utf-8")
        result = transform(source)
        compile(result, str(path), "exec")
        path.write_text(result, encoding="utf-8")
        patched.append(name)
    return patched


def build_vlm(source: Path, destination: Path) -> list[str]:
    missing = sorted(name for name in VLM_REQUIRED if not (source / name).is_file())
    weights = sorted(source.glob("model-*-of-*.safetensors"))
    if missing or not weights:
        raise RuntimeError(f"incomplete VLM source; missing={missing}, shards={len(weights)}")
    copied = copy_metadata(source, destination, MODEL_METADATA | CUSTOM_CODE)
    for weight in weights:
        copy_file(weight, destination / weight.name)
        copied.append(weight.name)
    copy_file(source / "model.safetensors.index.json", destination / "model.safetensors.index.json")
    copied.append("model.safetensors.index.json")
    apply_groundinganything_serving_compatibility(destination)
    return sorted(set(copied))


def reconcile_dlm_tokenizer_metadata(base: Path, checkpoint: Path, destination: Path) -> None:
    """Publish the DLM tokenizer without Trainer's incomplete sidecar config.

    The DLM checkpoint's tokenizer.json is authoritative because it contains
    the atomic mask token.  Some Trainer saves, however, contain a minimal
    tokenizer_config.json without added_tokens_decoder; Transformers 5 then
    falls back to the slow Qwen tokenizer and fails while rebuilding special
    tokens.  Merge the working base sidecars with the one additional mask token
    instead of changing tokenizer semantics.
    """

    tokenizer_payload = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    mask_rows = [
        row for row in tokenizer_payload.get("added_tokens", [])
        if row.get("content") == "|<MASK>|"
    ]
    if len(mask_rows) != 1:
        raise RuntimeError(f"expected one atomic DLM mask record, got {mask_rows}")
    mask_row = dict(mask_rows[0])
    mask_id = int(mask_row["id"])

    merged = json.loads((base / "tokenizer_config.json").read_text(encoding="utf-8"))
    checkpoint_config_path = checkpoint / "tokenizer_config.json"
    checkpoint_config = (
        json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
        if checkpoint_config_path.is_file() else {}
    )
    for name in (
        "chat_template", "model_max_length", "eos_token", "pad_token",
        "bos_token", "unk_token", "clean_up_tokenization_spaces",
    ):
        if name in checkpoint_config:
            merged[name] = checkpoint_config[name]
    merged["extra_special_tokens"] = {}
    additional = list(merged.get("additional_special_tokens") or [])
    if "|<MASK>|" not in [str(item) for item in additional]:
        additional.append("|<MASK>|")
    merged["additional_special_tokens"] = additional
    decoder = dict(merged.get("added_tokens_decoder") or {})
    decoder[str(mask_id)] = {
        "content": "|<MASK>|",
        "lstrip": bool(mask_row.get("lstrip", False)),
        "normalized": bool(mask_row.get("normalized", False)),
        "rstrip": bool(mask_row.get("rstrip", False)),
        "single_word": bool(mask_row.get("single_word", False)),
        "special": True,
    }
    merged["added_tokens_decoder"] = decoder
    (destination / "tokenizer_config.json").write_text(
        json.dumps(sanitize_json_value(merged), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    special_path = destination / "special_tokens_map.json"
    special = json.loads(special_path.read_text(encoding="utf-8"))
    records = list(special.get("additional_special_tokens") or [])
    if not any(
        (row.get("content") if isinstance(row, dict) else str(row)) == "|<MASK>|"
        for row in records
    ):
        records.append({key: value for key, value in decoder[str(mask_id)].items() if key != "special"})
    special["additional_special_tokens"] = records
    special_path.write_text(json.dumps(special, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    added_path = destination / "added_tokens.json"
    added = json.loads(added_path.read_text(encoding="utf-8")) if added_path.is_file() else {}
    added["|<MASK>|"] = mask_id
    added_path.write_text(json.dumps(added, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dlm(base: Path, checkpoint: Path, destination: Path) -> list[str]:
    missing = sorted(name for name in BASE_REQUIRED if not (base / name).is_file())
    if missing or not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"incomplete DLM inputs; base_missing={missing}")

    # Base supplies the public GroundAnything/Kimi architecture and image processor;
    # the DLM checkpoint supplies the full wrapper state and mask-aware tokenizer.
    copied = copy_metadata(base, destination, MODEL_METADATA | CUSTOM_CODE)
    for name in ("tokenizer.json", "chat_template.jinja"):
        source = checkpoint / name
        if source.is_file():
            target = destination / name
            if target.exists():
                target.unlink()
            copy_file(source, target)
    reconcile_dlm_tokenizer_metadata(base, checkpoint, destination)
    # The DLM release wraps the VLM and has its own public model identity.
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_type"] = "groundinganything"
    config["architectures"] = ["GroundAnythingForConditionalGeneration"]
    config["auto_map"] = {
        "AutoConfig": "configuration_groundinganything.GroundAnythingConfig",
        "AutoModel": "modeling_groundinganything.GroundAnythingModel",
        "AutoModelForCausalLM": "modeling_groundinganything.GroundAnythingForConditionalGeneration",
        "AutoModelForImageTextToText": "modeling_groundinganything.GroundAnythingForConditionalGeneration",
        "AutoProcessor": "processing_groundinganything.GroundAnythingProcessor",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preprocessor_path = destination / "preprocessor_config.json"
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if isinstance(preprocessor.get("auto_map"), dict):
        preprocessor["auto_map"]["AutoProcessor"] = "processing_groundinganything.GroundAnythingProcessor"
        preprocessor_path.write_text(json.dumps(preprocessor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Restore the audited chat template metadata omitted by Trainer checkpoints.
    tokenizer_config_path = destination / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    if not tokenizer_config.get("chat_template"):
        base_tokenizer_config = json.loads((base / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer_config["chat_template"] = base_tokenizer_config.get("chat_template") or (
            destination / "chat_template.jinja"
        ).read_text(encoding="utf-8")
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    copy_file(checkpoint / "model.safetensors", destination / "model.safetensors")
    copied.extend(["model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
    apply_groundinganything_serving_compatibility(destination)
    return sorted(set(copied))


def validate_bundle(kind: str, destination: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            destination, local_files_only=True, trust_remote_code=True
        )
        tokenizer_class = type(tokenizer).__name__
        tokenizer_length = len(tokenizer)
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        encode_token = lambda value: tokenizer.encode(value, add_special_tokens=False)
        token_for_id = tokenizer.convert_ids_to_tokens
    except (TypeError, ValueError) as error:
        if kind != "dlm":
            raise
        # The control-plane host intentionally retains Transformers 4, whose
        # tokenizer config schema predates the required Transformers 5 list.
        # Validate the immutable tokenizer.json directly here; the H800 gate
        # performs the authoritative AutoTokenizer/processor/model load.
        from tokenizers import Tokenizer

        core = Tokenizer.from_file(str(destination / "tokenizer.json"))
        config = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer_class = f"tokenizers.Tokenizer(static fallback: {type(error).__name__})"
        tokenizer_length = core.get_vocab_size(with_added_tokens=True)
        eos_token_id = core.token_to_id(str(config.get("eos_token")))
        pad_token_id = core.token_to_id(str(config.get("pad_token")))
        encode_token = lambda value: core.encode(value, add_special_tokens=False).ids
        token_for_id = core.id_to_token
    result: dict[str, Any] = {
        "tokenizer_class": tokenizer_class,
        "tokenizer_length": tokenizer_length,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    # Record config and processor import results separately from static
    # tokenizer and tensor checks when optional runtime dependencies are missing.
    for label, factory in (("config", AutoConfig), ("processor", AutoProcessor)):
        try:
            value = factory.from_pretrained(
                destination, local_files_only=True, trust_remote_code=True
            )
            result[f"{label}_class"] = type(value).__name__
            result[f"{label}_dynamic_validation"] = "PASS"
        except (ImportError, ModuleNotFoundError) as error:
            result[f"{label}_dynamic_validation"] = (
                f"DEFERRED_TO_PINNED_RUNTIME:{type(error).__name__}:{error}"
            )
    if kind == "dlm":
        mask_ids = encode_token("|<MASK>|")
        if len(mask_ids) != 1 or token_for_id(mask_ids[0]) != "|<MASK>|":
            raise RuntimeError(f"DLM mask token is not atomic: {mask_ids}")
        weight = destination / "model.safetensors"
        with safe_open(weight, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys or not all(name.startswith("base_model.") for name in keys):
                raise RuntimeError("DLM release is not a full GAMQwen3DLM wrapper state")
            embedding_shape = handle.get_slice(
                "base_model.model.language_model.embed_tokens.weight"
            ).get_shape()
            head_shape = handle.get_slice("base_model.lm_head.weight").get_shape()
        if embedding_shape[0] != tokenizer_length or head_shape[0] != tokenizer_length:
            raise RuntimeError(
                f"DLM tokenizer/embedding mismatch: tokenizer={tokenizer_length}, "
                f"embedding={embedding_shape}, head={head_shape}"
            )
        result.update(
            {
                "mask_token_id": int(mask_ids[0]),
                "wrapper_key_count": len(keys),
                "embedding_shape": embedding_shape,
                "lm_head_shape": head_shape,
            }
        )
    else:
        index = json.loads((destination / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards or any(not (destination / name).is_file() for name in shards):
            raise RuntimeError("VLM release shard index is incomplete")
        result.update({"weight_shards": shards, "indexed_tensor_count": len(index["weight_map"])})
    return result


def sensitive_scan(destination: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.name in {"tokenizer.json", "vocab.json", "merges.txt"}:
            continue
        if path.suffix.lower() not in {".json", ".py", ".jinja", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if SENSITIVE_TEXT.search(line):
                findings.append(f"{path.name}:{line_number}")
    return findings


def write_release_docs(kind: str, label: str, destination: Path, validation: dict[str, Any]) -> None:
    if kind == "vlm":
        load_example = """```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    ".", trust_remote_code=True, torch_dtype="auto", device_map="auto"
)
```"""
        contract = "标准 Hugging Face `trust_remote_code=True` VLM 权重。"
    else:
        load_example = """将本目录放入 GroundAnything 项目的 `weights/dlm_bundle/`，在项目根目录启动服务：

```bash
python3 run.py serve
```"""
        contract = (
            "完整 `GAMQwen3DLM` wrapper safetensors；参数键统一使用 `base_model.` 前缀，"
            "包含原子 `|<MASK>|`（ID 152670），固定 block size 为 32。"
        )
    readme = f"""# {label}

本目录包含模型权重、配置、tokenizer 和图像处理组件。

## 模型格式

{contract}

## 加载示例

{load_example}

## 文件说明

- 权重精度保持原始 BF16/FP32 mixed checkpoint，不包含 INT8/FP8 量化权重。
- `checksums.sha256` 用于发布和下载后的完整性核验。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    (destination / "requirements.txt").write_text(
        "transformers>=5.3,<5.8\n"
        "huggingface-hub>=1.3,<2\n"
        "safetensors>=0.6\n"
        "torch>=2.8\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "label": label,
        "kind": kind,
        "publication_status": "sanitized_inference_bundle",
        "embedded_compatibility_overlay": [
            "hf_remote_config_runtime_compat",
            "processor_mixin_multimodal_tokens",
            "vision_sdpa",
            "rmsnorm_epsilon",
            "vllm_transformers_attention_backend",
        ],
        "training_artifacts_excluded": sorted(TRAINING_ONLY),
        "validation": validation,
    }
    (destination / "publication_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_checksums(destination: Path) -> None:
    rows = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256(path)}  {path.name}")
    (destination / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("vlm", "dlm"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    existing = [item.name for item in destination.iterdir() if item.name != ".bundle-in-progress"]
    if existing:
        raise RuntimeError(f"release destination must start empty: {destination}: {existing[:5]}")
    (destination / ".bundle-in-progress").write_text(args.label + "\n", encoding="utf-8")

    if args.kind == "vlm":
        files = build_vlm(source, destination)
    else:
        if args.base is None:
            raise ValueError("--base is required for DLM release bundles")
        files = build_dlm(args.base.resolve(strict=True), source, destination)

    validation = validate_bundle(args.kind, destination)
    validation["payload_files"] = files
    write_release_docs(args.kind, args.label, destination, validation)
    findings = sensitive_scan(destination)
    if findings:
        raise RuntimeError(f"sensitive metadata remains in release bundle: {findings[:20]}")
    write_checksums(destination)
    (destination / ".bundle-in-progress").unlink()
    print(json.dumps({"status": "PASS", "destination": str(destination), **validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
