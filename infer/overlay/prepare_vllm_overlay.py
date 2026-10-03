"""Create a source-only compatibility overlay for GroundAnything-VLM serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from models import vlm_compat

ARCHITECTURE = "GroundAnythingVLMForConditionalGeneration"
PATCHERS = {
    "configuration_groundinganything.py": vlm_compat._patched_config_source,
    "modeling_groundinganything_vision.py": vlm_compat._patched_kimi_model_source,
    "modeling_groundinganything.py": vlm_compat._patched_model_source,
    "processing_groundinganything.py": vlm_compat._patched_processor_source,
}
SERVING_SUFFIXES = {".jinja", ".json", ".model", ".py", ".safetensors", ".tiktoken", ".txt"}
EXCLUDED_FILES = {"args.json", "trainer_state.json", "zero_to_fp32.py"}


def source_files(model_dir: Path, root: Path):
    """Validate the VLM checkpoint without importing Torch or loading tensors."""
    model_dir = model_dir.resolve(strict=True)
    if not model_dir.is_relative_to(root.resolve()):
        raise ValueError("model must be inside the project root")
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("architectures") != [ARCHITECTURE]
            or config.get("model_type") != "groundinganything_vlm"
            or config.get("text_config", {}).get("model_type") != "qwen3"):
        raise ValueError("vLLM requires GroundAnything-VLM with plain Qwen3; DLM checkpoints are not supported")
    tokenizer = json.loads((model_dir / "tokenizer_config.json").read_text())
    if any(token.get("content") == "|<MASK>|"
           for token in tokenizer.get("added_tokens_decoder", {}).values() if isinstance(token, dict)):
        raise ValueError("vLLM VLM route received a DLM tokenizer")
    required = {"config.json", *PATCHERS, "image_processing_groundinganything.py", "media_utils.py",
                "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"}
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("weight_map must be nonempty")
        shards = list(mapping.values())
        if any(not isinstance(s, str) or Path(s).name != s or "\\" in s
               or not s.endswith(".safetensors") for s in shards):
            raise ValueError("weight shards must be safetensors filenames inside the model directory")
        required |= set(shards) | {"model.safetensors.index.json"}
    else:
        required.add("model.safetensors")
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"source model is incomplete: {missing}")
    sources = [p for p in sorted(model_dir.iterdir()) if p.is_file()
               and p.name not in EXCLUDED_FILES and p.suffix in SERVING_SUFFIXES]
    if any(not p.resolve().is_relative_to(root.resolve()) for p in sources):
        raise ValueError("source file symlink escapes project")
    return sources, sorted(name for name in required if name.endswith(".safetensors"))


def create_overlay(model_dir: Path, output_dir: Path, root: Path = PROJECT_ROOT):
    root = root.resolve()
    model_dir = model_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("overlay must be inside the project root")
    if output_dir.is_relative_to(model_dir) or model_dir.is_relative_to(output_dir):
        raise ValueError("model and overlay directories must not contain one another")
    if output_dir.exists():
        raise FileExistsError(f"use a new overlay directory: {output_dir}")
    sources, weights = source_files(model_dir, root)
    patched = {name: patcher((model_dir / name).read_text(encoding="utf-8"))
               for name, patcher in PATCHERS.items()}
    manifest = {
        "schema_version": 1, "architecture": ARCHITECTURE,
        "source_model": model_dir.relative_to(root).as_posix(), "path_base": "project_root",
        "weight_files": weights,
        "linked_files": [p.name for p in sources if p.name not in patched],
        "source_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        "compatibility_adapter": "models/vlm_compat.py",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".vllm-overlay-", dir=output_dir.parent))
    try:
        for source in sources:
            target = temporary / source.name
            if source.name in patched:
                target.write_text(patched[source.name], encoding="utf-8")
            else:
                os.symlink(os.path.relpath(source, output_dir), target)
        (temporary / "vllm_overlay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            if temporary.resolve().parent != output_dir.parent.resolve() or not temporary.resolve().is_relative_to(root):
                raise RuntimeError("refusing cleanup outside the overlay parent")
            shutil.rmtree(temporary)
    return manifest


def verify_overlay(model_dir: Path, output_dir: Path, root: Path = PROJECT_ROOT):
    root, model_dir, output_dir = root.resolve(), model_dir.resolve(), output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("overlay must remain inside the project root")
    sources, weights = source_files(model_dir, root)
    manifest = json.loads((output_dir / "vllm_overlay_manifest.json").read_text())
    if (manifest.get("architecture") != ARCHITECTURE
            or manifest.get("source_model") != model_dir.relative_to(root).as_posix()
            or manifest.get("weight_files") != weights
            or manifest.get("source_config_sha256") != hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()):
        raise ValueError("overlay does not match the selected VLM checkpoint; use a new overlay")
    linked = sorted(p.name for p in sources if p.name not in PATCHERS)
    if sorted(manifest.get("linked_files", [])) != linked:
        raise ValueError("overlay source file set changed; use a new overlay")
    for name, patcher in PATCHERS.items():
        if (output_dir / name).read_text(encoding="utf-8") != patcher((model_dir / name).read_text(encoding="utf-8")):
            raise ValueError(f"stale or modified serving adapter: {name}; use a new overlay")
    for name in linked:
        if not (output_dir / name).is_symlink() or (output_dir / name).resolve() != (model_dir / name).resolve():
            raise ValueError(f"overlay does not reference the selected checkpoint file: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = create_overlay(PROJECT_ROOT / args.model, PROJECT_ROOT / args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
