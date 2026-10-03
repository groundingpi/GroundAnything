#!/usr/bin/env python3
"""Materialize an immutable HF causal view of a GAM DLM wrapper checkpoint."""

from __future__ import annotations


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


MATERIALIZER_SCHEMA = 2


def patch_groundinganything_config_contract(path: Path) -> None:
    """Expose the language context length on the multimodal outer config.

    SGLang validates ``model_hf_config.max_position_embeddings`` before it
    constructs the external model.  GroundAnything/GroundAnythingVLM historically serialized the
    value only under ``text_config``.  Keep the model code private to this
    immutable causal view and mirror, rather than invent, the exact text value.
    """

    source = path.read_text(encoding="utf-8")
    field_anchor = "    pad_token_id: int | None = None\n\n    def __post_init__(self, **kwargs):"
    field_replacement = (
        "    pad_token_id: int | None = None\n"
        "    # Required by SGLang's outer multimodal config validation.\n"
        "    max_position_embeddings: int | None = None\n\n"
        "    def __post_init__(self, **kwargs):"
    )
    mirror_anchor = "        super().__post_init__(**kwargs)"
    mirror_replacement = (
        "        text_max_positions = getattr(self.text_config, "
        "\"max_position_embeddings\", None)\n"
        "        if self.max_position_embeddings is None and text_max_positions is not None:\n"
        "            self.max_position_embeddings = int(text_max_positions)\n\n"
        "        super().__post_init__(**kwargs)"
    )
    if source.count(field_anchor) != 1 or source.count(mirror_anchor) != 1:
        raise RuntimeError("unexpected GroundAnything GroundAnythingVLMConfig source; refusing an unsafe patch")
    patched = source.replace(field_anchor, field_replacement).replace(
        mirror_anchor, mirror_replacement
    )
    path.write_text(patched, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def materialize(base: Path, checkpoint: Path, output: Path) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = base.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source_weight = checkpoint / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    manifest_path = output / "rlv3_causal_model_manifest.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            value.get("status") == "PASS"
            and value.get("materializer_schema") == MATERIALIZER_SCHEMA
            and Path(str(value.get("base_model"))).resolve() == base
            and Path(str(value.get("wrapper_checkpoint"))).resolve() == checkpoint
            and (output / "model.safetensors").is_file()
        ):
            return {"status": "EXISTING", **value}
        raise RuntimeError(f"existing causal model provenance drift: {output}")
    if output.exists():
        raise FileExistsError(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.mkdir(mode=0o750)
    try:
        excluded = {
            "config.json", "model.safetensors", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
        }
        records: list[dict[str, str]] = []
        for source in sorted(base.iterdir()):
            if not source.is_file() or source.name in excluded:
                continue
            if source.name.startswith(("model-", "rng_state_")):
                continue
            if source.suffix in {".log", ".jsonl", ".db", ".pt", ".pth"}:
                continue
            # This remote config receives a causal-view-only compatibility
            # contract below.  Never hard-link it back to the immutable base.
            if source.name == "configuration_groundinganything.py":
                shutil.copy2(source, staging / source.name)
                mode = "private-copy-patched"
            else:
                mode = link_or_copy(source, staging / source.name)
            records.append({"name": source.name, "mode": mode})

        groundinganything_config_source = staging / "configuration_groundinganything.py"
        if not groundinganything_config_source.is_file():
            raise FileNotFoundError(groundinganything_config_source)
        patch_groundinganything_config_contract(groundinganything_config_source)

        tokenizer_source = (
            checkpoint
            if all((checkpoint / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json"))
            else base
        )
        for name in (
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
        ):
            source = tokenizer_source / name
            if source.is_file():
                shutil.copy2(source, staging / name)
                records.append({"name": name, "mode": "private-copy"})

        state = {}
        with safe_open(str(source_weight), framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            keys = list(source.keys())
            if not keys or any(not key.startswith("base_model.") for key in keys):
                raise RuntimeError("RLV3 causal materializer requires a full GAM wrapper state dict")
            for key in keys:
                state[key.removeprefix("base_model.")] = source.get_tensor(key)
        output_weight = staging / "model.safetensors"
        save_file(
            state,
            str(output_weight),
            metadata={**metadata, "gam_rlv3_view": "causal_base_model", "source_prefix": "base_model."},
        )
        del state

        with safe_open(str(output_weight), framework="pt", device="cpu") as causal:
            vocab_size = int(causal.get_slice("model.language_model.embed_tokens.weight").get_shape()[0])
            if causal.get_slice("lm_head.weight").get_shape()[0] != vocab_size:
                raise RuntimeError("causal embedding/head vocabulary mismatch")

        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        # The HF policy still resolves through ``auto_map.AutoModelForCausalLM``.
        # SGLang, however, selects an external implementation from the
        # architecture name, so advertise only the audited causal GAM adapter.
        config["architectures"] = [
            # The first entry makes verl's generic HF checkpoint exporter use
            # AutoModelForCausalLM.  The remote ``auto_map`` still resolves to
            # GroundAnythingVLMForConditionalGeneration; this alias is metadata only.
            "GroundAnythingVLMForCausalLM",
            # SGLang filters the list for registered architectures and selects
            # this second, externally registered implementation.
            "Fast_dVLMForConditionalGeneration",
        ]
        config["text_config"]["vocab_size"] = vocab_size
        config["text_config"]["use_cache"] = False
        text_max_positions = config["text_config"].get("max_position_embeddings")
        if not isinstance(text_max_positions, int) or text_max_positions <= 0:
            raise RuntimeError(
                f"invalid text max_position_embeddings: {text_max_positions!r}"
            )
        config["max_position_embeddings"] = text_max_positions
        config["use_cache"] = False
        (staging / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Do not instantiate a tokenizer here.  Login nodes may carry an older
        # Transformers ABI than the H800 runtime, and construction can fail on
        # unrelated tokenizer_config schema fields.  The immutable tokenizer
        # JSON is the authoritative serialization: audit the atomic row and the
        # contiguous effective vocabulary directly.
        tokenizer_json = json.loads((staging / "tokenizer.json").read_text(encoding="utf-8"))
        mask_ids = [
            int(item["id"])
            for item in tokenizer_json.get("added_tokens", [])
            if isinstance(item, dict) and item.get("content") == "|<MASK>|"
        ]
        if mask_ids != [vocab_size - 1]:
            raise RuntimeError(
                f"atomic GAM mask row drift: mask_ids={mask_ids} model_vocab={vocab_size}"
            )
        serialized_ids = set()
        model_vocab = tokenizer_json.get("model", {}).get("vocab", {})
        if isinstance(model_vocab, dict):
            serialized_ids.update(int(value) for value in model_vocab.values())
        serialized_ids.update(
            int(item["id"])
            for item in tokenizer_json.get("added_tokens", [])
            if isinstance(item, dict) and "id" in item
        )
        if min(serialized_ids, default=-1) != 0 or max(serialized_ids, default=-1) != vocab_size - 1:
            raise RuntimeError(
                "tokenizer/model vocabulary range drift: "
                f"min={min(serialized_ids, default=-1)} max={max(serialized_ids, default=-1)} "
                f"model={vocab_size}"
            )

        manifest: dict[str, object] = {
            "status": "PASS",
            "materializer_schema": MATERIALIZER_SCHEMA,
            "base_model": str(base),
            "wrapper_checkpoint": str(checkpoint),
            "source_weight": str(source_weight),
            "causal_weight": str(output / "model.safetensors"),
            "source_prefix_removed": "base_model.",
            "tensor_count": len(keys),
            "vocab_size": vocab_size,
            "mask_token_id": int(mask_ids[0]),
            "max_position_embeddings": text_max_positions,
            "max_position_embeddings_source": "text_config.max_position_embeddings",
            "hf_export_architecture": "GroundAnythingVLMForCausalLM",
            "sglang_architecture": "Fast_dVLMForConditionalGeneration",
            "model_sha256": sha256(output_weight),
            "files": records,
            "policy_mode": "causal",
            "training_inference_mode": "SGLang causal; DecodeV4 is evaluation-only",
        }
        (staging / manifest_path.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.base, args.checkpoint, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
