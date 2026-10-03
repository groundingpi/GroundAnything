#!/usr/bin/env python3
"""Create an immutable GAM-DLM evaluation wrapper from a DARE HF checkpoint."""

from __future__ import annotations


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)


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


def tensor_shapes(paths: list[Path]) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in shapes:
                    raise RuntimeError(f"duplicate safetensor key {key!r}")
                shapes[key] = tuple(stream.get_slice(key).get_shape())
    return shapes


def materialize(hf: Path, template: Path, gate: Path, output: Path) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    hf = hf.resolve(strict=True)
    template = template.resolve(strict=True)
    gate = gate.resolve(strict=True)
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("status") != "PASS" or Path(
        str(gate_payload.get("hf_checkpoint", ""))
    ).resolve() != hf:
        raise RuntimeError("RLV3 formal gate does not authorize this HF checkpoint")

    manifest_path = output / "rlv3_dlm_eval_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "PASS"
            and Path(str(existing.get("source_hf_checkpoint"))).resolve() == hf
            and Path(str(existing.get("formal_gate"))).resolve() == gate
            and (output / "model.safetensors").is_file()
        ):
            return {"status": "EXISTING", **existing}
        raise RuntimeError(f"existing RLV3 eval wrapper provenance drift: {output}")
    if output.exists():
        raise FileExistsError(output)

    source_weights = sorted(hf.glob("*.safetensors"))
    reference_weight = template / "model.safetensors"
    if not source_weights or not reference_weight.is_file():
        raise FileNotFoundError("missing DARE HF or reference wrapper weights")
    source_shapes = tensor_shapes(source_weights)
    reference_shapes = tensor_shapes([reference_weight])
    expected = {
        key.removeprefix("base_model."): shape
        for key, shape in reference_shapes.items()
        if key.startswith("base_model.")
    }
    if len(expected) != len(reference_shapes) or source_shapes != expected:
        missing = sorted(set(expected) - set(source_shapes))[:20]
        extra = sorted(set(source_shapes) - set(expected))[:20]
        mismatched = sorted(
            key
            for key in set(source_shapes) & set(expected)
            if source_shapes[key] != expected[key]
        )[:20]
        raise RuntimeError(
            "DARE HF/GAM wrapper tensor contract drift: "
            f"source={len(source_shapes)} expected={len(expected)} "
            f"missing={missing} extra={extra} shape={mismatched}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.mkdir(mode=0o750)
    try:
        state = {}
        metadata: dict[str, str] = {}
        for path in source_weights:
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                metadata.update(stream.metadata() or {})
                for key in stream.keys():
                    state[f"base_model.{key}"] = stream.get_tensor(key)
        target_weight = staging / "model.safetensors"
        save_file(
            state,
            str(target_weight),
            metadata={
                **metadata,
                "gam_rlv3_view": "dlm_eval_wrapper",
                "target_prefix": "base_model.",
            },
        )
        del state
        if tensor_shapes([target_weight]) != reference_shapes:
            raise RuntimeError("materialized RLV3 DLM wrapper failed exact shape audit")

        files = [{"name": "model.safetensors", "mode": "materialized-prefix"}]
        for name in SIDECARS:
            source = template / name
            if source.is_file():
                files.append({"name": name, "mode": link_or_copy(source, staging / name)})
        for required in ("tokenizer.json", "tokenizer_config.json"):
            if not (staging / required).is_file():
                raise FileNotFoundError(staging / required)

        manifest: dict[str, object] = {
            "status": "PASS",
            "source_hf_checkpoint": str(hf),
            "formal_gate": str(gate),
            "formal_global_step": int(gate_payload["global_step"]),
            "wrapper_template": str(template),
            "tensor_count": len(reference_shapes),
            "source_weight_files": [str(path) for path in source_weights],
            "model_bytes": target_weight.stat().st_size,
            "model_sha256": sha256(target_weight),
            "prefix_added": "base_model.",
            "mode": "DLM",
            "decode": "DecodeV4/task_profiles",
            "files": files,
        }
        (staging / manifest_path.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--wrapper-template", type=Path, required=True)
    parser.add_argument("--formal-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.hf_checkpoint,
                args.wrapper_template,
                args.formal_gate,
                args.output,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
