"""Serve the GroundAnything-VLM causal checkpoint through bundled SGLang."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="VLM checkpoint inside this project")
    parser.add_argument("--output", default="outputs/sglang/vlm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--served-model-name", default="groundinganything-vlm")
    args = parser.parse_args()
    model, output = [(ROOT / value).resolve() for value in (args.model, args.output)]
    if any(not path.is_relative_to(ROOT) for path in (model, output)):
        raise ValueError("model and output must remain inside the project")
    if model == output or model in output.parents or output in model.parents:
        raise ValueError("output must not overlap the model checkpoint")

    from models.dependency_contract import verify_dependency

    revision = verify_dependency("sglang")
    vendor = ROOT / "vendor/sglang"
    paths = [str(vendor), str(ROOT / "infer/engines"), str(ROOT)]
    sys.path[:0] = paths
    import sglang

    if not Path(sglang.__file__).resolve().is_relative_to(vendor.resolve()):
        raise RuntimeError("SGLang did not load the bundled custom source")
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from infer.serve_sglang import build_launch

    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    if (config.model_type != "groundinganything_vlm"
            or config.text_config.model_type != "qwen3"
            or config.architectures != ["GroundAnythingVLMForConditionalGeneration"]):
        raise ValueError("expected the GroundAnything-VLM Qwen3 checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    if tokenizer.encode("|<MASK>|", add_special_tokens=False) == [len(tokenizer) - 1]:
        raise ValueError("VLM route received a DLM tokenizer")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    embedding = "model.language_model.embed_tokens.weight"
    head = "lm_head.weight"
    if embedding not in mapping or head not in mapping:
        raise ValueError("missing VLM input or output weights")
    with safe_open(model / mapping[embedding], framework="pt") as weights:
        vocab = weights.get_slice(embedding).get_shape()[0]
    with safe_open(model / mapping[head], framework="pt") as weights:
        head_vocab = weights.get_slice(head).get_shape()[0]
    if vocab != len(tokenizer) or head_vocab != vocab:
        raise ValueError("VLM tokenizer and weight vocabulary differ")

    command, environment, _, _ = build_launch(
        model, output, "causal", -1, vocab,
        tokenizer.convert_tokens_to_ids("<|im_end|>"), paths, os.environ,
        args.host, args.port, args.served_model_name,
    )
    environment["GAM_SGLANG_WEIGHT_PREFIX"] = ""
    environment["GAM_SGLANG_MODEL_KIND"] = "vlm"
    environment["GAM_SGLANG_LOGITS_VOCAB_SIZE"] = str(vocab)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "engine": "sglang",
        "revision": revision,
        "engine_file": sglang.__file__,
        "checkpoint": str(model.relative_to(ROOT)),
        "model_kind": "vlm",
        "decoder": "causal",
        "vocab_size": vocab,
        "command": command,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("sglang", "torch", "transformers", "sgl-kernel", "triton", "flashinfer-python")},
    }
    (output / "engine_runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
