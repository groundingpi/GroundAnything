"""Serve the autoregressive GroundAnything-VLM checkpoint with platform vLLM."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser_for():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--overlay", default="outputs/vllm-vlm")
    p.add_argument("--platform", choices=("ppu", "gpu"), required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--served-model-name", default="groundinganything-vlm")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, choices=(1,), default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--tensor-parallel-size", type=int, choices=(1,), default=1)
    return p


def build_command(args, overlay, python=sys.executable):
    if not 1 <= args.port <= 65535 or args.max_model_len < 1:
        raise ValueError("port and max-model-len must be positive and within range")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    if args.max_num_seqs != 1 or args.tensor_parallel_size != 1:
        raise ValueError("this VLM serving profile requires max-num-seqs=1 and tensor-parallel-size=1")
    return [python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(overlay), "--served-model-name", args.served_model_name,
            "--host", args.host, "--port", str(args.port), "--trust-remote-code",
            "--model-impl", "transformers", "--dtype", "bfloat16", "--enforce-eager",
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--limit-mm-per-prompt", '{"image":1,"video":0}']


def main():
    args = parser_for().parse_args()
    from scripts.config_contract import relative
    model, overlay = [relative(ROOT, value) for value in (args.model, args.overlay)]
    command = build_command(args, overlay)
    # Reject a DLM model before attempting accelerator imports or making an overlay.
    from infer.overlay.prepare_vllm_overlay import source_files, create_overlay, verify_overlay
    source_files(model, ROOT)
    import torch
    import transformers
    import vllm
    from models.dependency_contract import verify_dependency
    revision = verify_dependency("transformers", ROOT)
    source = Path(transformers.__file__).resolve()
    if transformers.__version__ != "5.7.0" or not source.is_relative_to((ROOT / "vendor/transformers/src").resolve()):
        raise RuntimeError("vLLM requires the bundled Transformers 5.7.0 fork; run setup vllm for this platform")
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.18.") or ("ppu" in version.lower()) != (args.platform == "ppu"):
        raise RuntimeError(f"vLLM {version} does not match the {args.platform} 0.18.x platform environment")
    if not torch.cuda.is_available():
        raise RuntimeError("no accelerator available")
    device = torch.cuda.get_device_name(0)
    if ("PPU" in device.upper()) != (args.platform == "ppu"):
        raise RuntimeError(f"device {device} does not match platform {args.platform}")
    if not overlay.exists():
        create_overlay(model, overlay)
    manifest = verify_overlay(model, overlay)
    evidence = {
        "engine": "vllm", "engine_version": version, "engine_file": vllm.__file__,
        "model_impl": "transformers", "adapter": "models/vlm_compat.py",
        "adapter_sha256": hashlib.sha256((ROOT / "models/vlm_compat.py").read_bytes()).hexdigest(),
        "platform": args.platform, "device": device, "torch": torch.__version__,
        "transformers": transformers.__version__, "transformers_file": str(source),
        "transformers_revision": revision, "checkpoint": manifest["source_model"],
        "command": command, "generation_validation": "not_run_by_launcher",
    }
    (overlay / "engine_runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
