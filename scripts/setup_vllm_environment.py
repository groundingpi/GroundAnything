"""Create an isolated vLLM environment over a matching accelerator platform image."""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config_contract import relative


def environment_plan(venv, platform, index_url="https://pypi.org/simple", root=ROOT):
    target = relative(root, venv)
    if target == root.resolve() or target.exists():
        raise ValueError("use a new project-relative environment directory")
    if platform not in ("gpu", "ppu"):
        raise ValueError("platform must be gpu or ppu")
    python = str(target / "bin/python")
    commands = [
        [sys.executable, "-m", "venv", "--system-site-packages", str(target)],
        [sys.executable, str(root / "scripts/prepare_dependencies.py"), "--apply", "--name", "transformers"],
        [python, "-m", "pip", "install", "--index-url", index_url, "--no-deps", "-r", str(root / "requirements/vllm.txt")],
        [python, "-m", "pip", "install", "--index-url", index_url, "--no-deps", "--no-build-isolation", "-e", str(root / "vendor/transformers")],
    ]
    return target, commands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venv", required=True)
    p.add_argument("--platform", choices=("gpu", "ppu"), required=True)
    p.add_argument("--index-url", default="https://pypi.org/simple")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args(argv)
    target, commands = environment_plan(args.venv, args.platform, args.index_url)
    print(json.dumps({"platform": args.platform, "commands": commands, "apply": args.apply,
                      "inherits_platform_image": True, "gpu_verified": False}, indent=2))
    if not args.apply:
        return
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("vLLM runtime requires Linux Python 3.12")
    version = metadata.version("vllm")
    if not version.startswith("0.18.") or ("ppu" in version.lower()) != (args.platform == "ppu"):
        raise ValueError(f"base-image vLLM {version} does not match {args.platform} 0.18.x")
    native_paths = {}
    if args.platform == "ppu":
        base = Path(metadata.distribution("acext").locate_file("")).resolve()
        for name in ("lib", "include"):
            if not (base / name).is_dir():
                raise ValueError(f"PPU image is missing ACEXT {name}")
            native_paths[name] = base / name
    env = dict(os.environ)
    env.pop("PIP_CONSTRAINT", None)
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env["PIP_CONFIG_FILE"] = os.devnull
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    # ACEXT looks for native libraries under the active interpreter's purelib.
    for name, source in native_paths.items():
        (target / "lib/python3.12/site-packages" / name).symlink_to(source, target_is_directory=True)
    python = str(target / "bin/python")
    probe = subprocess.run([python, "-c",
        "import pathlib, torch, transformers, vllm; "
        "source=pathlib.Path(transformers.__file__).resolve(); "
        f"expected=pathlib.Path({str(ROOT / 'vendor/transformers/src')!r}).resolve(); "
        "assert transformers.__version__ == '5.7.0' and source.is_relative_to(expected), "
        "'serving requires the bundled Transformers 5.7.0 fork'; "
        "print(torch.__version__, transformers.__version__, source, vllm.__version__)"],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    check = subprocess.run([python, "-m", "pip", "check"], cwd=ROOT, env=env, capture_output=True, text=True)
    report = {"platform": args.platform, "vllm": version, "import_probe": probe.stdout,
              "inherited_container": True, "pip_check_exit": check.returncode,
              "pip_check": check.stdout + check.stderr, "gpu_verified": False}
    (target / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    freeze = subprocess.run([python, "-m", "pip", "freeze", "--all"], cwd=ROOT, env=env,
                            check=True, capture_output=True, text=True)
    (target / "installed.freeze.txt").write_text(freeze.stdout)
    print(probe.stdout)
    if check.returncode:
        print("Package metadata conflicts are recorded in runtime.json; review them before deployment.")


if __name__ == "__main__":
    main()
