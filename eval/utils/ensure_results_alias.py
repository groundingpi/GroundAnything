#!/usr/bin/env python3
"""Attach a canonical checkpoint result path to one legacy fingerprint path.

Older GroundAnything evaluation used a fingerprint directory as ``model_version``.
The resumed canonical checkpoint path therefore resolves to a different lmms-eval
directory even though both names identify the same checkpoint.  This helper only
creates an atomic relative symlink when the canonical path is absent and exactly
one non-empty legacy directory has the same ``checkpoint-N`` suffix.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from gam_paths import results_dir


def ensure_alias(eval_log_root: Path, model_path: Path) -> tuple[Path, Path | None]:
    canonical = Path(results_dir(str(eval_log_root), str(model_path)))
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.exists() or canonical.is_symlink():
        return canonical, None

    match = re.fullmatch(r"checkpoint-(\d+)", model_path.name)
    if match is None:
        return canonical, None

    suffix = f"__checkpoint-{match.group(1)}"
    candidates = []
    for candidate in canonical.parent.iterdir():
        if candidate == canonical or not candidate.name.endswith(suffix):
            continue
        if not candidate.is_dir():
            continue
        has_results = any(candidate.glob("*_results.json"))
        has_samples = any(candidate.glob("*_samples_*.jsonl"))
        if has_results and has_samples:
            candidates.append(candidate)
    if not candidates:
        return canonical, None
    if len(candidates) != 1:
        names = ", ".join(sorted(path.name for path in candidates))
        raise RuntimeError(
            f"ambiguous legacy result directories for {model_path.name}: {names}"
        )

    legacy = candidates[0]
    temporary = canonical.with_name(
        f".{canonical.name}.alias.{os.getpid()}"
    )
    os.symlink(legacy.name, temporary)
    try:
        os.replace(temporary, canonical)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return canonical, legacy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_log_root", type=Path)
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    canonical, legacy = ensure_alias(args.eval_log_root, args.model_path)
    if legacy is None:
        print(canonical)
    else:
        print(f"{canonical} -> {legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
