#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locate lmms-eval result directories using its model-name sanitization rules."""

import os
import re
import sys


def sanitize_model_name(model_name: str, full_path: bool = False) -> str:
    """与 lmms_eval/utils.py: sanitize_model_name 完全一致的实现。"""
    if full_path:
        return re.sub(r'["<>:/\|\\?\*\[\]]+', "__", model_name)
    parts = model_name.split("/")
    last_two = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return re.sub(r'["<>:/\|\\?\*\[\]]+', "__", last_two)


def results_dir(eval_log_root: str, model_path: str) -> str:
    """给定 GAM_EVAL_LOG_ROOT和 model_path，
    返回 samples_*.jsonl / results.json 实际所在目录。"""
    return os.path.join(eval_log_root, "log_eval", sanitize_model_name(model_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "sanitized_name" and len(sys.argv) == 3:
        print(sanitize_model_name(sys.argv[2]))
    elif cmd == "results_dir" and len(sys.argv) == 4:
        print(results_dir(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(1)
