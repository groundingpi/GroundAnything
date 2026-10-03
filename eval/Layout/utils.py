"""Rex-Omni DocLayNet/M6Doc layout evaluation glue."""

from __future__ import annotations

import importlib.util
import os
import sys


_EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_UTILS_DIR = os.path.join(_EVAL_ROOT, "utils")
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

import detection_utils as _detection  # noqa: E402
from detection_utils import *  # noqa: E402,F401,F403
from prompt_mode import build_mode_layout_prompt, is_native_spatial_mode  # noqa: E402

_PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "_gam_layout_prompts", os.path.join(os.path.dirname(__file__), "prompts.py")
)
if _PROMPTS_SPEC is None or _PROMPTS_SPEC.loader is None:
    raise ImportError("cannot load Layout/prompts.py")
_prompts = importlib.util.module_from_spec(_PROMPTS_SPEC)
_PROMPTS_SPEC.loader.exec_module(_prompts)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if is_native_spatial_mode():
        return build_mode_layout_prompt(_detection._categories(doc))
    return _prompts.layout_prompt(
        _detection._categories(doc), gam_mode=False
    )
