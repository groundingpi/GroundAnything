"""
薄垫片：lmms_eval 的 `!function utils.xxx` 要求 utils.py 与引用它的 yaml 物理同目录，
真正的实现维护在 eval/utils/detection_utils.py（Referring/Grounding/Dense 三个
类别共用，改一处、三处同时生效，避免维护三份相同代码）。
"""
import os
import sys

_SHARED_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

from detection_utils import *  # noqa: F401,F403
