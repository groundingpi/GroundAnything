"""
薄垫片：lmms_eval 的 `!function utils.xxx` 要求 utils.py 与引用它的 yaml 物理同目录，
真正的实现维护在 eval/utils/ 下（Referring/Grounding/Dense 三个类别共用
detection_utils.py；Referring 额外需要 refcoco_utils.py 里 lmms-lab RefCOCO 系列的实现）。
改一处、多处同时生效，避免维护多份相同代码。
"""
import os
import sys

_SHARED_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

from detection_utils import *  # noqa: F401,F403
from refcoco_utils import *  # noqa: F401,F403
