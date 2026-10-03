"""FSC147 shim: reuse the shared visual-prompt detection implementation."""

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[2] / "utils"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from detection_utils import *  # noqa: F401,F403
