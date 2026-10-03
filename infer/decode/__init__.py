"""GAM DLM decoding policies.

The package is intentionally independent from the training implementation.
Pure policy modules can be tested on CPU; :mod:`sglang_algorithm` is imported
only by the dedicated SGLang route.
"""

from .config import DecodeConfig

__all__ = ["DecodeConfig"]
