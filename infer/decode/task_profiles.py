"""Named, ablatable GAM decoder profiles.

These are experiment starting points, not claims that WeDLM's calibration
transfers to GAM.  Production remains on the separate ``GAMHierarchyBlock``
route until measured results select a profile.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "acceptance_policy": "confidence",
    "confidence_threshold": 0.99,
    "entropy_threshold": None,
    "position_penalty": 0.0,
    "repetition_mode": "none",
    "repetition_weight": 0.0,
    "repetition_ngram_size": 4,
    "repetition_window": 128,
    "repetition_block_size": 16,
    "enable_eos_early_stop": False,
    "eos_require_top1": True,
    "eos_confidence_threshold": None,
    "eos_entropy_threshold": None,
    "eos_stability_steps": 1,
    "block_size": 32,
    "sub_block_size": 4,
    "denoise_steps": 4,
    "legacy_force_fill_all": False,
}


PROFILES: dict[str, dict[str, Any]] = {
    "legacy": {**BASE, "legacy_force_fill_all": True},
    "framework": dict(BASE),
    "confidence_090": {**BASE, "confidence_threshold": 0.90},
    "confidence_095": {**BASE, "confidence_threshold": 0.95},
    "confidence_097": {**BASE, "confidence_threshold": 0.97},
    # Seed values are intentionally labelled calibration probes.  Thresholds
    # used for final profiles must be selected from GAM telemetry quantiles.
    "entropy_seed": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.4,
    },
    "entropy_position_seed": {
        **BASE,
        "acceptance_policy": "entropy_position",
        "entropy_threshold": 0.4,
        "position_penalty": 0.02,
    },
    "entropy_position_repeat_seed": {
        **BASE,
        "acceptance_policy": "entropy_position_repetition",
        "entropy_threshold": 0.4,
        "position_penalty": 0.02,
        "repetition_mode": "ngram",
        "repetition_weight": 0.25,
    },
    "eos_seed": {
        **BASE,
        "acceptance_policy": "entropy_position_repetition",
        "entropy_threshold": 0.4,
        "position_penalty": 0.02,
        "repetition_mode": "ngram",
        "repetition_weight": 0.25,
        "enable_eos_early_stop": True,
        "eos_stability_steps": 1,
    },
    # Matched task profiles for confidence and entropy acceptance.
    # Structured token selection remains deterministic; raw-logit entropy
    # controls which positions may commit in parallel.
    "final2_tuned_default": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.4,
    },
    "final2_tuned_short": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 3.0,
    },
    # Explicit three-tier Final/2.0 route.  These values govern the true DLM
    # token-choice branch; the GT-density router lives in
    # ``infer.engines.dlm_task_profiles`` so the 42 GAM benches remain mutually
    # exclusive and fail closed. Top-k stays disabled for this mixed task tier.
    "final2_tier_single": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "final2_tier_medium": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "final2_tier_dense": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    # Four-tier successor to ``final2_three_tier_v1``.  OCR is separated
    # because matched sweeps selected T=0.3 for ICDAR2015 while COCO selected
    # T=0.1.  The four profiles are intentionally distinct names so a result
    # manifest can prove which routing contract produced it.
    "final2_four_tier_single": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "final2_four_tier_medium": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "final2_four_tier_dense": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "final2_four_tier_ocr": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    # DecodeV4 keeps the audited strict/medium/dense routing and splits OCR by
    # its real output density.  Matched checkpoint-local sweeps selected
    # stochastic T=0.3 for medium OCR (ICDAR/TotalText), while dense OCR
    # (SROIE/HierText) was best with deterministic token choice.  The two OCR
    # profiles remain explicit so their contract can be audited independently.
    "task_single": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "task_medium": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "task_dense": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "task_ocr_medium": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 0,
        "sampling_seed": 42,
    },
    "task_ocr_dense": {
        **BASE,
        "acceptance_policy": "entropy",
        "entropy_threshold": 0.8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "sampling_seed": 42,
    },
}


def decoder_profile(name: str, task: str) -> dict[str, Any]:
    if name not in PROFILES:
        raise ValueError(f"unknown GAM decoder profile {name!r}; choices={sorted(PROFILES)}")
    result = deepcopy(PROFILES[name])
    result.update(profile=name, task=task)
    return result
