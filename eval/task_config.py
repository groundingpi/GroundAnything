#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-task CPU allocation, image limits and generation settings.

Task identifiers use the gam_ prefix to avoid collisions with built-in
lmms-eval tasks. Add matching task YAML and dataset paths for new tasks."""

# Generation settings for dense and other tasks.
#   高密度生成类别：max_tokens=8192, temperature=0.2, top_p=0.9,
#                    repetition_penalty=1.05
#   其余类别：      max_tokens=4096, temperature=0.0, top_p=1.0,
#                    repetition_penalty=1.0
#   top_k 对全部类别禁用。这里刻意不写 top_k；请求中缺省即 None/不截断，
#   避免向 OpenAI-compatible server 发送可能不被接受的 JSON null。
# vLLM 的采样链为 repetition penalty -> temperature -> top-k -> top-p ->
# sample/argmax；temperature=0 时走 greedy/argmax。

# ===== Referring / Grounding / Dense =====
# 像素范围对齐检测类任务推理默认值: min=16*28*28=12544, max=2560*28*28=2007040
_TEXT_STOP = ["\n\n"]

_LOW_DENSITY_GEN = {
    "repetition_penalty": 1.0,
    "top_p": 1.0,
    "temperature": 0.0,
    "max_tokens": 4096,
    "until": list(_TEXT_STOP),
    "enable_thinking": False,
}

_HIGH_DENSITY_GEN = {
    "repetition_penalty": 1.05,
    "top_p": 0.9,
    "temperature": 0.2,
    "max_tokens": 8192,
    "until": list(_TEXT_STOP),
    "enable_thinking": False,
}


def _task_cfg(cpu_num, pixel_range, generation_kwargs, **generation_overrides):
    """Build an independent config so per-task adjustments cannot leak."""

    generation = dict(generation_kwargs)
    generation.update(generation_overrides)
    if isinstance(generation.get("until"), list):
        generation["until"] = list(generation["until"])
    return {
        "cpu_num": cpu_num,
        "task_pixel_min_max": list(pixel_range),
        "generation_kwargs": generation,
    }


_DETECTION_PIXELS = [12544, 2007040]
_POINTING_PIXELS = [100352, 4000014080]
_REFERRING_CFG = _task_cfg(8, _DETECTION_PIXELS, _LOW_DENSITY_GEN)
_GROUNDING_CFG = _task_cfg(8, _DETECTION_PIXELS, _HIGH_DENSITY_GEN)
_DENSE_CFG = _task_cfg(8, _DETECTION_PIXELS, _HIGH_DENSITY_GEN)

# ===== Point-in-mask：Robo / Referring / Grounding / Dense =====
_ROBO_POINT_CFG = _task_cfg(2, _POINTING_PIXELS, _LOW_DENSITY_GEN)
_REFERRING_POINT_CFG = _task_cfg(2, _POINTING_PIXELS, _LOW_DENSITY_GEN)
_GROUNDING_POINT_CFG = _task_cfg(2, _POINTING_PIXELS, _HIGH_DENSITY_GEN)
_DENSE_POINT_CFG = _task_cfg(2, _POINTING_PIXELS, _HIGH_DENSITY_GEN)

# ===== GUI: ScreenSpot-Pro / ScreenSpot-v2 / OSWorld-G =====
_GUI_CFG = _task_cfg(
    2,
    _POINTING_PIXELS,
    _LOW_DENSITY_GEN,
    num_beams=1,
)
_OSWORLD_CFG = {
    "cpu_num": _GUI_CFG["cpu_num"],
    "task_pixel_min_max": list(_GUI_CFG["task_pixel_min_max"]),
    "generation_kwargs": {
        **_GUI_CFG["generation_kwargs"],
        # Qwen may answer with native box/coordinate AddedToken objects.  Keep
        # them for the OSWorld-specific multi-protocol parser.
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    },
}

# ===== OCR: HierText / ICDAR2015 / TotalText / SROIE scene-text E2E =====
# Dense pages can produce long transcription+box streams.  Keep the reference
# scene-text E2E cap (4096); limited-sample truncation remains controlled by
# GAM_EVAL_SMOKE_MAX_TOKENS.
_OCR_CFG = _task_cfg(
    4,
    _DETECTION_PIXELS,
    _LOW_DENSITY_GEN,
    until=[],
        # Qwen-2B can answer OCR with native object-ref/box coordinate tokens
        # even in VLM mode.  Dropping special tokens silently erases all boxes.
    skip_special_tokens=False,
    spaces_between_special_tokens=False,
)

# ===== Layout: DocLayNet / M6Doc =====
_LAYOUT_CFG = _task_cfg(4, _DETECTION_PIXELS, _LOW_DENSITY_GEN)

# ===== VisualPrompt: FSC147 + Rex-Omni visual_prompt_eval =====
_VISUAL_PROMPT_CFG = _task_cfg(4, _DETECTION_PIXELS, _HIGH_DENSITY_GEN)

qwen3vl_task_config = {
    # ---- Referring ----
    "gam_humanref": dict(_REFERRING_CFG),
    "gam_refcocog_val": dict(_REFERRING_CFG),
    "gam_refcocog_test": dict(_REFERRING_CFG),
    "gam_refcoco": dict(_REFERRING_CFG),
    "gam_refcocog": dict(_REFERRING_CFG),
    "gam_refcocoplus": dict(_REFERRING_CFG),
    # ---- Grounding ----
    "gam_coco": dict(_GROUNDING_CFG),
    "gam_lvis": dict(_GROUNDING_CFG),
    "gam_coco_Labelless": dict(_GROUNDING_CFG),
    "gam_lvis_Labelless": dict(_GROUNDING_CFG),
    # ---- Dense ----
    "gam_dense200": dict(_DENSE_CFG),
    "gam_visdrone": dict(_DENSE_CFG),
    "gam_dense200_Labelless": dict(_DENSE_CFG),
    "gam_visdrone_Labelless": dict(_DENSE_CFG),
    # ---- Point-in-mask / Refer point（非 Rex） ----
    "gam_refspatial_location": dict(_ROBO_POINT_CFG),
    "gam_refspatial_placement": dict(_ROBO_POINT_CFG),
    "gam_refspatial_unseen": dict(_ROBO_POINT_CFG),
    "gam_robospatial_context": dict(_ROBO_POINT_CFG),
    # ---- Referring Point-in-mask ----
    "gam_rex_point_humanref": dict(_REFERRING_POINT_CFG),
    "gam_rex_point_refcocog_test": dict(_REFERRING_POINT_CFG),
    "gam_rex_point_refcocog_val": dict(_REFERRING_POINT_CFG),
    # ---- Grounding Point-in-mask ----
    "gam_rex_point_coco": dict(_GROUNDING_POINT_CFG),
    "gam_rex_point_lvis": dict(_GROUNDING_POINT_CFG),
    # ---- Dense Point-in-mask ----
    "gam_rex_point_dense200": dict(_DENSE_POINT_CFG),
    "gam_rex_point_visdrone": dict(_DENSE_POINT_CFG),
    # ---- GUI ----
    "gam_screenspot_pro": dict(_GUI_CFG),
    "gam_screenspot_v2": dict(_GUI_CFG),
    "gam_osworld_g": dict(_OSWORLD_CFG),
    # ---- OCR ----
    "gam_hiertext": dict(_OCR_CFG),
    "gam_icdar2015": dict(_OCR_CFG),
    "gam_totaltext": dict(_OCR_CFG),
    "gam_sroie": dict(_OCR_CFG),
    "gam_hiertext_Boxonly": dict(_OCR_CFG),
    "gam_icdar2015_Boxonly": dict(_OCR_CFG),
    "gam_totaltext_Boxonly": dict(_OCR_CFG),
    "gam_sroie_Boxonly": dict(_OCR_CFG),
    # ---- Layout ----
    "gam_doclaynet": dict(_LAYOUT_CFG),
    "gam_m6doc": dict(_LAYOUT_CFG),
    # ---- Visual prompt ----
    "gam_fsc147": dict(_VISUAL_PROMPT_CFG),
    "gam_visual_coco": dict(_VISUAL_PROMPT_CFG),
    "gam_visual_dense200": dict(_VISUAL_PROMPT_CFG),
    "gam_visual_lvis": dict(_VISUAL_PROMPT_CFG),
}

# 兜底参数（未在 qwen3vl_task_config 里显式配置的任务，走这里的默认值）
DEFAULT_CPU_NUM = 1
DEFAULT_TASK_PIXEL_MIN_MAX = [1, 40014080]
DEFAULT_GENERATION_KWARGS = {
    "repetition_penalty": 1.0,
    "top_p": 1.0,
    "temperature": 0.0,
    "max_tokens": 4096,
    "until": list(_TEXT_STOP),
    "enable_thinking": False,
}
DEFAULT_TASK_CONFIG = {t: dict(cfg) for t, cfg in qwen3vl_task_config.items()}

# Native VLM baselines share the same evaluation workload and decoding contract.
# Coordinate conversion remains launcher-scoped through GAM_COORD_MODE; adding
# a model type here does not alter prompts or metrics.
MODELS_TASK_CONFIG = {
    "qwen25vl": qwen3vl_task_config,
    "qwen3vl": qwen3vl_task_config,
    "qwen35": qwen3vl_task_config,
    "deepseekvl2": qwen3vl_task_config,
}
