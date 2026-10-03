#!/usr/bin/env python3
"""DLM-only task generation profiles.

The canonical GAM task table is also used by AR/VLM evaluation and must stay
unchanged.  This module takes a private deep copy inside the DLM SGLang runner,
reuses the compatible GAM sampling contract, and adds a conservative
single-target termination contract only for tasks whose query schema is
strictly one target.  Median target count is never used to infer cardinality.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Any, Mapping

from infer.decode.request_contract import unpacked_generation_fields
from infer.decode.task_profiles import decoder_profile


# Ground-truth output-density taxonomy.  This is deliberately independent of
# the Python representation of GT (notably Point RLE dict/list containers) and
# of the broader benchmark family.  It is metadata for decode analysis/sweep;
# production generation parameters are not silently changed by this table.
DENSE_OUTPUT_TASKS = frozenset(
    {
        "gam_dense200",
        "gam_visdrone",
        "gam_dense200_Labelless",
        "gam_visdrone_Labelless",
        "gam_rex_point_dense200",
        "gam_rex_point_visdrone",
        "gam_sroie",
        "gam_sroie_Boxonly",
        "gam_hiertext",
        "gam_hiertext_Boxonly",
        "gam_fsc147",
        "gam_visual_dense200",
    }
)

MEDIUM_OUTPUT_TASKS = frozenset(
    {
        "gam_lvis",
        "gam_lvis_Labelless",
        "gam_coco",
        "gam_coco_Labelless",
        "gam_m6doc",
        "gam_doclaynet",
        "gam_rex_point_lvis",
        "gam_rex_point_coco",
        "gam_totaltext",
        "gam_totaltext_Boxonly",
        "gam_icdar2015",
        "gam_icdar2015_Boxonly",
        "gam_humanref",
        "gam_rex_point_humanref",
        "gam_visual_lvis",
        "gam_visual_coco",
    }
)

# OCR gets a dedicated Decode V2 route in the four-tier contract.  These are
# all and only the OCR benches in the audited GAM-42 suite.  The old three-tier
# sets above remain unchanged so prior evaluation results stay reproducible.
OCR_OUTPUT_TASKS = frozenset(
    {
        "gam_totaltext",
        "gam_totaltext_Boxonly",
        "gam_icdar2015",
        "gam_icdar2015_Boxonly",
        "gam_sroie",
        "gam_sroie_Boxonly",
        "gam_hiertext",
        "gam_hiertext_Boxonly",
    }
)

# Compatibility generation-budget class used by the already-launched
# Final/2.0 formal evaluation.  Its name predates the exact three-tier audit;
# do not interpret it as the GT-density taxonomy above.  Changing it while a
# formal run is in flight would invalidate the run fingerprint.
HIGH_DENSITY_TASKS = frozenset(
    {
        "gam_coco",
        "gam_lvis",
        "gam_coco_Labelless",
        "gam_lvis_Labelless",
        "gam_dense200",
        "gam_visdrone",
        "gam_dense200_Labelless",
        "gam_visdrone_Labelless",
        "gam_rex_point_coco",
        "gam_rex_point_lvis",
        "gam_rex_point_dense200",
        "gam_rex_point_visdrone",
        "gam_fsc147",
        "gam_visual_coco",
        "gam_visual_dense200",
        "gam_visual_lvis",
    }
)

REFERRING_BOX_TASKS = frozenset(
    {
        "gam_refcocog_val",
        "gam_refcocog_test",
        "gam_refcoco",
        "gam_refcocog",
        "gam_refcocoplus",
    }
)

REFERRING_POINT_TASKS = frozenset(
    {
        "gam_rex_point_refcocog_test",
        "gam_rex_point_refcocog_val",
    }
)

# These datasets have median one target but valid rows with multiple targets
# (HumanRef up to five; Visual Prompt averages roughly 2--3).  They must never
# inherit cardinality=1 merely from their family name or aggregate median.
VARIABLE_CARDINALITY_TASKS = frozenset(
    {
        "gam_humanref",
        "gam_rex_point_humanref",
        "gam_visual_coco",
        "gam_visual_lvis",
    }
)

GUI_POINT_TASKS = frozenset(
    {
        "gam_screenspot_pro",
        "gam_screenspot_v2",
        "gam_osworld_g",
    }
)

ROBO_POINT_TASKS = frozenset(
    {
        "gam_refspatial_location",
        "gam_refspatial_placement",
        "gam_refspatial_unseen",
        "gam_robospatial_context",
    }
)

SINGLE_TARGET_POINT_TASKS = (
    REFERRING_POINT_TASKS | ROBO_POINT_TASKS | GUI_POINT_TASKS
)
SINGLE_TARGET_TASKS = REFERRING_BOX_TASKS | SINGLE_TARGET_POINT_TASKS
# Compatibility name used by older diagnostics and summaries.
REFERRING_SINGLE_TARGET_TASKS = REFERRING_BOX_TASKS | REFERRING_POINT_TASKS

# Four-tier taxonomy: remove OCR from its former medium/dense density buckets,
# while preserving strict query-level cardinality for the single-target tier.
FOUR_TIER_MEDIUM_OUTPUT_TASKS = MEDIUM_OUTPUT_TASKS - OCR_OUTPUT_TASKS
FOUR_TIER_DENSE_OUTPUT_TASKS = DENSE_OUTPUT_TASKS - OCR_OUTPUT_TASKS
FOUR_TIER_TASKS = (
    SINGLE_TARGET_TASKS
    | FOUR_TIER_MEDIUM_OUTPUT_TASKS
    | FOUR_TIER_DENSE_OUTPUT_TASKS
    | OCR_OUTPUT_TASKS
)

# DecodeV4 separates OCR by the same audited target-density boundary used
# before OCR became its own DecodeV3 tier.  This avoids applying ICDAR's
# beneficial sampling temperature to dense SROIE/HierText trajectories.
MEDIUM_OCR_OUTPUT_TASKS = OCR_OUTPUT_TASKS & MEDIUM_OUTPUT_TASKS
DENSE_OCR_OUTPUT_TASKS = OCR_OUTPUT_TASKS & DENSE_OUTPUT_TASKS
FIVE_TIER_MEDIUM_OUTPUT_TASKS = FOUR_TIER_MEDIUM_OUTPUT_TASKS
FIVE_TIER_DENSE_OUTPUT_TASKS = FOUR_TIER_DENSE_OUTPUT_TASKS
FIVE_TIER_TASKS = (
    SINGLE_TARGET_TASKS
    | FIVE_TIER_MEDIUM_OUTPUT_TASKS
    | FIVE_TIER_DENSE_OUTPUT_TASKS
    | MEDIUM_OCR_OUTPUT_TASKS
    | DENSE_OCR_OUTPUT_TASKS
)

if DENSE_OUTPUT_TASKS & MEDIUM_OUTPUT_TASKS:
    raise RuntimeError("dense and medium output task tiers must be disjoint")
if (DENSE_OUTPUT_TASKS | MEDIUM_OUTPUT_TASKS) & SINGLE_TARGET_TASKS:
    raise RuntimeError("strict-single tasks must not overlap multi-target tiers")
if not OCR_OUTPUT_TASKS <= (DENSE_OUTPUT_TASKS | MEDIUM_OUTPUT_TASKS):
    raise RuntimeError("every OCR task must originate in an audited density tier")
_four_tier_sets = (
    SINGLE_TARGET_TASKS,
    FOUR_TIER_MEDIUM_OUTPUT_TASKS,
    FOUR_TIER_DENSE_OUTPUT_TASKS,
    OCR_OUTPUT_TASKS,
)
if any(
    left & right
    for index, left in enumerate(_four_tier_sets)
    for right in _four_tier_sets[index + 1 :]
):
    raise RuntimeError("four-tier GAM task routes must be pairwise disjoint")
if len(FOUR_TIER_TASKS) != 42:
    raise RuntimeError(
        f"four-tier GAM task routes must cover exactly 42 tasks, got "
        f"{len(FOUR_TIER_TASKS)}"
    )
_five_tier_sets = (
    SINGLE_TARGET_TASKS,
    FIVE_TIER_MEDIUM_OUTPUT_TASKS,
    FIVE_TIER_DENSE_OUTPUT_TASKS,
    MEDIUM_OCR_OUTPUT_TASKS,
    DENSE_OCR_OUTPUT_TASKS,
)
if any(
    left & right
    for index, left in enumerate(_five_tier_sets)
    for right in _five_tier_sets[index + 1 :]
):
    raise RuntimeError("five-tier GAM task routes must be pairwise disjoint")
if len(MEDIUM_OCR_OUTPUT_TASKS) != 4 or len(DENSE_OCR_OUTPUT_TASKS) != 4:
    raise RuntimeError("DecodeV4 requires four medium and four dense OCR tasks")
if len(FIVE_TIER_TASKS) != 42:
    raise RuntimeError(
        f"five-tier GAM task routes must cover exactly 42 tasks, got "
        f"{len(FIVE_TIER_TASKS)}"
    )

THREE_TIER_DECODE_PROFILE = "final2_three_tier_v1"
THREE_TIER_PROFILE_NAMES = {
    "strict_single": "final2_tier_single",
    "medium": "final2_tier_medium",
    "dense": "final2_tier_dense",
}

FOUR_TIER_DECODE_PROFILE = "final2_four_tier_v1"
FOUR_TIER_PROFILE_NAMES = {
    "strict_single": "final2_four_tier_single",
    "medium": "final2_four_tier_medium",
    "dense": "final2_four_tier_dense",
    "ocr": "final2_four_tier_ocr",
}

FIVE_TIER_DECODE_PROFILE = "task_profiles"
FIVE_TIER_PROFILE_NAMES = {
    "strict_single": "task_single",
    "medium": "task_medium",
    "dense": "task_dense",
    "ocr_medium": "task_ocr_medium",
    "ocr_dense": "task_ocr_dense",
}


def _three_tier_profile_name(task: str) -> str:
    """Resolve one of the audited 42 GAM routes, failing closed otherwise."""

    matches = [
        THREE_TIER_PROFILE_NAMES["strict_single"]
        if task in SINGLE_TARGET_TASKS
        else None,
        THREE_TIER_PROFILE_NAMES["medium"]
        if task in MEDIUM_OUTPUT_TASKS
        else None,
        THREE_TIER_PROFILE_NAMES["dense"]
        if task in DENSE_OUTPUT_TASKS
        else None,
    ]
    matches = [item for item in matches if item is not None]
    if len(matches) != 1:
        raise ValueError(
            f"{THREE_TIER_DECODE_PROFILE} requires exactly one audited tier "
            f"for task={task!r}; matches={matches}"
        )
    return matches[0]


def _four_tier_profile_name(task: str) -> str:
    """Resolve exactly one audited GAM-42 four-tier route, failing closed."""

    matches = [
        FOUR_TIER_PROFILE_NAMES["strict_single"]
        if task in SINGLE_TARGET_TASKS
        else None,
        FOUR_TIER_PROFILE_NAMES["medium"]
        if task in FOUR_TIER_MEDIUM_OUTPUT_TASKS
        else None,
        FOUR_TIER_PROFILE_NAMES["dense"]
        if task in FOUR_TIER_DENSE_OUTPUT_TASKS
        else None,
        FOUR_TIER_PROFILE_NAMES["ocr"] if task in OCR_OUTPUT_TASKS else None,
    ]
    matches = [item for item in matches if item is not None]
    if len(matches) != 1:
        raise ValueError(
            f"{FOUR_TIER_DECODE_PROFILE} requires exactly one audited tier "
            f"for task={task!r}; matches={matches}"
        )
    return matches[0]


def _five_tier_profile_name(task: str) -> str:
    """Resolve exactly one audited GAM-42 DecodeV4 route, failing closed."""

    matches = [
        FIVE_TIER_PROFILE_NAMES["strict_single"]
        if task in SINGLE_TARGET_TASKS
        else None,
        FIVE_TIER_PROFILE_NAMES["medium"]
        if task in FIVE_TIER_MEDIUM_OUTPUT_TASKS
        else None,
        FIVE_TIER_PROFILE_NAMES["dense"]
        if task in FIVE_TIER_DENSE_OUTPUT_TASKS
        else None,
        FIVE_TIER_PROFILE_NAMES["ocr_medium"]
        if task in MEDIUM_OCR_OUTPUT_TASKS
        else None,
        FIVE_TIER_PROFILE_NAMES["ocr_dense"]
        if task in DENSE_OCR_OUTPUT_TASKS
        else None,
    ]
    matches = [item for item in matches if item is not None]
    if len(matches) != 1:
        raise ValueError(
            f"{FIVE_TIER_DECODE_PROFILE} requires exactly one audited tier "
            f"for task={task!r}; matches={matches}"
        )
    return matches[0]

# Atomic spatial token ids in the StageI-0816 tokenizer.  They are passed as
# an explicit, request-scoped contract and consumed only by the dedicated DLM
# SGLang route.  The server fails closed if these values do not describe a
# valid coordinate interval.
SPATIAL_TOKEN_CONTRACT = {
    "box_start_token_id": 151648,
    "box_end_token_id": 151649,
    "coord_token_min_id": 151669,
    "coord_token_max_id": 152668,
}

HIGH_DENSITY_GENERATION = {
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "until": ["\n\n"],
    "enable_thinking": False,
}

LOW_DENSITY_GENERATION = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "until": ["\n\n"],
    "enable_thinking": False,
}


def _referring_max_tokens() -> int:
    raw = os.environ.get("GAM_DLM_REFERRING_MAX_TOKENS", "512")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GAM_DLM_REFERRING_MAX_TOKENS must be an integer") from exc
    if not 32 <= value <= 4096:
        raise ValueError("GAM_DLM_REFERRING_MAX_TOKENS must be within 32..4096")
    return value


def _single_target_rule_enabled() -> bool:
    raw = os.environ.get("GAM_DLM_SINGLE_TARGET_RULE", "1")
    if raw not in {"0", "1"}:
        raise ValueError("GAM_DLM_SINGLE_TARGET_RULE must be 0 or 1")
    return raw == "1"


def _optional_float(name: str, *, minimum: float, maximum: float) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return value


def _optional_int(name: str, *, minimum: int) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _apply_dlm_sampling_overrides(generation: dict[str, Any]) -> None:
    """Apply explicit DLM-only A/B values without touching GAM/VLM config."""

    temperature = _optional_float(
        "GAM_DLM_TEMPERATURE_OVERRIDE", minimum=0.0, maximum=10.0
    )
    top_p = _optional_float("GAM_DLM_TOP_P_OVERRIDE", minimum=1e-9, maximum=1.0)
    top_k = _optional_int("GAM_DLM_TOP_K_OVERRIDE", minimum=1)
    if temperature is not None:
        generation["temperature"] = temperature
    if top_p is not None:
        generation["top_p"] = top_p
    if top_k is not None:
        generation["top_k"] = top_k


def _decode_profile(task: str) -> dict[str, Any]:
    """Resolve named profile then apply explicit experiment overrides."""

    name = os.environ.get("GAM_DLM_DECODE_PROFILE", "framework")
    raw_profile_map = os.environ.get("GAM_DLM_DECODE_TASK_PROFILE_JSON")
    if raw_profile_map:
        profile_map = json.loads(raw_profile_map)
        if not isinstance(profile_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in profile_map.items()
        ):
            raise ValueError(
                "GAM_DLM_DECODE_TASK_PROFILE_JSON must map task names to profile names"
            )
        name = profile_map.get(task, profile_map.get("*", name))
    if name == THREE_TIER_DECODE_PROFILE:
        name = _three_tier_profile_name(task)
    elif name == FOUR_TIER_DECODE_PROFILE:
        name = _four_tier_profile_name(task)
    elif name == FIVE_TIER_DECODE_PROFILE:
        name = _five_tier_profile_name(task)
    result = decoder_profile(name, task)
    # Existing scalar overrides are explicit request-level choices and must
    # affect both the OpenAI prefill sampler and the DLM token-choice branch.
    temperature = _optional_float(
        "GAM_DLM_TEMPERATURE_OVERRIDE", minimum=0.0, maximum=10.0
    )
    top_p = _optional_float("GAM_DLM_TOP_P_OVERRIDE", minimum=1e-9, maximum=1.0)
    top_k = _optional_int("GAM_DLM_TOP_K_OVERRIDE", minimum=1)
    if temperature is not None:
        result["temperature"] = temperature
    if top_p is not None:
        result["top_p"] = top_p
    if top_k is not None:
        result["top_k"] = top_k

    raw = os.environ.get("GAM_DLM_DECODE_OVERRIDES_JSON")
    if raw:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError("GAM_DLM_DECODE_OVERRIDES_JSON must be an object")
        result.update(overrides)
    raw_task_overrides = os.environ.get("GAM_DLM_DECODE_TASK_OVERRIDES_JSON")
    if raw_task_overrides:
        task_overrides = json.loads(raw_task_overrides)
        if not isinstance(task_overrides, dict):
            raise ValueError(
                "GAM_DLM_DECODE_TASK_OVERRIDES_JSON must be an object"
            )
        for key in ("*", task):
            values = task_overrides.get(key)
            if values is not None:
                if not isinstance(values, dict):
                    raise ValueError(
                        "each GAM_DLM_DECODE_TASK_OVERRIDES_JSON value must be an object"
                    )
                result.update(values)
    return result


def _apply_three_tier_generation_contract(
    task: str, generation: dict[str, Any], decode: Mapping[str, Any]
) -> None:
    """Keep the outer anchor sampler and DLM branch on one tier contract."""

    profile = decode.get("profile")
    if profile not in THREE_TIER_PROFILE_NAMES.values():
        return
    expected = _three_tier_profile_name(task)
    if profile != expected:
        raise ValueError(
            f"three-tier profile mismatch for task={task!r}: "
            f"resolved={profile!r} expected={expected!r}"
        )
    generation.update(
        temperature=float(decode["temperature"]),
        top_p=float(decode["top_p"]),
        repetition_penalty=1.0,
        max_tokens=(
            8192
            if task in DENSE_OUTPUT_TASKS
            else 4096
            if task in MEDIUM_OUTPUT_TASKS
            else _referring_max_tokens()
        ),
    )
    # ``top_k=0`` means disabled inside Decode V2.  The OpenAI-compatible
    # outer sampler represents the same state by omitting the field.
    generation.pop("top_k", None)


def _apply_four_tier_generation_contract(
    task: str, generation: dict[str, Any], decode: Mapping[str, Any]
) -> None:
    """Align the outer anchor sampler with the four-tier DLM token sampler."""

    profile = decode.get("profile")
    if profile not in FOUR_TIER_PROFILE_NAMES.values():
        return
    expected = _four_tier_profile_name(task)
    if profile != expected:
        raise ValueError(
            f"four-tier profile mismatch for task={task!r}: "
            f"resolved={profile!r} expected={expected!r}"
        )
    generation.update(
        temperature=float(decode["temperature"]),
        top_p=float(decode["top_p"]),
        repetition_penalty=1.0,
        max_tokens=(
            8192
            if task in FOUR_TIER_DENSE_OUTPUT_TASKS
            else _referring_max_tokens()
            if task in SINGLE_TARGET_TASKS
            else 4096
        ),
    )
    # Decode V2 uses zero for disabled top-k; the OpenAI-compatible outer
    # request expresses the same contract by omitting the field.
    generation.pop("top_k", None)


def _apply_five_tier_generation_contract(
    task: str, generation: dict[str, Any], decode: Mapping[str, Any]
) -> None:
    """Align outer anchor and DLM token sampling with DecodeV4."""

    profile = decode.get("profile")
    if profile not in FIVE_TIER_PROFILE_NAMES.values():
        return
    expected = _five_tier_profile_name(task)
    if profile != expected:
        raise ValueError(
            f"five-tier profile mismatch for task={task!r}: "
            f"resolved={profile!r} expected={expected!r}"
        )
    generation.update(
        temperature=float(decode["temperature"]),
        top_p=float(decode["top_p"]),
        repetition_penalty=1.0,
        max_tokens=(
            8192
            if task in FIVE_TIER_DENSE_OUTPUT_TASKS
            else _referring_max_tokens()
            if task in SINGLE_TARGET_TASKS
            else 4096
        ),
    )
    generation.pop("top_k", None)


def build_dlm_task_config(
    source: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return an isolated DLM config without mutating the VLM/GAM table."""

    result = deepcopy(dict(source))
    for task, config in result.items():
        inherited = dict(config.get("generation_kwargs") or {})
        explicit_decode_fields = {
            key: value
            for key, value in inherited.items()
            if key.startswith("gam_dlm_decode_")
        }
        profile = (
            HIGH_DENSITY_GENERATION
            if task in HIGH_DENSITY_TASKS
            else LOW_DENSITY_GENERATION
        )
        inherited.update(deepcopy(profile))
        inherited.pop("top_k", None)
        _apply_dlm_sampling_overrides(inherited)

        decode = _decode_profile(task)
        _apply_three_tier_generation_contract(task, inherited, decode)
        _apply_four_tier_generation_contract(task, inherited, decode)
        _apply_five_tier_generation_contract(task, inherited, decode)

        if task in SINGLE_TARGET_TASKS and _single_target_rule_enabled():
            # Referring has exactly one target.  SGLang checks the decoded stop
            # after every token returned by a DLM block, so this preserves the
            # first complete box/point wrapper and discards later B32 loop
            # tokens.  no_stop_trim is required because the native parser needs
            # the closing <|box_end|> token itself.
            inherited["max_tokens"] = _referring_max_tokens()
            inherited["until"] = ["<|box_end|>"]
            inherited["no_stop_trim"] = True
            # Keep every CLI generation value scalar: lmms-eval's
            # ``--gen_kwargs`` parser is comma-delimited and cannot represent
            # nested dictionaries. The async DLM adapter assembles these
            # audited scalars into SGLang's request-scoped custom_params.
            inherited.update(
                gam_dlm_coordinate_limit=(
                    4 if task in REFERRING_BOX_TASKS else 2
                ),
                gam_dlm_box_start_token_id=SPATIAL_TOKEN_CONTRACT[
                    "box_start_token_id"
                ],
                gam_dlm_box_end_token_id=SPATIAL_TOKEN_CONTRACT[
                    "box_end_token_id"
                ],
                gam_dlm_coord_token_min_id=SPATIAL_TOKEN_CONTRACT[
                    "coord_token_min_id"
                ],
                gam_dlm_coord_token_max_id=SPATIAL_TOKEN_CONTRACT[
                    "coord_token_max_id"
                ],
                gam_dlm_expected_cardinality=1,
            )
            decode.update(
                expected_cardinality=1,
                box_start_token_id=SPATIAL_TOKEN_CONTRACT["box_start_token_id"],
                box_end_token_id=SPATIAL_TOKEN_CONTRACT["box_end_token_id"],
                coord_token_min_id=SPATIAL_TOKEN_CONTRACT["coord_token_min_id"],
                coord_token_max_id=SPATIAL_TOKEN_CONTRACT["coord_token_max_id"],
                coordinates_per_target=(4 if task in REFERRING_BOX_TASKS else 2),
            )

        inherited.update(unpacked_generation_fields(decode))
        # A caller that explicitly supplied a decoder scalar wins over the
        # named task profile.  This is the request > profile > server contract.
        inherited.update(explicit_decode_fields)

        config["generation_kwargs"] = inherited
    return result


def profile_summary() -> dict[str, Any]:
    return {
        "output_density_tiers": {
            "dense": sorted(DENSE_OUTPUT_TASKS),
            "medium_or_variable": sorted(MEDIUM_OUTPUT_TASKS),
            "strict_single": sorted(SINGLE_TARGET_TASKS),
        },
        "high_density_tasks": sorted(HIGH_DENSITY_TASKS),
        "high_density_generation": deepcopy(HIGH_DENSITY_GENERATION),
        "low_density_generation": deepcopy(LOW_DENSITY_GENERATION),
        "referring_single_target_tasks": sorted(REFERRING_SINGLE_TARGET_TASKS),
        "gui_single_target_tasks": sorted(GUI_POINT_TASKS),
        "robo_single_target_tasks": sorted(ROBO_POINT_TASKS),
        "single_target_tasks": sorted(SINGLE_TARGET_TASKS),
        "variable_cardinality_tasks": sorted(VARIABLE_CARDINALITY_TASKS),
        "single_target_rule_enabled": _single_target_rule_enabled(),
        "sampling_overrides": {
            "temperature": _optional_float(
                "GAM_DLM_TEMPERATURE_OVERRIDE", minimum=0.0, maximum=10.0
            ),
            "top_p": _optional_float(
                "GAM_DLM_TOP_P_OVERRIDE", minimum=1e-9, maximum=1.0
            ),
            "top_k": _optional_int("GAM_DLM_TOP_K_OVERRIDE", minimum=1),
        },
        "decode_profile": os.environ.get("GAM_DLM_DECODE_PROFILE", "framework"),
        "three_tier_decode_profile": {
            "name": THREE_TIER_DECODE_PROFILE,
            "profile_names": deepcopy(THREE_TIER_PROFILE_NAMES),
        },
        "four_tier_decode_profile": {
            "name": FOUR_TIER_DECODE_PROFILE,
            "profile_names": deepcopy(FOUR_TIER_PROFILE_NAMES),
            "tasks": {
                "strict_single": sorted(SINGLE_TARGET_TASKS),
                "medium": sorted(FOUR_TIER_MEDIUM_OUTPUT_TASKS),
                "dense": sorted(FOUR_TIER_DENSE_OUTPUT_TASKS),
                "ocr": sorted(OCR_OUTPUT_TASKS),
            },
        },
        "five_tier_decode_profile": {
            "name": FIVE_TIER_DECODE_PROFILE,
            "profile_names": deepcopy(FIVE_TIER_PROFILE_NAMES),
            "tasks": {
                "strict_single": sorted(SINGLE_TARGET_TASKS),
                "medium": sorted(FIVE_TIER_MEDIUM_OUTPUT_TASKS),
                "dense": sorted(FIVE_TIER_DENSE_OUTPUT_TASKS),
                "ocr_medium": sorted(MEDIUM_OCR_OUTPUT_TASKS),
                "ocr_dense": sorted(DENSE_OCR_OUTPUT_TASKS),
            },
        },
        "decode_task_profiles": json.loads(
            os.environ.get("GAM_DLM_DECODE_TASK_PROFILE_JSON", "{}")
        ),
        "decode_overrides": json.loads(
            os.environ.get("GAM_DLM_DECODE_OVERRIDES_JSON", "{}")
        ),
        "decode_task_overrides": json.loads(
            os.environ.get("GAM_DLM_DECODE_TASK_OVERRIDES_JSON", "{}")
        ),
        "referring_generation": (
            {
                **deepcopy(LOW_DENSITY_GENERATION),
                "max_tokens": _referring_max_tokens(),
                "until": ["<|box_end|>"],
                "no_stop_trim": True,
                "gam_dlm_coordinate_limit": "4 (bbox) / 2 (point)",
                "gam_dlm_spatial_token_contract": deepcopy(
                    SPATIAL_TOKEN_CONTRACT
                ),
            }
            if _single_target_rule_enabled()
            else deepcopy(LOW_DENSITY_GENERATION)
        ),
    }
