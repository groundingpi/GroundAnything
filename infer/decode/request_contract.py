"""Bridge scalar lmms-eval generation fields to one SGLang custom object."""

from __future__ import annotations

from typing import Any, Mapping


PREFIX = "gam_dlm_decode_"

# lmms-eval's comma-delimited ``--gen_kwargs`` can carry only scalars.  Keep
# this table explicit so no unrelated GAM/VLM generation key is consumed.
SCALAR_FIELDS = frozenset(
    {
        "profile",
        "task",
        "temperature",
        "top_p",
        "top_k",
        "sampling_seed",
        "acceptance_policy",
        "confidence_threshold",
        "entropy_threshold",
        "position_penalty",
        "repetition_mode",
        "repetition_weight",
        "repetition_ngram_size",
        "repetition_window",
        "repetition_block_size",
        "enable_eos_early_stop",
        "eos_require_top1",
        "eos_confidence_threshold",
        "eos_entropy_threshold",
        "eos_stability_steps",
        "expected_cardinality",
        "box_start_token_id",
        "box_end_token_id",
        "coord_token_min_id",
        "coord_token_max_id",
        "coordinates_per_target",
        "block_size",
        "sub_block_size",
        "denoise_steps",
        "legacy_force_fill_all",
    }
)


def pack_decode_custom_params(
    generation_kwargs: Mapping[str, Any],
) -> dict[str, Any] | None:
    packed = {
        key[len(PREFIX) :]: value
        for key, value in generation_kwargs.items()
        if key.startswith(PREFIX) and key[len(PREFIX) :] in SCALAR_FIELDS
    }
    return packed or None


def unpacked_generation_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a decoder profile to scalar DLM-only generation fields."""

    unknown = sorted(set(config) - SCALAR_FIELDS)
    if unknown:
        raise ValueError(f"decoder profile contains non-transport fields: {unknown}")
    return {f"{PREFIX}{key}": value for key, value in config.items() if value is not None}
