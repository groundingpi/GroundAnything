"""Validated configuration for the GAM B32 decoder.

Precedence is implemented by merging the server YAML first and the request's
``custom_params.gam_dlm_decode`` mapping second.  Task profiles are converted
to that request mapping by ``request_contract.py``; an explicit eval override
therefore naturally wins over the task profile before the request is sent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Any, Mapping


ACCEPTANCE_POLICIES = frozenset(
    {
        "confidence",
        "entropy",
        "entropy_position",
        "entropy_position_repetition",
    }
)
REPETITION_MODES = frozenset({"none", "ngram", "block"})


@dataclass(frozen=True)
class DecodeConfig:
    block_size: int = 32
    sub_block_size: int = 4
    denoise_steps: int = 4
    token_shift: int = 1

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    sampling_seed: int | None = None

    acceptance_policy: str = "confidence"
    confidence_threshold: float = 0.99
    entropy_threshold: float | None = None
    position_penalty: float = 0.0

    repetition_mode: str = "none"
    repetition_weight: float = 0.0
    repetition_ngram_size: int = 4
    repetition_window: int = 128
    repetition_block_size: int = 16
    repetition_ignore_token_ids: tuple[int, ...] = ()

    enable_eos_early_stop: bool = False
    eos_token_id: int | None = None
    eos_require_top1: bool = True
    eos_confidence_threshold: float | None = None
    eos_entropy_threshold: float | None = None
    eos_stability_steps: int = 1

    expected_cardinality: int | None = None
    box_start_token_id: int | None = None
    box_end_token_id: int | None = None
    coord_token_min_id: int | None = None
    coord_token_max_id: int | None = None
    coordinates_per_target: int | None = None

    # Used only by the explicit ``legacy`` experiment profile.  New policies
    # always drain one position at a time and never fill all residual masks in
    # one operation.
    legacy_force_fill_all: bool = False
    task: str | None = None
    profile: str | None = None

    @classmethod
    def from_mappings(
        cls,
        server: Mapping[str, Any] | None,
        request: Mapping[str, Any] | None = None,
        *,
        eos_token_id: int | None = None,
    ) -> "DecodeConfig":
        valid = {item.name for item in fields(cls)}
        merged: dict[str, Any] = {}
        for source in (server or {}, request or {}):
            source = dict(source)
            # Existing GAM YAMLs call the confidence threshold ``threshold``.
            if "threshold" in source and "confidence_threshold" not in source:
                source["confidence_threshold"] = source.pop("threshold")
            unknown = sorted(set(source) - valid - {"debug", "use_AR_for_first_token"})
            if unknown:
                raise ValueError(f"unknown GAM decode parameters: {unknown}")
            merged.update({key: value for key, value in source.items() if key in valid})
        if merged.get("eos_token_id") is None and eos_token_id is not None:
            merged["eos_token_id"] = eos_token_id
        if "repetition_ignore_token_ids" in merged:
            merged["repetition_ignore_token_ids"] = tuple(
                int(item) for item in merged["repetition_ignore_token_ids"]
            )
        config = cls(**merged)
        config.validate()
        return config

    def with_request(self, request: Mapping[str, Any] | None) -> "DecodeConfig":
        return self.from_mappings(self.to_dict(), request)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size must be at least two")
        if not 1 <= self.sub_block_size <= self.block_size:
            raise ValueError("sub_block_size must be within the block")
        if self.denoise_steps < 1:
            raise ValueError("denoise_steps must be positive")
        if self.token_shift not in (0, 1):
            raise ValueError("token_shift must be zero or one")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be within (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative; zero disables filtering")
        if self.acceptance_policy not in ACCEPTANCE_POLICIES:
            raise ValueError(f"unsupported acceptance_policy={self.acceptance_policy!r}")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if self.acceptance_policy != "confidence" and self.entropy_threshold is None:
            raise ValueError("entropy_threshold is required by entropy policies")
        if self.entropy_threshold is not None and (
            not math.isfinite(self.entropy_threshold) or self.entropy_threshold < 0
        ):
            raise ValueError("entropy_threshold must be finite and non-negative")
        if not math.isfinite(self.position_penalty) or self.position_penalty < 0:
            raise ValueError("position_penalty must be finite and non-negative")
        if self.repetition_mode not in REPETITION_MODES:
            raise ValueError(f"unsupported repetition_mode={self.repetition_mode!r}")
        if not math.isfinite(self.repetition_weight) or self.repetition_weight < 0:
            raise ValueError("repetition_weight must be finite and non-negative")
        if self.repetition_ngram_size < 2:
            raise ValueError("repetition_ngram_size must be at least two")
        if self.repetition_window < self.repetition_ngram_size:
            raise ValueError("repetition_window is shorter than repetition_ngram_size")
        if self.repetition_block_size < 2:
            raise ValueError("repetition_block_size must be at least two")
        for name in ("eos_confidence_threshold",):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.eos_entropy_threshold is not None and self.eos_entropy_threshold < 0:
            raise ValueError("eos_entropy_threshold must be non-negative")
        if self.eos_stability_steps < 1:
            raise ValueError("eos_stability_steps must be positive")
        if self.enable_eos_early_stop and self.eos_token_id is None:
            raise ValueError("EOS early stop requires eos_token_id")
        if self.expected_cardinality is not None and self.expected_cardinality < 1:
            raise ValueError("expected_cardinality must be positive")
        coordinate_fields = (
            self.box_start_token_id,
            self.box_end_token_id,
            self.coord_token_min_id,
            self.coord_token_max_id,
            self.coordinates_per_target,
        )
        if self.expected_cardinality is not None and any(item is None for item in coordinate_fields):
            raise ValueError("cardinality decoding requires the complete coordinate contract")
        if self.coordinates_per_target is not None and self.coordinates_per_target not in (2, 4):
            raise ValueError("coordinates_per_target must be two or four")


def request_decode_mapping(custom_params: Any) -> Mapping[str, Any]:
    """Extract the nested request decoder mapping without accepting aliases."""

    if not isinstance(custom_params, Mapping):
        return {}
    value = custom_params.get("gam_dlm_decode")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("custom_params.gam_dlm_decode must be an object")
    return value
