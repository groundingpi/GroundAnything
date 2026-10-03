"""Low-overhead JSONL telemetry for decoder calibration and ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import atexit
import json
import os
from pathlib import Path
import threading
from typing import Any

import torch


@dataclass
class BlockTelemetry:
    request_id: str
    request_index: int
    block_index: int
    task: str | None
    profile: str | None
    generated_tokens: int = 0
    normal_accept_tokens: int = 0
    forced_one_tokens: int = 0
    forced_all_tokens: int = 0
    num_denoise_forwards: int = 0
    confidence_sum: float = 0.0
    entropy_sum: float = 0.0
    reliability_observations: int = 0
    repetition_trigger_count: int = 0
    max_repetition_score: float = 0.0
    eos_top1_count: int = 0
    eos_candidate_count: int = 0
    eos_accepted_count: int = 0
    eos_rejected_count: int = 0
    eos_probability_sum: float = 0.0
    eos_probability_observations: int = 0
    eos_max_probability: float = 0.0
    eos_candidate_entropy_sum: float = 0.0
    eos_candidate_observations: int = 0
    eos_max_stability: int = 0
    first_eos_position: int | None = None
    finish_reason: str | None = None
    accepted_per_step: list[int] = field(default_factory=list)
    confidence_histogram: list[int] = field(default_factory=lambda: [0] * 10)
    entropy_histogram: list[int] = field(default_factory=lambda: [0] * 12)
    adjusted_entropy_histogram: list[int] = field(default_factory=lambda: [0] * 12)
    _confidence_values: list[float] = field(default_factory=list, repr=False)
    _entropy_values: list[float] = field(default_factory=list, repr=False)
    _adjusted_values: list[float] = field(default_factory=list, repr=False)
    # ``detail`` is retained for calibration runs.  Production evaluation can
    # select ``summary`` to avoid copying every token's confidence/entropy to
    # host memory on every denoise forward.  The decoder math and commit
    # decisions never read this flag.
    telemetry_mode: str = field(
        default_factory=lambda: os.environ.get(
            "GAM_DLM_DECODE_TELEMETRY_MODE", "detail"
        ),
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.telemetry_mode not in {"detail", "summary", "off"}:
            raise ValueError(
                "GAM_DLM_DECODE_TELEMETRY_MODE must be 'detail', 'summary' or 'off'"
            )

    @property
    def enabled(self) -> bool:
        return self.telemetry_mode != "off"

    def observe_reliability(
        self,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
        mask: torch.Tensor,
        adjusted: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.telemetry_mode == "summary":
            # Keep reductions on device.  A single small host transfer carries
            # the two sums (and, when enabled, the adjusted sum) instead of
            # materialising O(vocab) / per-token CPU lists and histograms.
            selected_mask = mask.to(dtype=torch.bool)
            count = int(selected_mask.sum().item())
            if count == 0:
                return
            values = [
                confidence.masked_select(selected_mask).float().sum(),
                entropy.masked_select(selected_mask).float().sum(),
            ]
            if adjusted is not None:
                values.append(adjusted.masked_select(selected_mask).float().sum())
            reduced = torch.stack(values).detach().cpu().tolist()
            self.num_denoise_forwards += 1
            self.confidence_sum += float(reduced[0])
            self.entropy_sum += float(reduced[1])
            self.reliability_observations += count
            return

        selected_conf = confidence[mask].detach().float().cpu()
        selected_entropy = entropy[mask].detach().float().cpu()
        count = int(selected_conf.numel())
        if count == 0:
            return
        self.num_denoise_forwards += 1
        self.confidence_sum += float(selected_conf.sum().item())
        self.entropy_sum += float(selected_entropy.sum().item())
        self.reliability_observations += count
        self._confidence_values.extend(float(item) for item in selected_conf.tolist())
        self._entropy_values.extend(float(item) for item in selected_entropy.tolist())
        conf_bins = torch.clamp((selected_conf * 10).long(), 0, 9)
        entropy_bins = torch.clamp((selected_entropy / 0.5).long(), 0, 11)
        for index in conf_bins.tolist():
            self.confidence_histogram[int(index)] += 1
        for index in entropy_bins.tolist():
            self.entropy_histogram[int(index)] += 1
        if adjusted is not None:
            selected_adjusted = adjusted[mask].detach().float().cpu()
            self._adjusted_values.extend(float(item) for item in selected_adjusted.tolist())
            adjusted_bins = torch.clamp((selected_adjusted / 0.5).long(), 0, 11)
            for index in adjusted_bins.tolist():
                self.adjusted_entropy_histogram[int(index)] += 1

    def observe_eos(
        self,
        *,
        eos_probability: torch.Tensor,
        entropy: torch.Tensor,
        local_mask: torch.Tensor,
        eos_candidates: torch.Tensor,
        stability: torch.Tensor,
    ) -> None:
        """Record EOS calibration even when early-stop is disabled.

        This lets a baseline run select its EOS gate from the checkpoint's
        actual distribution instead of requiring an already-enabled policy.
        """

        if not self.enabled:
            return
        if self.telemetry_mode == "summary":
            masked_probability = eos_probability.masked_select(local_mask).float()
            candidate_entropy = entropy.masked_select(eos_candidates).float()
            values: list[torch.Tensor] = []
            if masked_probability.numel():
                values.extend(
                    [masked_probability.sum(), masked_probability.max()]
                )
            if candidate_entropy.numel():
                values.extend(
                    [
                        candidate_entropy.sum(),
                        stability.masked_select(eos_candidates).max().float(),
                    ]
                )
            if values:
                reduced = torch.stack(values).detach().cpu().tolist()
                index = 0
                if masked_probability.numel():
                    self.eos_probability_sum += float(reduced[index])
                    self.eos_probability_observations += int(
                        masked_probability.numel()
                    )
                    self.eos_max_probability = max(
                        self.eos_max_probability, float(reduced[index + 1])
                    )
                    index += 2
                if candidate_entropy.numel():
                    self.eos_candidate_entropy_sum += float(reduced[index])
                    self.eos_candidate_observations += int(
                        candidate_entropy.numel()
                    )
                    self.eos_max_stability = max(
                        self.eos_max_stability, int(reduced[index + 1])
                    )
            return

        masked_probability = eos_probability[local_mask].detach().float().cpu()
        if masked_probability.numel():
            self.eos_probability_sum += float(masked_probability.sum().item())
            self.eos_probability_observations += int(masked_probability.numel())
            self.eos_max_probability = max(
                self.eos_max_probability, float(masked_probability.max().item())
            )
        candidate_entropy = entropy[eos_candidates].detach().float().cpu()
        if candidate_entropy.numel():
            self.eos_candidate_entropy_sum += float(candidate_entropy.sum().item())
            self.eos_candidate_observations += int(candidate_entropy.numel())
            self.eos_max_stability = max(
                self.eos_max_stability,
                int(stability[eos_candidates].max().item()),
            )

    def row(self, config: dict[str, Any]) -> dict[str, Any]:
        payload = asdict(self)
        telemetry_mode = payload.pop("telemetry_mode")
        confidence_values = payload.pop("_confidence_values")
        entropy_values = payload.pop("_entropy_values")
        adjusted_values = payload.pop("_adjusted_values")
        observations = max(self.reliability_observations, 1)
        def quantiles(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            tensor = torch.tensor(values, dtype=torch.float32)
            points = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
            result = torch.quantile(tensor, points).tolist()
            return dict(zip(("p05", "p25", "p50", "p75", "p95"), result))
        payload.update(
            event="gam_decode_block",
            normal_accept_ratio=self.normal_accept_tokens / max(self.generated_tokens, 1),
            forced_one_ratio=self.forced_one_tokens / max(self.generated_tokens, 1),
            mean_raw_confidence=self.confidence_sum / observations,
            mean_raw_entropy=self.entropy_sum / observations,
            mean_eos_probability=(
                self.eos_probability_sum / max(self.eos_probability_observations, 1)
            ),
            mean_eos_candidate_entropy=(
                self.eos_candidate_entropy_sum
                / max(self.eos_candidate_observations, 1)
            ),
            tokens_per_forward=self.generated_tokens / max(self.num_denoise_forwards, 1),
            raw_confidence_quantiles=quantiles(confidence_values),
            raw_entropy_quantiles=quantiles(entropy_values),
            adjusted_entropy_quantiles=quantiles(adjusted_values),
            telemetry_mode=telemetry_mode,
            config=config,
        )
        return payload


class TelemetryWriter:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._stream = None
        self._stream_path: Path | None = None
        self._rows_since_flush = 0
        try:
            self._flush_every = max(
                1, int(os.environ.get("GAM_DLM_DECODE_TELEMETRY_FLUSH_EVERY", "16"))
            )
        except ValueError:
            self._flush_every = 16
        self._lock = threading.RLock()
        atexit.register(self.close)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def append(self, row: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self._lock:
            if self._stream is None or self._stream_path != self.path:
                self.close()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open(
                    "a", encoding="utf-8", buffering=1024 * 1024
                )
                self._stream_path = self.path
            self._stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._rows_since_flush += 1
            if self._rows_since_flush >= self._flush_every:
                self._stream.flush()
                self._rows_since_flush = 0

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            stream = getattr(self, "_stream", None)
            self._stream = None
            self._stream_path = None
            self._rows_since_flush = 0
            if stream is None:
                return
            try:
                stream.flush()
                stream.close()
            except (OSError, ValueError):
                pass


def telemetry_from_environment() -> TelemetryWriter:
    mode = os.environ.get("GAM_DLM_DECODE_TELEMETRY_MODE", "detail")
    # ``off`` is intentionally an explicit benchmark-only mode.  It avoids
    # both device-to-host telemetry reductions and JSON serialization; the
    # decoder's token decisions are unchanged.  Formal/eval defaults remain
    # ``detail`` (or caller-selected ``summary``).
    path = None if mode == "off" else os.environ.get("GAM_DLM_DECODE_TELEMETRY_PATH")
    return TelemetryWriter(path)
