#!/usr/bin/env python3
"""Isolated ``--mode DLM`` training entrypoint for GAM Direct Conversion."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if os.environ.get("GAM_DLM_RANK_ISOLATE_INDUCTOR_CACHE") == "1":
    cache_root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    local_rank_for_cache = os.environ.get("LOCAL_RANK")
    if not cache_root or local_rank_for_cache is None:
        raise RuntimeError("rank-isolated Inductor cache requires cache root and LOCAL_RANK")
    rank_cache = str(Path(cache_root) / f"rank_{local_rank_for_cache}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = rank_cache
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(rank_cache) / "triton"))

import torch
import yaml
from train.release_parameters import validate_training_fields, resolve_dlm_controls
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from train.dlm.data import DLMDataCollator, IndexedCacheDataset, PrecomputedPackingSampler
from models.dlm.hybrid import GAMQwen35DLM, initialize_mask_token
from models.dlm.vlm import GAMQwen3DLM


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_actual_length_preflight(
    config: dict[str, Any],
    model_path: str,
) -> dict[str, Any]:
    report_path = Path(config["runtime"]["selected_tail_actual_length_audit"]).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling_manifest = Path(config["data"]["sampling_manifest"]).resolve(strict=True)
    require(report.get("status") == "PASS", f"actual-length preflight failed: {report_path}")
    require(
        Path(report["sampling_manifest"]).resolve() == sampling_manifest,
        "actual-length preflight sampling manifest drift",
    )
    require(
        report["sampling_manifest_sha256"] == sha256_file(sampling_manifest),
        "actual-length preflight sampling SHA drift",
    )
    require(
        report["model_chat_template_sha256"]
        == sha256_file(Path(model_path) / "chat_template.jinja"),
        "actual-length preflight chat template drift",
    )
    require(
        report["model_tokenizer_sha256"] == sha256_file(Path(model_path) / "tokenizer.json"),
        "actual-length preflight tokenizer drift",
    )
    require(not report.get("invalid_rows"), "actual-length preflight contains invalid rows")
    model_family = str(config["model"].get("family", "qwen3_5"))
    if model_family == "groundinganything_qwen3":
        require(
            report.get("model_family") == model_family,
            "actual-length preflight was not rendered with the GroundAnything Qwen3 template",
        )
        require(
            int(report.get("min_cached_length", 1 << 30)) <= 2000,
            "GroundAnything actual-length audit did not cover the conservative cached-length tail",
        )
        require(
            report.get("include_multi_image") is True,
            "GroundAnything actual-length audit did not cover every multi-image row",
        )
    require(
        int(report["actual_max"]) <= int(config["runtime"]["max_length"]),
        "actual-length preflight exceeds runtime max_length",
    )
    require(
        int(report["actual_joint_max"]) <= int(config["runtime"]["max_joint_length"]),
        "actual-length preflight exceeds runtime max_joint_length",
    )
    audit = {
        "status": "PASS",
        "report": str(report_path),
        "audited_rows": int(report["audited_rows"]),
        "actual_max": int(report["actual_max"]),
        "actual_joint_max": int(report["actual_joint_max"]),
        "delta_max": int(report["delta_max"]),
    }
    if model_family == "groundinganything_qwen3":
        workload_path = Path(report["packing_workload_lengths"]).resolve(strict=True)
        require(
            report["packing_workload_lengths_sha256"] == sha256_file(workload_path),
            "Qwen3 packing-workload sidecar SHA drift",
        )
        audit.update(
            {
                "packing_workload_lengths": str(workload_path),
                "packing_workload_lengths_sha256": report["packing_workload_lengths_sha256"],
                "packing_workload_definition": report["packing_workload_definition"],
            }
        )
    return audit


def audit_completed_source_epoch_image_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicit completed-source-epoch image provenance contract.

    The released post-training caches already completed a full Swift epoch.  A
    second decode of several million identical source images is unnecessary,
    but that prior run may only be used by the isolated ``dlm_posttrain_sft``
    profile.  The contract binds the immutable source recipe, its normalized
    DLM recipe, the exact sampling manifest, and a final trainer state.  This is
    deliberately reported as provenance rather than a new strict decode audit.
    """

    runtime = config["runtime"]
    training = config["training"]
    require(
        str(training.get("training_profile")) == "dlm_posttrain_sft",
        "completed source-epoch image provenance is restricted to dlm_posttrain_sft",
    )
    provenance = runtime.get("completed_source_epoch_image_provenance")
    require(isinstance(provenance, dict), "missing completed source-epoch image provenance")

    sampling_path = Path(config["data"]["sampling_manifest"]).resolve(strict=True)
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    sampled_rows = int(config["data"]["sampled_rows"])
    require(int(sampling.get("sampled_rows", -1)) == sampled_rows, "sampling row drift")
    raw_recipe_rows = int(sampling.get("raw_recipe_effective_rows", -1))
    deleted_rejections = sampling.get("deleted_actual_length_rejections", [])
    require(isinstance(deleted_rejections, list), "invalid actual-length rejection provenance")
    require(
        raw_recipe_rows == sampled_rows + len(deleted_rejections),
        "post-training sampling differs from the released recipe beyond explicit length rejects",
    )
    require(
        int(sampling.get("image_excluded_unique_physical_rows", -1)) == 0,
        "completed-epoch provenance forbids image exclusions",
    )

    normalized_path = Path(sampling["source_config"]).resolve(strict=True)
    require(
        sampling.get("source_config_sha256") == sha256_file(normalized_path),
        "normalized source recipe SHA drift",
    )
    normalized = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
    normalized_meta = normalized.get("meta", {})
    require(
        int(normalized_meta.get("effective_rows", -1)) == raw_recipe_rows,
        "normalized source recipe row drift",
    )

    source_recipe = Path(provenance["source_recipe"]).resolve(strict=True)
    source_recipe_sha = str(provenance["source_recipe_sha256"])
    require(source_recipe_sha == sha256_file(source_recipe), "released source recipe SHA drift")
    require(
        Path(normalized_meta["source_recipe"]).resolve() == source_recipe,
        "normalized recipe points to a different released source recipe",
    )
    require(
        normalized_meta.get("source_recipe_sha256") == source_recipe_sha,
        "normalized recipe source SHA drift",
    )

    checkpoint = Path(provenance["completed_checkpoint"]).resolve(strict=True)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    global_step = int(trainer_state.get("global_step", -1))
    max_steps = int(trainer_state.get("max_steps", -1))
    epoch = float(trainer_state.get("epoch", -1.0))
    require(global_step > 0 and global_step == max_steps, "source epoch checkpoint is not final")
    require(epoch >= 1.0, "source epoch checkpoint did not complete one epoch")
    require(
        int(provenance.get("expected_global_step", global_step)) == global_step,
        "source epoch global-step drift",
    )
    require((checkpoint / "model.safetensors.index.json").is_file(), "source checkpoint lacks weights")

    return {
        "status": "PASS",
        "method": "completed_source_epoch_provenance",
        "sampling_manifest": str(sampling_path),
        "sampled_rows": sampled_rows,
        "raw_recipe_rows": raw_recipe_rows,
        "deleted_actual_length_rejections": len(deleted_rejections),
        "normalized_recipe": str(normalized_path),
        "source_recipe": str(source_recipe),
        "source_recipe_sha256": source_recipe_sha,
        "completed_checkpoint": str(checkpoint),
        "global_step": global_step,
        "max_steps": max_steps,
        "epoch": epoch,
        "claim": "prior full-epoch source-loader completion; not a new strict image decode audit",
    }


def audit_image_integrity_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on a strict decode audit or explicit source-epoch provenance."""

    runtime = config["runtime"]
    if not runtime.get("selected_image_integrity_audit"):
        return audit_completed_source_epoch_image_provenance(config)

    report_path = Path(runtime["selected_image_integrity_audit"]).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling_manifest = Path(config["data"]["sampling_manifest"]).resolve(strict=True)
    sampled_rows = int(config["data"]["sampled_rows"])
    require(report.get("status") == "PASS", f"image integrity preflight failed: {report_path}")
    require(
        report.get("image_decode_enabled") is True,
        "image integrity preflight did not strictly decode selected images",
    )
    require(
        Path(report["sampling_manifest"]).resolve() == sampling_manifest,
        "image integrity preflight sampling manifest drift",
    )
    require(
        report["sampling_manifest_sha256"] == sha256_file(sampling_manifest),
        "image integrity preflight sampling SHA drift",
    )
    require(not report.get("invalid_rows"), "image integrity preflight contains invalid rows")
    require(
        int(report["selected_rows"]) == sampled_rows
        and int(report["audited_rows"]) == sampled_rows,
        "image integrity preflight did not audit every selected row",
    )
    require(
        int(report.get("validated_multimodal_rows", -1)) == sampled_rows,
        "image integrity preflight did not validate every messages/images contract",
    )
    return {
        "status": "PASS",
        "report": str(report_path),
        "audited_rows": sampled_rows,
        "decoded_images": int(report["decoded_images"]),
        "validated_multimodal_rows": sampled_rows,
    }


def audit_response_block_preflight(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an optional exact supervised-response B32 admission report.

    Legacy configs omit this field and keep their existing behavior. New
    long-completion routes fail closed on manifest drift or ratio regression.
    """

    runtime = config["runtime"]
    raw_path = runtime.get("selected_response_block_audit")
    if not raw_path:
        return None
    report_path = Path(raw_path).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling_manifest = Path(config["data"]["sampling_manifest"]).resolve(strict=True)
    minimum = float(runtime.get("minimum_over_32_response_ratio", 0.4))
    require(report.get("status") == "PASS", f"response-block preflight failed: {report_path}")
    require(int(report.get("block_size", -1)) == 32, "response-block audit is not B32")
    require(
        Path(report["sampling_manifest"]).resolve() == sampling_manifest,
        "response-block preflight sampling manifest drift",
    )
    require(
        report["sampling_manifest_sha256"] == sha256_file(sampling_manifest),
        "response-block preflight sampling SHA drift",
    )
    ratio = float(report["overall"]["over_32_ratio"])
    require(ratio >= minimum, f"response-block ratio below gate: {ratio:.6f} < {minimum:.6f}")
    route_token_shares: dict[str, float] = {}
    maximum_ocr_dense = runtime.get("maximum_ocr_dense_response_token_share")
    minimum_route_token_shares = runtime.get("minimum_route_response_token_shares", {})
    minimum_route_rows = runtime.get("minimum_route_rows", {})
    if maximum_ocr_dense is not None or minimum_route_token_shares or minimum_route_rows:
        routes = report.get("routes")
        require(isinstance(routes, dict) and routes, "route token-budget contract lacks route audit")
        token_totals = {
            str(route): float(summary["mean"]) * int(summary["rows"])
            for route, summary in routes.items()
        }
        total_tokens = sum(token_totals.values())
        require(total_tokens > 0.0, "route token-budget audit is empty")
        route_token_shares = {
            route: tokens / total_tokens for route, tokens in token_totals.items()
        }
        if maximum_ocr_dense is not None:
            require("ocr" in token_totals and "dense" in token_totals, "OCR/Dense route audit is missing")
            ocr_dense_share = route_token_shares["ocr"] + route_token_shares["dense"]
            require(
                ocr_dense_share <= float(maximum_ocr_dense),
                "OCR+Dense response-token share exceeds hard cap: "
                f"{ocr_dense_share:.6f} > {float(maximum_ocr_dense):.6f}",
            )
        for route, raw_floor in minimum_route_token_shares.items():
            require(route in route_token_shares, f"minimum token-share route is missing: {route}")
            require(
                route_token_shares[route] >= float(raw_floor),
                f"route token share below floor: {route} "
                f"{route_token_shares[route]:.6f} < {float(raw_floor):.6f}",
            )
        for route, raw_floor in minimum_route_rows.items():
            require(route in routes, f"minimum-row route is missing: {route}")
            require(
                int(routes[route]["rows"]) >= int(raw_floor),
                f"route rows below floor: {route} "
                f"{int(routes[route]['rows'])} < {int(raw_floor)}",
            )
    parent = report.get("parent_response_audit")
    require(isinstance(parent, dict), "response-block report lacks exact tokenizer-audit provenance")
    parent_path = Path(parent["report"]).resolve(strict=True)
    require(parent["report_sha256"] == sha256_file(parent_path), "parent response audit SHA drift")
    parent_report = json.loads(parent_path.read_text(encoding="utf-8"))
    require(parent_report.get("status") == "PASS", "parent exact response audit is not PASS")
    require(
        parent_report.get("response_tokens_sha256") == report.get("response_tokens_sha256"),
        "response token sidecar provenance drift",
    )
    audit = {
        "status": "PASS",
        "report": str(report_path),
        "rows": int(report["overall"]["rows"]),
        "over_32_ratio": ratio,
        "over_64_ratio": float(report["overall"]["over_64_ratio"]),
        "over_128_ratio": float(report["overall"]["over_128_ratio"]),
    }
    if route_token_shares:
        audit["route_response_token_shares"] = route_token_shares
        audit["ocr_dense_response_token_share"] = (
            route_token_shares["ocr"] + route_token_shares["dense"]
        )
    return audit


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"config root must be a mapping: {path}")
    require(value.get("mode") == "DLM", "DLM entrypoint rejects configs without mode: DLM")
    validate_training_fields(value)
    return value


def load_initial_dlm_checkpoint(model: GAMQwen35DLM, checkpoint: Path) -> dict[str, Any]:
    """Load DLM wrapper weights without resuming optimizer/scheduler state."""

    from safetensors.torch import load_file

    checkpoint = checkpoint.resolve(strict=True)
    weights = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    weights = weights.resolve(strict=True)
    state = load_file(str(weights), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    require(not missing, f"initial DLM checkpoint missing keys: {missing[:20]}")
    require(not unexpected, f"initial DLM checkpoint unexpected keys: {unexpected[:20]}")
    require(
        state and all(key.startswith("base_model.") for key in state),
        "initial DLM checkpoint is not a GAM DLM wrapper state dict",
    )
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "weights": str(weights),
        "tensors": len(state),
    }


class AllocatorCacheCleanupCallback(TrainerCallback):
    """Release only inactive CUDA allocator blocks at optimizer-step boundaries.

    Packed DLM batches intentionally vary in shape.  Some accelerator allocators
    retain incompatible historical blocks until ``reserved`` reaches the device
    limit even though live ``allocated`` memory is low.  ``empty_cache`` never
    frees live tensors, parameters, gradients, or optimizer state, so this is an
    allocator hygiene operation rather than a training-dynamics change.
    """

    def __init__(self, every_n_steps: int) -> None:
        every_n_steps = int(every_n_steps)
        require(every_n_steps > 0, "allocator cache cleanup interval must be positive")
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        del kwargs
        step = int(state.global_step)
        if step <= 0 or step % self.every_n_steps or not torch.cuda.is_available():
            return control
        allocated = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved()
        if int(getattr(args, "local_process_index", 0)) == 0:
            print(
                json.dumps(
                    {
                        "allocator_cache_cleanup": {
                            "status": "PASS",
                            "step": step,
                            "every_n_steps": self.every_n_steps,
                            "allocated_gib": allocated / 2**30,
                            "reserved_before_gib": reserved_before / 2**30,
                            "reserved_after_gib": reserved_after / 2**30,
                            "released_gib": max(0, reserved_before - reserved_after) / 2**30,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return control


class CompilerCacheCleanupCallback(TrainerCallback):
    """Diagnose and bound live HBM retained by compiled dynamic-shape graphs.

    Resetting compiler caches changes only compilation/runtime state. It does
    not mutate parameters, gradients, optimizer state, RNG, data order, or the
    LR scheduler. The option is disabled unless an experiment opts in.
    """

    def __init__(self, every_n_steps: int) -> None:
        every_n_steps = int(every_n_steps)
        require(every_n_steps > 0, "compiler cache reset interval must be positive")
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        del kwargs
        step = int(state.global_step)
        if step <= 0 or step % self.every_n_steps or not torch.cuda.is_available():
            return control
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        unique_graphs = None
        try:
            from torch._dynamo.utils import counters

            unique_graphs = int(counters["stats"].get("unique_graphs", 0))
        except Exception:
            pass
        reset = getattr(getattr(torch, "compiler", None), "reset", None)
        if reset is None:
            from torch import _dynamo

            reset = _dynamo.reset
        reset()
        torch.cuda.empty_cache()
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        if int(getattr(args, "local_process_index", 0)) == 0:
            print(
                json.dumps(
                    {
                        "compiler_cache_cleanup": {
                            "status": "PASS",
                            "step": step,
                            "every_n_steps": self.every_n_steps,
                            "dynamo_unique_graphs_before": unique_graphs,
                            "allocated_before_gib": allocated_before / 2**30,
                            "allocated_after_gib": allocated_after / 2**30,
                            "reserved_before_gib": reserved_before / 2**30,
                            "reserved_after_gib": reserved_after / 2**30,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return control


class AllocatorSnapshotDeltaCallback(TrainerCallback):
    """Summarize newly retained active device allocations at selected steps."""

    def __init__(self, steps: list[int]) -> None:
        normalized = sorted({int(step) for step in steps})
        require(normalized and normalized[0] > 0, "allocator snapshot steps must be positive")
        self.steps = normalized
        self._previous: Counter[tuple[int, int, str]] | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        del state, kwargs
        if int(getattr(args, "process_index", 0)) != 0 or not torch.cuda.is_available():
            return control
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=200000,
            )
            status = "PASS"
        except Exception as error:
            status = f"UNAVAILABLE:{type(error).__name__}:{error}"
        print(json.dumps({"allocator_history": {"status": status}}, sort_keys=True), flush=True)
        return control

    @staticmethod
    def _active_allocations() -> Counter[tuple[int, int, str]]:
        snapshot = torch.cuda.memory._snapshot()
        allocations: Counter[tuple[int, int, str]] = Counter()
        for segment in snapshot.get("segments", []):
            for block in segment.get("blocks", []):
                if not str(block.get("state", "")).startswith("active"):
                    continue
                requested = int(block.get("requested_size", block.get("size", 0)))
                size = int(block.get("size", requested))
                source = "<history-unavailable>"
                history = block.get("history") or []
                if history:
                    frames = history[-1].get("frames") or []
                    if frames:
                        selected = [
                            f"{frame.get('filename', '?')}:{frame.get('line', '?')}:{frame.get('name', '?')}"
                            for frame in frames[-4:]
                        ]
                        source = " <- ".join(selected)
                allocations[(requested, size, source)] += 1
        return allocations

    def on_step_end(self, args, state, control, **kwargs):
        del kwargs
        step = int(state.global_step)
        if (
            step not in self.steps
            or int(getattr(args, "process_index", 0)) != 0
            or not torch.cuda.is_available()
        ):
            return control
        try:
            current = self._active_allocations()
            baseline = self._previous or Counter()
            positive = current - baseline
            entries = sorted(
                (
                    {
                        "requested_bytes": key[0],
                        "allocated_bytes": key[1],
                        "count_delta": count,
                        "requested_bytes_delta": key[0] * count,
                        "source": key[2],
                    }
                    for key, count in positive.items()
                ),
                key=lambda item: item["requested_bytes_delta"],
                reverse=True,
            )[:24]
            payload = {
                "status": "PASS",
                "step": step,
                "active_block_count": sum(current.values()),
                "active_requested_gib": sum(key[0] * count for key, count in current.items()) / 2**30,
                "positive_requested_gib_since_previous": sum(
                    key[0] * count for key, count in positive.items()
                )
                / 2**30,
                "top_positive_deltas": entries,
            }
            self._previous = current
        except Exception as error:
            payload = {"status": "FAILED", "step": step, "error": f"{type(error).__name__}:{error}"}
        print(json.dumps({"allocator_snapshot_delta": payload}, sort_keys=True), flush=True)
        return control


class StopAfterStepCallback(TrainerCallback):
    """Test-only clean interruption at an optimizer-step boundary."""

    def __init__(self, stop_after_step: int) -> None:
        self.stop_after_step = int(stop_after_step)
        require(self.stop_after_step > 0, "stop-after-step must be positive")

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if int(state.global_step) >= self.stop_after_step:
            control.should_training_stop = True
        return control


class TorchProfilerStepCallback(TrainerCallback):
    """Advance a rank-local scheduled profiler after each optimizer step."""

    def __init__(self, profiler: Any) -> None:
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.profiler.step()
        return control


class DLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        group_learning_rates: dict[str, float] | None = None,
        optimizer_foreach: bool | None = None,
        **kwargs,
    ):
        self._dlm_loss_sums = {"mdm_loss": 0.0, "causal_loss": 0.0}
        self._dlm_loss_count = 0
        self._dlm_group_learning_rates = group_learning_rates
        self._dlm_optimizer_foreach = optimizer_foreach
        super().__init__(*args, **kwargs)
        # GAMQwen35DLM.forward accepts **kwargs for multimodal compatibility,
        # but it does not consume num_items_in_batch.  Transformers 5.2 would
        # otherwise skip the standard /gradient_accumulation_steps scaling.
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self, model=None):
        """Create explicit multimodal LR groups only for the Qwen3 route."""

        if self._dlm_group_learning_rates is None:
            return super().create_optimizer(model)
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model if model is None else model
        expected = {"vision_encoder", "projector", "language"}
        if set(self._dlm_group_learning_rates) != expected:
            raise RuntimeError("DLM optimizer LR group contract drift")
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        grouped: list[dict[str, Any]] = []
        audit: dict[str, dict[str, float | int]] = {}
        for group_name in sorted(expected):
            names_and_parameters = [
                (name, parameter)
                for name, parameter in opt_model.named_parameters()
                if parameter.requires_grad and _parameter_group(name) == group_name
            ]
            audit[group_name] = {
                "parameters": sum(parameter.numel() for _, parameter in names_and_parameters),
                "tensors": len(names_and_parameters),
                "peak_lr": float(self._dlm_group_learning_rates[group_name]),
            }
            for decay in (True, False):
                parameters = [
                    parameter
                    for name, parameter in names_and_parameters
                    if (name in decay_parameters) == decay
                ]
                if parameters:
                    grouped.append(
                        {
                            "params": parameters,
                            "weight_decay": self.args.weight_decay if decay else 0.0,
                            "lr": float(self._dlm_group_learning_rates[group_name]),
                            "group_name": group_name,
                        }
                    )
        # ``audit_trainable_parameters`` already enforces the YAML freeze
        # contract.  Stage-2 post-training is intentionally LLM-only, so an
        # empty projector optimizer group is valid there while language never
        # is.
        if audit["language"]["parameters"] <= 0:
            raise RuntimeError(f"DLM optimizer lost a trainable parameter group: {audit}")
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        if any(key in optimizer_kwargs for key in ("params", "model", "optimizer_dict")):
            raise RuntimeError("selected optimizer cannot preserve explicit multimodal LR groups")
        if self._dlm_optimizer_foreach is not None:
            if optimizer_cls is not torch.optim.AdamW:
                raise RuntimeError("optimizer_foreach is supported only by torch.optim.AdamW")
            optimizer_kwargs["foreach"] = self._dlm_optimizer_foreach
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        if self.args.process_index == 0:
            print(
                json.dumps(
                    {
                        "optimizer_group_audit": {
                            "status": "PASS",
                            "groups": audit,
                            "foreach": self._dlm_optimizer_foreach,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return self.optimizer

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if getattr(dataset, "dlm_complementary_length_sampling", False):
            packing_manifest = getattr(dataset, "dlm_packing_manifest", None)
            if not packing_manifest:
                raise RuntimeError("DLM packing requires a standalone packing manifest")
            sampler = PrecomputedPackingSampler(
                packing_manifest,
                rows=len(dataset),
                sampling_manifest_path=dataset.manifest_path,
                world_size=int(self.args.world_size),
                pack_size=int(self.args.per_device_train_batch_size),
                ordering_lengths_path=getattr(dataset, "_dlm_packing_workload_path", None),
            )
            if self.args.process_index == 0:
                print(
                    json.dumps(
                        {
                            "packing_pair_audit": {
                                "status": "PASS",
                                "algorithm": sampler.algorithm,
                                "rows": len(dataset),
                                "manifest": str(sampler.manifest_path),
                                **sampler.audit,
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return sampler
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        outputs = model(**inputs)
        audit_model = getattr(model, "module", model)
        self._dlm_last_packing_activation_cpu_offload = bool(
            getattr(audit_model, "_last_packing_activation_cpu_offload", False)
        )
        self._dlm_last_packing_workload_tokens = int(
            getattr(audit_model, "_last_packing_workload_tokens", 0)
        )
        self._dlm_loss_sums["mdm_loss"] += float(outputs.mdm_loss.item())
        self._dlm_loss_sums["causal_loss"] += float(outputs.causal_loss.item())
        self._dlm_loss_count += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._dlm_loss_count:
            totals = torch.tensor(
                [
                    self._dlm_loss_sums["mdm_loss"],
                    self._dlm_loss_sums["causal_loss"],
                    float(self._dlm_loss_count),
                ],
                dtype=torch.float64,
                device=self.args.device,
            )
            # The regular training-log path is entered by every rank.  Reduce
            # these auxiliary components so they describe the same global
            # batch as Trainer's already-gathered ``loss`` metric.
            if "loss" in logs and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            count = totals[2].item()
            logs = {
                **logs,
                "mdm_loss": totals[0].item() / count,
                "causal_loss": totals[1].item() / count,
                "packing_activation_cpu_offload": float(
                    getattr(self, "_dlm_last_packing_activation_cpu_offload", False)
                ),
                "packing_workload_tokens": float(
                    getattr(self, "_dlm_last_packing_workload_tokens", 0)
                ),
            }
            self._dlm_loss_sums = {"mdm_loss": 0.0, "causal_loss": 0.0}
            self._dlm_loss_count = 0
        if torch.cuda.is_available():
            logs = {
                **logs,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            torch.cuda.reset_peak_memory_stats()
        super().log(logs, *args, **kwargs)


def configure_processor(
    processor: Any,
    image_max_token_num: int,
    model_family: str = "qwen3_5",
) -> None:
    # Qwen3.5 uses 16x16 patches and a 2x2 spatial merge: one LLM visual
    # token covers 32x32 pixels.  This matches the cache's 1024-token contract.
    image_processor = processor.image_processor
    if model_family == "groundinganything_qwen3":
        merge_size = int(getattr(image_processor, "merge_size", 2))
        if hasattr(image_processor, "media_proc_cfg"):
            image_processor.media_proc_cfg["in_patch_limit"] = image_max_token_num * merge_size**2
        if hasattr(image_processor, "max_output_tokens"):
            image_processor.max_output_tokens = image_max_token_num
        return
    max_pixels = image_max_token_num * 32 * 32
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        image_processor.size["longest_edge"] = max_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


def configure_vision_gradient_checkpointing(
    model: torch.nn.Module,
    enabled: bool,
    model_family: str,
) -> dict[str, Any]:
    """Enable checkpointing only in ViT while keeping the language tower off.

    Hugging Face initializes the private checkpoint function through the
    top-level helper.  We then disable every language-side flag explicitly;
    this avoids enabling the custom DLM language checkpoint path and preserves
    padding-free packing semantics.
    """

    visual = model.model.visual
    language = model.model.language_model
    if enabled and model_family != "groundinganything_qwen3":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    vision_flags: list[str] = []
    dynamically_wrapped_blocks = 0
    if enabled and model_family == "groundinganything_qwen3":
        from functools import wraps
        from torch.utils.checkpoint import checkpoint

        blocks = model.model.visual.vision_tower.encoder.blocks
        require(len(blocks) > 0, "Kimi-K3 ViT checkpointing found no encoder blocks")
        for index, block in enumerate(blocks):
            original_forward = block.forward

            @wraps(original_forward)
            def checkpointed_forward(*args, __forward=original_forward, __block=block, **kwargs):
                if not torch.is_grad_enabled() or not __block.training:
                    return __forward(*args, **kwargs)
                return checkpoint(
                    __forward,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )

            block.forward = checkpointed_forward
            block._gam_dlm_vision_gradient_checkpointing = True
            vision_flags.append(f"vision_tower.encoder.blocks.{index}")
            dynamically_wrapped_blocks += 1
    for name, module in visual.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = bool(enabled)
            vision_flags.append(name or "<vision_root>")

    language_flags: list[str] = []
    for name, module in language.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
            if bool(module.gradient_checkpointing):
                language_flags.append(name or "<language_root>")

    if enabled:
        require(vision_flags, "ViT gradient checkpointing requested but no vision flag was found")
        if model_family == "groundinganything_qwen3":
            require(
                dynamically_wrapped_blocks == 27,
                f"Kimi-K3 ViT GC must wrap exactly 27 blocks, got {dynamically_wrapped_blocks}",
            )
    require(not language_flags, f"language gradient checkpointing remained enabled: {language_flags}")
    return {
        "status": "PASS",
        "vision_enabled": bool(enabled),
        "dynamic_block_wrappers": dynamically_wrapped_blocks,
        "vision_flag_modules": vision_flags,
        "language_enabled_modules": language_flags,
    }


def _parameter_group(name: str) -> str:
    if ".visual.merger." in name or ".visual.projector." in name:
        return "projector"
    if ".visual." in name:
        return "vision_encoder"
    return "language"


def configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    freeze_vision_encoder: bool,
    freeze_projector: bool = False,
) -> None:
    """Apply the stage-specific full/frozen parameter contract.

    General VQA keeps the ViT frozen while training the multimodal projector
    and language model.  Specialist Direct Conversion trains all three.  The
    merger is intentionally classified as the projector, not as part of ViT.
    """

    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        frozen = (freeze_vision_encoder and group == "vision_encoder") or (
            freeze_projector and group == "projector"
        )
        parameter.requires_grad_(not frozen)


def audit_trainable_parameters(
    model: torch.nn.Module,
    *,
    freeze_vision_encoder: bool,
    freeze_projector: bool = False,
) -> dict[str, Any]:
    """Fail closed unless each stage has exactly the requested trainable groups."""

    groups: dict[str, dict[str, int]] = {
        "vision_encoder": {"total": 0, "trainable": 0},
        "projector": {"total": 0, "trainable": 0},
        "language": {"total": 0, "trainable": 0},
    }
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        groups[group]["total"] += parameter.numel()
        if parameter.requires_grad:
            groups[group]["trainable"] += parameter.numel()
    expected_trainable = {
        "vision_encoder": not freeze_vision_encoder,
        "projector": not freeze_projector,
        "language": True,
    }
    for group, counts in groups.items():
        require(counts["total"] > 0, f"parameter audit found no {group} parameters")
        expected = counts["total"] if expected_trainable[group] else 0
        require(
            counts["trainable"] == expected,
            f"{group} trainability drift: {counts['trainable']}/{counts['total']} "
            f"(expected {expected})",
        )
    return {
        "status": "PASS",
        "freeze_vision_encoder": freeze_vision_encoder,
        "freeze_projector": freeze_projector,
        "expected_trainable": expected_trainable,
        "groups": groups,
    }


def build_training_args(
    config: dict[str, Any],
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_steps: int,
    disable_checkpointing: bool,
    expected_global_batch_size: int | None = None,
) -> TrainingArguments:
    validate_training_fields(config)
    training = config["training"]
    max_steps = training.get("max_steps", max_steps)
    disable_checkpointing = training.get("disable_checkpointing", disable_checkpointing)
    runtime = config["runtime"]
    model_family = str(config["model"].get("family", "qwen3_5"))
    require(model_family in {"qwen3_5", "groundinganything_qwen3"}, f"unsupported DLM model family: {model_family}")
    world_size = int(os.environ.get("WORLD_SIZE", runtime["expected_world_size"]))
    effective_batch = world_size * batch_size * gradient_accumulation_steps
    expected_global_batch = int(
        training.get("expected_global_batch_size", 256)
        if expected_global_batch_size is None
        else expected_global_batch_size
    )
    require(
        effective_batch == expected_global_batch,
        f"effective global batch drift: {effective_batch} != {expected_global_batch}",
    )
    print(
        json.dumps(
            {
                "mode": "DLM",
                "world_size": world_size,
                "per_device_batch": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_checkpointing": gradient_checkpointing,
                "effective_global_batch": effective_batch,
                "expected_global_batch": expected_global_batch,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tensorboard_dir = str(Path(output_dir) / "tensorboard")
    os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_dir
    save_strategy = "no" if disable_checkpointing else str(training["save_strategy"])
    require(save_strategy in {"no", "steps"}, f"unsupported DLM save_strategy: {save_strategy}")
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        lr_scheduler_kwargs=dict(training.get("lr_scheduler_kwargs", {})),
        # Transformers 5.2 accepts a fractional warmup directly via
        # warmup_steps; this is exactly the paper's 3% warmup ratio.
        warmup_steps=float(training["warmup_ratio"]),
        optim=str(training["optim"]),
        adam_beta1=float(training["adam_beta1"]),
        adam_beta2=float(training["adam_beta2"]),
        adam_epsilon=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=True,
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(Path(runtime["deepspeed_config"]).resolve(strict=True)),
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        disable_tqdm=True,
        save_strategy=save_strategy,
        save_steps=float(training["save_steps"]),
        save_total_limit=None,
        save_only_model=False,
        report_to=["tensorboard"],
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=int(training["dataloader_num_workers"]) > 0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=int(training["seed"]),
        data_seed=int(training["data_seed"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("DLM",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--deepspeed-config", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--expected-global-batch-size", type=int)
    parser.add_argument(
        "--casuallossenable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the legacy auxiliary causal CE objective; default comes from YAML and remains true.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vision-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--language-checkpoint-stride", type=int)
    parser.add_argument("--packing-language-checkpoint-stride", type=int)
    parser.add_argument("--padding-free-packing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--packing-manifest")
    parser.add_argument("--packing-linear-kernel", choices=("chunk",))
    parser.add_argument(
        "--packing-lm-head-loss-backend",
        choices=("fused", "checkpointed_chunk"),
    )
    parser.add_argument("--packing-lm-head-loss-chunk-tokens", type=int)
    parser.add_argument("--packing-activation-cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--packing-offload-threshold-mib", type=int)
    parser.add_argument("--packing-offload-min-tokens", type=int)
    parser.add_argument("--packing-offload-layer-stride", type=int)
    parser.add_argument(
        "--packing-offload-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--disable-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--initial-dlm-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(
        not (args.resume_from_checkpoint and args.initial_dlm_checkpoint),
        "initial-dlm-checkpoint and resume-from-checkpoint are mutually exclusive",
    )
    config = load_config(args.config.resolve(strict=True))
    resolve_dlm_controls(args, config)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        # Bind before model/Trainer construction.  Otherwise optional CUDA
        # imports create one context per local process on GPU0 (~612 MiB each).
        torch.cuda.set_device(local_rank)
    training = config["training"]
    runtime = config["runtime"]
    if args.deepspeed_config is not None:
        # Qwen3-only admission runs may tune ZeRO communication-buffer
        # capacity without mutating the already validated Qwen3.5 route.
        runtime["deepspeed_config"] = str(args.deepspeed_config.resolve(strict=True))
    model_family = str(config["model"].get("family", "qwen3_5"))
    require(
        model_family in {"qwen3_5", "groundinganything_qwen3"},
        f"unsupported DLM model family: {model_family}",
    )
    packed_clean_attention_backend = os.environ.get(
        "GAM_DLM_PACKED_CLEAN_ATTENTION_BACKEND",
        "flash_attention_2",
    )
    require(
        packed_clean_attention_backend
        in {"flash_attention_2", "flash_attention_3", "cudnn_sdpa"},
        f"unsupported packed clean attention backend: {packed_clean_attention_backend}",
    )
    require(
        model_family == "groundinganything_qwen3" or packed_clean_attention_backend == "flash_attention_2",
        "experimental packed clean attention backends are isolated to GroundAnything Qwen3",
    )
    model_path = str(Path(args.model_path or config["model"]["path"]).resolve(strict=True))
    output_dir = str(Path(args.output_dir or config["model"]["output_dir"]).resolve())
    batch_size = args.batch_size or int(training["per_device_train_batch_size"])
    accumulation = args.gradient_accumulation_steps or int(training["gradient_accumulation_steps"])
    casuallossenable = (
        bool(training.get("casuallossenable", True))
        if args.casuallossenable is None
        else bool(args.casuallossenable)
    )
    gradient_checkpointing = (
        bool(training["gradient_checkpointing"])
        if args.gradient_checkpointing is None
        else args.gradient_checkpointing
    )
    padding_free_packing = (
        bool(training.get("padding_free_packing", False))
        if args.padding_free_packing is None
        else args.padding_free_packing
    )
    language_checkpoint_stride = (
        args.language_checkpoint_stride
        if args.language_checkpoint_stride is not None
        else int(training.get("language_checkpoint_stride", 0))
    )
    packing_language_checkpoint_stride = (
        args.packing_language_checkpoint_stride
        if args.packing_language_checkpoint_stride is not None
        else int(training.get("packing_language_checkpoint_stride", 1))
    )
    packing_linear_kernel = (
        args.packing_linear_kernel
        or str(training.get("packing_linear_kernel", "chunk"))
    )
    packing_lm_head_loss_backend = (
        args.packing_lm_head_loss_backend
        or str(training.get("packing_lm_head_loss_backend", "fused"))
    )
    packing_lm_head_loss_chunk_tokens = (
        args.packing_lm_head_loss_chunk_tokens
        or int(training.get("packing_lm_head_loss_chunk_tokens", 128))
    )
    packing_activation_cpu_offload = (
        bool(training.get("packing_activation_cpu_offload", False))
        if args.packing_activation_cpu_offload is None
        else args.packing_activation_cpu_offload
    )
    packing_offload_threshold_mib = (
        args.packing_offload_threshold_mib
        or int(training.get("packing_offload_threshold_mib", 8))
    )
    packing_offload_min_tokens = (
        args.packing_offload_min_tokens
        if args.packing_offload_min_tokens is not None
        else int(training.get("packing_offload_min_tokens", 0))
    )
    packing_offload_layer_stride = (
        args.packing_offload_layer_stride
        if args.packing_offload_layer_stride is not None
        else int(training.get("packing_offload_layer_stride", 1))
    )
    packing_offload_pin_memory = (
        bool(training.get("packing_offload_pin_memory", True))
        if args.packing_offload_pin_memory is None
        else args.packing_offload_pin_memory
    )
    optimizer_foreach = training.get("optimizer_foreach")
    require(
        optimizer_foreach is None or isinstance(optimizer_foreach, bool),
        "optimizer_foreach must be a boolean when configured",
    )
    allocator_empty_cache_steps = int(training.get("allocator_empty_cache_steps", 0))
    compiler_cache_reset_steps = int(training.get("compiler_cache_reset_steps", 0))
    allocator_snapshot_steps = training.get("allocator_snapshot_steps", [])
    require(
        isinstance(allocator_snapshot_steps, list)
        and all(isinstance(step, int) and step > 0 for step in allocator_snapshot_steps),
        "allocator_snapshot_steps must be a list of positive integers",
    )
    if args.stop_after_step is not None:
        require(args.max_steps > 0, "stop-after-step requires an explicit positive max-steps")
        require(
            0 < args.stop_after_step < args.max_steps,
            "stop-after-step must be inside the immutable scheduler horizon",
        )
    batch_size_limit = 8
    validated_large_batch_size = int(training.get("validated_large_batch_size", 0))
    large_batch_pressure = os.environ.get("GAM_DLM_ALLOW_LARGE_BATCH_PRESSURE") == "1"
    # The formal large-batch contract intentionally rejects a finite, partial
    # schedule.  A smoke gate still needs to exercise the *same* BS32 packed
    # kernels and memory path without consuming a full epoch, however.  Keep
    # this escape hatch explicit and process-local: callers must opt in with a
    # smoke-only environment variable, checkpointing remains enabled, and no
    # formal launch can inherit it accidentally.
    smoke_partial_schedule = os.environ.get("GAM_DLM_SMOKE_PARTIAL_SCHEDULE") == "1"
    if smoke_partial_schedule:
        require(
            args.max_steps > 0 and not args.disable_checkpointing,
            "smoke partial schedule requires finite max-steps with checkpointing enabled",
        )
    require(
        not (large_batch_pressure and validated_large_batch_size),
        "pressure and formal large-batch admission controls are mutually exclusive",
    )
    if large_batch_pressure:
        batch_size_limit = int(os.environ.get("GAM_DLM_PRESSURE_BATCH_LIMIT", "64"))
        require(
            8 < batch_size_limit <= 64,
            "large-batch pressure limit must be in [9, 64]",
        )
        require(
            args.max_steps > 0 and args.disable_checkpointing,
            "large-batch override is restricted to finite checkpoint-free pressure tests",
        )
    elif validated_large_batch_size:
        require(
            model_family == "groundinganything_qwen3",
            "validated formal large batches are isolated to the GroundAnything Qwen3 route",
        )
        require(
            8 < validated_large_batch_size <= 64,
            "validated formal large-batch size must be in [9, 64]",
        )
        require(
            batch_size == validated_large_batch_size,
            "runtime batch size must match validated_large_batch_size",
        )
        require(
            (
                args.max_steps <= 0 and not args.disable_checkpointing
            ) or smoke_partial_schedule,
            "validated formal large batches require the full checkpointed epoch schedule "
            "(only an explicit smoke partial-schedule gate may override this)",
        )
        require(
            padding_free_packing and gradient_checkpointing and not packing_activation_cpu_offload,
            "validated formal large batches require packed LLM checkpointing without CPU offload",
        )
        packing_manifest_path = Path(
            args.packing_manifest or config["data"].get("packing_manifest", "")
        ).resolve(strict=True)
        packing_contract = json.loads(packing_manifest_path.read_text(encoding="utf-8"))
        require(
            int(packing_contract.get("pack_size", 0)) == validated_large_batch_size,
            "validated formal large-batch packing size drift",
        )
        require(
            packing_contract.get("packing_strategy") == "balanced",
            "validated formal large batches require balanced packing",
        )
        require(
            int(packing_contract.get("world_size", 0)) == int(runtime["expected_world_size"]),
            "validated formal large-batch packing world-size drift",
        )
        require(
            len(packing_contract.get("epoch_orders", [])) >= int(training["num_train_epochs"]),
            "validated formal large-batch packing does not cover every training epoch",
        )
        batch_size_limit = validated_large_batch_size
    require(
        1 <= batch_size <= batch_size_limit,
        f"DLM memory policy only permits per-device batch 1--{batch_size_limit}",
    )
    block_size = int(training["block_size"])
    block_experiment = training.get("fixed_block_size_experiment")
    require(not bool(training.get("block_annealing", False)), "block annealing is forbidden")
    require(
        block_size == 32 or (block_size == 8 and block_experiment == "B8"),
        "non-default block size is isolated to fixed_block_size_experiment: B8",
    )
    group_learning_rates: dict[str, float] | None = None
    if model_family == "groundinganything_qwen3":
        require(
            not packing_activation_cpu_offload or packing_offload_min_tokens == 0,
            "GroundAnything Qwen3 formal offload policy must be sample-independent (min_tokens=0)",
        )
        backbone_lr = float(training["backbone_learning_rate"])
        vision_lr = float(training["vision_learning_rate"])
        projector_lr = float(training.get("projector_learning_rate", backbone_lr))
        training_profile = str(training.get("training_profile", "direct_conversion"))
        if training_profile == "direct_conversion":
            require(
                (backbone_lr, vision_lr) in {(1.0e-5, 5.0e-6), (2.0e-5, 2.0e-6)},
                "Qwen3 LR contract must be legacy 1e-5/5e-6 or experimental 2e-5/2e-6",
            )
            require(projector_lr == backbone_lr, "Qwen3 projector must match the backbone LR")
        elif training_profile == "dlm_posttrain_sft":
            require(
                (backbone_lr, vision_lr, projector_lr) == (7.0e-6, 7.0e-7, 2.0e-6),
                "DLM post-training SFT must preserve the released 7e-6/7e-7/2e-6 LR contract",
            )
            require(
                str(training["lr_scheduler_type"]) == "cosine_with_min_lr"
                and dict(training.get("lr_scheduler_kwargs", {})) == {"min_lr_rate": 0.1},
                "DLM post-training SFT must preserve cosine_with_min_lr(min_lr_rate=0.1)",
            )
        else:
            raise RuntimeError(f"unsupported GroundAnything Qwen3 training_profile: {training_profile}")
        require(
            float(training["learning_rate"]) == backbone_lr,
            "TrainingArguments base LR must equal Qwen3 backbone LR",
        )
        deepspeed_config = json.loads(
            Path(runtime["deepspeed_config"]).resolve(strict=True).read_text(encoding="utf-8")
        )
        require(
            "optimizer" not in deepspeed_config,
            "Qwen3 differential LR requires a ZeRO config without a DeepSpeed-owned optimizer",
        )
        group_learning_rates = {
            "vision_encoder": vision_lr,
            "projector": projector_lr,
            "language": backbone_lr,
        }
    else:
        require(
            float(training["learning_rate"]) == 5.0e-6,
            "Qwen3.5 Direct Conversion peak learning_rate must be exactly 5e-6",
        )
    mdm_loss_weight = float(training["mdm_loss_weight"])
    causal_loss_weight = float(training["causal_loss_weight"])
    if casuallossenable:
        require(
            (mdm_loss_weight, causal_loss_weight) == (0.5, 0.5),
            "enabled causal loss requires the legacy 0.5/0.5 contract",
        )
    else:
        require(
            (mdm_loss_weight, causal_loss_weight) == (1.0, 0.0),
            "disabled causal loss requires the MDM-only 1.0/0.0 contract",
        )
    require(
        allocator_empty_cache_steps >= 0,
        "allocator_empty_cache_steps must be zero (disabled) or a positive integer",
    )
    require(
        compiler_cache_reset_steps >= 0,
        "compiler_cache_reset_steps must be zero (disabled) or a positive integer",
    )
    if padding_free_packing:
        if model_family == "qwen3_5":
            require(batch_size == 2, "Qwen3.5 DLM padding-free packing is validated only for BS2")
        else:
            require(
                batch_size in range(2, batch_size_limit + 1),
                f"Qwen3 DLM packing is validated for BS2--BS{batch_size_limit}",
            )
        require(language_checkpoint_stride == 0, "packed DLM forbids fine-grained activation checkpointing")
        require(
            packing_language_checkpoint_stride >= 1,
            "packed language checkpoint stride must be positive",
        )
        require(
            not (gradient_checkpointing and packing_activation_cpu_offload),
            "packed DLM checkpointing and saved-tensor CPU offload are mutually exclusive",
        )
        require(
            packing_lm_head_loss_chunk_tokens > 0,
            "packed LM-head loss chunk size must be positive",
        )

    actual_length_audit = audit_actual_length_preflight(config, model_path)
    image_integrity_audit = audit_image_integrity_preflight(config)
    response_block_audit = audit_response_block_preflight(config)
    if local_rank in (-1, 0):
        print(json.dumps({"actual_length_preflight": actual_length_audit}, sort_keys=True), flush=True)
        print(json.dumps({"image_integrity_preflight": image_integrity_audit}, sort_keys=True), flush=True)
        if response_block_audit is not None:
            print(json.dumps({"response_block_preflight": response_block_audit}, sort_keys=True), flush=True)

    set_seed(int(training["seed"]))
    trust_remote_code = model_family == "groundinganything_qwen3"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    configure_processor(processor, int(runtime["image_max_token_num"]), model_family)
    model_loader = AutoModelForCausalLM if model_family == "groundinganything_qwen3" else Qwen3_5ForConditionalGeneration
    model = model_loader.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=str(training["vision_attn_implementation"]),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if model_family == "groundinganything_qwen3":
        require(model.config.model_type == "groundinganything_vlm", "Qwen3 DLM loaded a non-GroundAnything config")
        require(model.config.text_config.model_type == "qwen3", "GroundAnything language backbone is not plain Qwen3")
        require(
            int(model.config.text_config.num_hidden_layers) == 36,
            "GroundAnything Qwen3 language layer count drift",
        )
        require(
            len(model.model.visual.vision_tower.encoder.blocks) == 27,
            "Kimi-K3 ViT layer count drift",
        )
    freeze_vision_encoder = bool(training.get("freeze_vision_encoder", False))
    freeze_projector = bool(training.get("freeze_projector", False))
    vision_gradient_checkpointing = (
        bool(training.get("vision_gradient_checkpointing", False))
        if args.vision_gradient_checkpointing is None
        else args.vision_gradient_checkpointing
    )
    require(
        not (freeze_vision_encoder and vision_gradient_checkpointing),
        "frozen ViT must not enable vision gradient checkpointing",
    )
    vision_checkpoint_audit = configure_vision_gradient_checkpointing(
        model,
        vision_gradient_checkpointing,
        model_family,
    )
    mask_id = initialize_mask_token(processor.tokenizer, model)
    im_end_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    require(im_end_id >= 0, "missing <|im_end|> token")
    wrapper_class = GAMQwen3DLM if model_family == "groundinganything_qwen3" else GAMQwen35DLM
    dlm_model = wrapper_class(
        model,
        mask_token_id=mask_id,
        im_end_token_id=im_end_id,
        block_size=block_size,
        minimum_noise_level=float(training["minimum_noise_level"]),
        allow_nondefault_block_size=block_experiment == "B8",
    )
    dlm_model.set_loss_contract(
        casuallossenable=casuallossenable,
        mdm_loss_weight=mdm_loss_weight,
        causal_loss_weight=causal_loss_weight,
    )
    dlm_model.set_language_checkpoint_stride(language_checkpoint_stride)
    dlm_model.set_padding_free_packing(padding_free_packing)
    dlm_model.set_packing_language_checkpoint_stride(packing_language_checkpoint_stride)
    dlm_model.set_packing_linear_kernel(packing_linear_kernel)
    dlm_model.set_packing_lm_head_loss(
        packing_lm_head_loss_backend,
        packing_lm_head_loss_chunk_tokens,
    )
    dlm_model.set_packing_activation_cpu_offload(
        packing_activation_cpu_offload,
        packing_offload_threshold_mib,
        packing_offload_min_tokens,
        packing_offload_layer_stride,
        packing_offload_pin_memory,
    )
    initial_dlm_audit = None
    if args.initial_dlm_checkpoint is not None:
        initial_dlm_audit = load_initial_dlm_checkpoint(dlm_model, args.initial_dlm_checkpoint)
        if local_rank in (-1, 0):
            print(json.dumps({"initial_dlm_checkpoint": initial_dlm_audit}, sort_keys=True), flush=True)
    dlm_model.train()
    configure_trainable_parameters(
        dlm_model,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_projector=freeze_projector,
    )
    parameter_audit = audit_trainable_parameters(
        dlm_model,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_projector=freeze_projector,
    )
    if local_rank in (-1, 0):
        print(json.dumps({"vision_checkpoint_audit": vision_checkpoint_audit}, sort_keys=True), flush=True)
        print(json.dumps({"parameter_audit": parameter_audit}, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "optimization_contract": {
                        "same_lr_all_trainable_groups": model_family == "qwen3_5",
                        "model_family": model_family,
                        "peak_learning_rate_by_group": {
                            "vision_encoder": None if freeze_vision_encoder else (
                                group_learning_rates or {"vision_encoder": float(training["learning_rate"])}
                            )["vision_encoder"],
                            "projector": None if freeze_projector else (
                                group_learning_rates or {"projector": float(training["learning_rate"])}
                            )["projector"],
                            "language": (
                                group_learning_rates or {"language": float(training["learning_rate"])}
                            )["language"],
                        },
                        "packing_linear_kernel": packing_linear_kernel,
                        "packing_lm_head_loss_backend": packing_lm_head_loss_backend,
                        "packing_lm_head_loss_chunk_tokens": packing_lm_head_loss_chunk_tokens,
                        "packing_language_checkpoint_stride": packing_language_checkpoint_stride,
                        "packed_clean_attention_backend": packed_clean_attention_backend,
                        "packing_activation_cpu_offload": packing_activation_cpu_offload,
                        "packing_offload_threshold_mib": packing_offload_threshold_mib,
                        "packing_offload_min_tokens": packing_offload_min_tokens,
                        "packing_offload_layer_stride": packing_offload_layer_stride,
                        "packing_offload_pin_memory": packing_offload_pin_memory,
                        "allocator_empty_cache_steps": allocator_empty_cache_steps,
                        "compiler_cache_reset_steps": compiler_cache_reset_steps,
                        "allocator_snapshot_steps": allocator_snapshot_steps,
                        "allocator_cleanup_changes_training_dynamics": False,
                        "peak_learning_rate": float(training["learning_rate"]),
                        "casuallossenable": casuallossenable,
                        "mdm_loss_weight": mdm_loss_weight,
                        "causal_loss_weight": causal_loss_weight,
                        "lr_scheduler_type": str(training["lr_scheduler_type"]),
                        "warmup_ratio": float(training["warmup_ratio"]),
                        "optim": str(training["optim"]),
                        "optimizer_foreach": optimizer_foreach,
                        "adam_betas": [float(training["adam_beta1"]), float(training["adam_beta2"])],
                        "adam_epsilon": float(training["adam_epsilon"]),
                        "weight_decay": float(training["weight_decay"]),
                        "max_grad_norm": float(training["max_grad_norm"]),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dataset = IndexedCacheDataset(config["data"]["sampling_manifest"])
    if model_family == "groundinganything_qwen3":
        dataset.attach_packing_workloads(actual_length_audit["packing_workload_lengths"])
    dataset.dlm_complementary_length_sampling = padding_free_packing
    dataset.dlm_packing_manifest = args.packing_manifest or config["data"].get("packing_manifest")
    if padding_free_packing:
        require(bool(dataset.dlm_packing_manifest), "DLM packing_manifest is required")
    require(len(dataset) == int(config["data"]["sampled_rows"]), "sampled row count drift")
    collator = DLMDataCollator(
        processor,
        max_length=int(runtime["max_length"]),
        max_joint_length=int(runtime["max_joint_length"]),
        model_family=model_family,
    )
    training_args = build_training_args(
        config,
        output_dir,
        batch_size,
        accumulation,
        gradient_checkpointing,
        args.max_steps,
        args.disable_checkpointing,
        args.expected_global_batch_size,
    )
    callbacks: list[TrainerCallback] = []
    if allocator_snapshot_steps:
        callbacks.append(AllocatorSnapshotDeltaCallback(allocator_snapshot_steps))
    if compiler_cache_reset_steps:
        callbacks.append(CompilerCacheCleanupCallback(compiler_cache_reset_steps))
    if allocator_empty_cache_steps:
        callbacks.append(AllocatorCacheCleanupCallback(allocator_empty_cache_steps))
    if args.stop_after_step is not None:
        callbacks.append(StopAfterStepCallback(args.stop_after_step))
    trainer = DLMTrainer(
        model=dlm_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        # GroundAnythingVLMProcessor is intentionally lightweight and does not
        # implement ProcessorMixin.save_pretrained().  Trainer only needs a
        # serializable processing class for checkpoint metadata; the collator
        # keeps using the full multimodal processor above.  Saving the
        # tokenizer here preserves every train/resume input while avoiding a
        # checkpoint-time AttributeError.  This has no forward/backward or
        # optimizer effect.
        processing_class=processor.tokenizer,
        callbacks=callbacks,
        group_learning_rates=group_learning_rates,
        optimizer_foreach=optimizer_foreach,
    )
    require(
        trainer.model_accepts_loss_kwargs is False,
        "Trainer must apply standard gradient-accumulation loss normalization",
    )
    if trainer.is_world_process_zero():
        print(
            json.dumps(
                {
                    "loss_scaling_contract": {
                        "status": "PASS",
                        "trainer_divides_by_gradient_accumulation_steps": True,
                        "gradient_accumulation_steps": accumulation,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
    profiler = None
    profile_dir_value = os.environ.get("GAM_DLM_TORCH_PROFILE_DIR")
    if profile_dir_value and trainer.is_world_process_zero():
        profile_dir = Path(profile_dir_value).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_wait = int(os.environ.get("GAM_DLM_PROFILE_WAIT", "2"))
        profile_warmup = int(os.environ.get("GAM_DLM_PROFILE_WARMUP", "1"))
        profile_active = int(os.environ.get("GAM_DLM_PROFILE_ACTIVE", "1"))
        profile_last_active_only = os.environ.get(
            "GAM_DLM_PROFILE_LAST_ACTIVE_ONLY", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if profile_last_active_only and profile_active > 1:
            # Preserve the last requested profiled optimizer step while avoiding
            # a multi-GiB event set that makes every other rank wait on rank 0.
            profile_wait += profile_active - 1
            profile_active = 1
        profile_export_trace = os.environ.get(
            "GAM_DLM_PROFILE_EXPORT_TRACE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        require(profile_wait >= 0, "profile wait must be non-negative")
        require(profile_warmup >= 1, "profile warmup must be positive")
        require(profile_active >= 1, "profile active steps must be positive")

        def save_profile(profile: Any) -> None:
            trace_path = profile_dir / "rank0-attention-trace.json"
            rows = []
            for event in profile.key_averages():
                rows.append(
                    {
                        "key": event.key,
                        "calls": int(event.count),
                        "self_cpu_time_total_us": float(event.self_cpu_time_total),
                        "cpu_time_total_us": float(event.cpu_time_total),
                        "self_cuda_time_total_us": float(
                            getattr(event, "self_device_time_total", 0.0)
                        ),
                        "cuda_time_total_us": float(getattr(event, "device_time_total", 0.0)),
                    }
                )
            rows.sort(key=lambda row: row["self_cuda_time_total_us"], reverse=True)
            summary = {
                "status": "PASS",
                "rank": 0,
                "schedule": {
                    "wait": profile_wait,
                    "warmup": profile_warmup,
                    "active": profile_active,
                    "repeat": 1,
                },
                "packed_clean_attention_backend": os.environ.get(
                    "GAM_DLM_PACKED_CLEAN_ATTENTION_BACKEND",
                    "flash_attention_2",
                ),
                "top_cuda_events": rows[:500],
                "trace": str(trace_path) if profile_export_trace else None,
            }
            summary_path = profile_dir / "rank0-attention-kernel-summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps({"attention_profile": summary}, sort_keys=True), flush=True)
            if profile_export_trace:
                profile.export_chrome_trace(str(trace_path))

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=profile_wait,
                warmup=profile_warmup,
                active=profile_active,
                repeat=1,
            ),
            on_trace_ready=save_profile,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()
        trainer.add_callback(TorchProfilerStepCallback(profiler))
    try:
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        if profiler is not None:
            profiler.stop()
    if not args.disable_checkpointing:
        require(
            bool(training.get("save_final_full_checkpoint", False)),
            "DLM formal runs require save_final_full_checkpoint=true",
        )
        final_checkpoint = Path(output_dir) / f"checkpoint-{trainer.state.global_step}"
        final_exists = torch.tensor(
            int(final_checkpoint.is_dir()),
            dtype=torch.int32,
            device=training_args.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(final_exists, op=torch.distributed.ReduceOp.MIN)
        if not bool(final_exists.item()):
            # Transformers 5.2's full checkpoint path saves model, ZeRO-2
            # optimizer, scheduler, scaler, RNG and trainer state.  All ranks
            # must participate for a checkpoint to be genuinely resumable.
            trainer._save_checkpoint(trainer.model, trial=None)
    if trainer.is_world_process_zero():
        metrics = {**result.metrics, "mask_token_id": mask_id, "mode": "DLM"}
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        # Every rank must finish CUDA work and all durable checkpoint/metric
        # writes before NCCL teardown.  The pinned H800 CUDA 12.9 image has an
        # intermittent libnvidia-ptxjitcompiler interpreter-destruction race:
        # training and checkpointing finish successfully, then one rank can
        # SIGSEGV while Python unloads CUDA libraries.  A final collective
        # proves that every rank reached the success path; the optional hard
        # exit below then bypasses only Python/C++ static destructors.
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
    if os.environ.get("GAM_DLM_HARD_EXIT_AFTER_SUCCESS") == "1":
        print(
            json.dumps(
                {
                    "successful_exit_contract": {
                        "checkpoint_and_metrics_closed": True,
                        "cuda_synchronized": True,
                        "distributed_success_barrier": True,
                        "python_static_destructors_bypassed": True,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
