"""24-H800 Causal-JustGRPO trainer for GAM DLM RL V2.

Rollout and teacher forcing are both ordinary causal policies.  The DLM
wrapper is retained as the trainable checkpoint format; DecodeV4 is used only
for downstream evaluation.  This module is isolated from the TraceRL V1
implementation by design.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoProcessor

from train.rl.release_data import load_mixture
from train.rl.corrected_data import CorrectedGroundingOCRMixture
from train.rl.data import GroundAnythingPromptEncoder
from train.rl.loss import group_normalized_advantage, tracerl_token_loss
from train.rl.reward_adapter import JointGAMRewardAdapter, RewardResult, gdpo_total_rewards
from train.rl.current.config import CausalJustGRPOConfig, load_config
from train.rl.current.rollout import CausalRollout, causal_teacher_forcing_logprobs
from models.dlm.hybrid import initialize_mask_token
from models.dlm.vlm import GAMQwen3DLM
from train.dlm.train_dlm import (
    audit_trainable_parameters,
    configure_processor,
    configure_trainable_parameters,
    load_initial_dlm_checkpoint,
)


DATA_KIND = "grounding_ocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grounding-data", type=Path)
    parser.add_argument("--ocr-data", type=Path)
    parser.add_argument("--multiroute-data", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stable-gate", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def distributed_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def build_generation_group(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 24 or group_size != 8 or world_size % group_size:
        raise RuntimeError(
            f"DLM RL V2 requires world24/group8, got world={world_size} group={group_size}"
        )
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to create DLM RL V2 generation group")
    return own_group, group_index, rank % group_size, world_size // group_size


def build_model(
    model_path: Path,
    initial_checkpoint: Path,
    device: torch.device,
    config: CausalJustGRPOConfig,
) -> tuple[GAMQwen3DLM, Any, dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    configure_processor(processor, 1024, "groundinganything_qwen3")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if base.config.model_type != "groundinganything_vlm" or base.config.text_config.model_type != "qwen3":
        raise RuntimeError("DLM RL V2 loaded a non GroundAnything/Qwen3 model")
    mask_id = initialize_mask_token(processor.tokenizer, base)
    eos_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if eos_id < 0:
        raise RuntimeError("missing GroundAnything <|im_end|> token")
    model = GAMQwen3DLM(
        base,
        mask_token_id=mask_id,
        im_end_token_id=eos_id,
        block_size=32,
        minimum_noise_level=0.0,
    )
    load_audit = load_initial_dlm_checkpoint(model, initial_checkpoint)
    configure_trainable_parameters(
        model,
        freeze_vision_encoder=config.freeze_vision_encoder,
        freeze_projector=config.freeze_projector,
    )
    trainable_audit = audit_trainable_parameters(
        model,
        freeze_vision_encoder=config.freeze_vision_encoder,
        freeze_projector=config.freeze_projector,
    )
    # Standard causal teacher forcing runs through the native Qwen3 path.  Its
    # checkpoint switch is separate from TraceRL's custom hybrid loop.
    model.language_model.gradient_checkpointing = True
    model.language_model._gradient_checkpointing_func = partial(
        checkpoint,
        use_reentrant=False,
    )
    model.to(device)
    return model, processor, {"load": load_audit, "trainable": trainable_audit}


def optimizer_groups(
    model: torch.nn.Module,
    config: CausalJustGRPOConfig,
) -> list[dict[str, Any]]:
    no_decay_fragments = ("bias", "norm.weight", "layernorm.weight", "embed_tokens.weight")
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if any(item in name.lower() for item in no_decay_fragments) else decay
        target.append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("DLM RL V2 optimizer parameter groups are empty")
    return [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def deepspeed_config(config: CausalJustGRPOConfig) -> dict[str, Any]:
    return {
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "none"},
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 100_000_000,
            "allgather_bucket_size": 100_000_000,
        },
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        "train_micro_batch_size_per_gpu": config.per_device_prompt_batch_size,
        "train_batch_size": 24,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
    }


def visible_response(tokenizer: Any, ids: list[int], eos_id: int) -> str:
    end = len(ids) - 1 if ids and ids[-1] == int(eos_id) else len(ids)
    return tokenizer.decode(ids[:end], skip_special_tokens=False)


def gather_reward_group(
    local: RewardResult,
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[float, float, float]:
    gathered: list[RewardResult | None] = [None] * 8
    dist.all_gather_object(gathered, local, group=group)
    rewards = [item for item in gathered if item is not None]
    if len(rewards) != 8:
        raise RuntimeError("incomplete DLM RL V2 reward group")
    weights = rewards[0].weights
    if any(item.weights != weights for item in rewards):
        raise RuntimeError("reward route drift within DLM RL V2 group")
    totals = gdpo_total_rewards([item.components for item in rewards], weights).to(device)
    advantages = group_normalized_advantage(totals)
    local_rank = dist.get_rank(group)
    return (
        float(totals[local_rank].item()),
        float(advantages[local_rank].item()),
        float(totals.std(unbiased=False).item()),
    )


def parameter_probe(model: torch.nn.Module) -> tuple[list[str], list[torch.Tensor]]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.numel()
    ]
    if not candidates:
        raise RuntimeError("DLM RL V2 has no trainable parameter")
    indices = sorted(
        {0, len(candidates) // 4, len(candidates) // 2, 3 * len(candidates) // 4, len(candidates) - 1}
    )
    return (
        [candidates[index][0] for index in indices],
        [
            candidates[index][1].detach().flatten()[:4096].float().cpu().clone()
            for index in indices
        ],
    )


def save_checkpoint(
    engine: Any,
    output_dir: Path,
    optimizer_step: int,
    client_state: dict[str, Any],
) -> Path:
    checkpoint = output_dir / f"checkpoint-{optimizer_step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(checkpoint), tag="deepspeed", client_state=client_state)
    dist.barrier()
    if dist.get_rank() == 0:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in engine.module.state_dict().items()
        }
        save_file(state, str(checkpoint / "model.safetensors"))
        atomic_json(checkpoint / "trainer_state.json", client_state)
    dist.barrier()
    return checkpoint


def load_resume(engine: Any, checkpoint: Path, config: CausalJustGRPOConfig) -> int:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"missing DLM RL V2 trainer state: {state_path}")
    recorded = json.loads(state_path.read_text(encoding="utf-8"))
    if recorded.get("config") != asdict(config):
        raise ValueError("DLM RL V2 resume config drift")
    loaded, client_state = engine.load_checkpoint(
        str(checkpoint),
        tag="deepspeed",
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
        load_module_only=False,
    )
    if loaded is None or client_state is None:
        raise RuntimeError(f"DeepSpeed failed to resume DLM RL V2 from {checkpoint}")
    step = int(client_state.get("optimizer_step", -1))
    if step != int(recorded.get("optimizer_step", -2)):
        raise ValueError("DLM RL V2 checkpoint/client optimizer step mismatch")
    return step


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mixture = load_mixture(args, config.seed, DATA_KIND, factory=CorrectedGroundingOCRMixture)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        raise RuntimeError("DLM RL V2 must be launched by torchrun")
    timeout_seconds = int(os.environ.get("GAM_RL_DIST_TIMEOUT_SECONDS", "7200"))
    if timeout_seconds < 600:
        raise ValueError("GAM_RL_DIST_TIMEOUT_SECONDS must be at least 600")
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(
        dist_backend="nccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    group, group_index, group_local_rank, groups_per_step = build_generation_group(
        config.num_generations,
        timeout=timedelta(seconds=timeout_seconds),
    )
    seed = config.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    expected_full_epoch_steps = math.ceil(len(mixture) / groups_per_step)
    if (
        config.training_budget == "full_dataset_epoch"
        and config.total_optimizer_steps != expected_full_epoch_steps
    ):
        raise RuntimeError(
            "DLM RL V2 full-epoch budget does not cover the corrected data exactly: "
            f"rows={len(mixture)} groups_per_step={groups_per_step} "
            f"steps={config.total_optimizer_steps} expected={expected_full_epoch_steps}"
        )
    if rank == 0:
        mixture.write_audit(args.output_dir / "corrected-data-audit.json")
        atomic_json(
            args.output_dir / "contract.json",
            {
                "status": "PASS",
                "created_at_utc": utc_now(),
                "implementation": "Causal-JustGRPO",
                "version": "V2",
                "run_kind": args.run_kind,
                "world_size": world_size,
                "groups_per_step": groups_per_step,
                "group_size": config.num_generations,
                "dataset_rows": len(mixture),
                "full_epoch_optimizer_steps": expected_full_epoch_steps,
                "scheduled_prompt_slots": config.total_optimizer_steps * groups_per_step,
                "repeated_prompt_slots_at_epoch_tail": (
                    config.total_optimizer_steps * groups_per_step - len(mixture)
                    if config.training_budget == "full_dataset_epoch"
                    else None
                ),
                "config": asdict(config),
                "initial_checkpoint": str(args.initial_checkpoint),
                "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
                "model_path": str(args.model_path),
                "data_audit_sha256": mixture.audit["ordered_effective_ids_sha256"],
                "no_dlm_trace_replay": True,
                "inference_after_training": "DecodeV4",
            },
        )

    model, processor, model_audit = build_model(
        args.model_path,
        args.initial_checkpoint,
        device,
        config,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, config),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        fused=True,
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        optimizer=optimizer,
        config=deepspeed_config(config),
    )
    start_step = load_resume(engine, args.resume_checkpoint, config) if args.resume_checkpoint else 0
    if rank == 0:
        print(
            json.dumps(
                {
                    "dlm_rl_v2_model_audit": model_audit,
                    "optimizer": {
                        "dtype": "float32",
                        "lr": config.learning_rate,
                        "scheduler": config.scheduler,
                        "zero_stage": 1,
                    },
                    "resume_optimizer_step": start_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model = engine.module
    encoder = GroundAnythingPromptEncoder(processor)
    reward_adapter = JointGAMRewardAdapter()
    rollout = CausalRollout(model, config, model.im_end_token_id)
    target_steps = config.total_optimizer_steps
    if args.max_optimizer_steps is not None:
        if args.max_optimizer_steps <= 0:
            raise ValueError("max optimizer steps must be positive")
        target_steps = min(target_steps, args.max_optimizer_steps)
    if args.run_kind == "smoke":
        target_steps = min(target_steps, 1)
    if start_step >= target_steps:
        raise ValueError(f"resume step {start_step} must be below target {target_steps}")

    start_time = time.monotonic()
    last_checkpoint: Path | None = args.resume_checkpoint
    for optimizer_step in range(start_step, target_steps):
        row = mixture.row_for_group(optimizer_step, group_index, groups_per_step)
        prompt = encoder(row).to(device)
        trajectory = rollout.generate(
            prompt,
            seed=config.seed + optimizer_step * 100_000 + group_index * 100 + group_local_rank,
            prompt_id=str(row["id"]),
            policy_version=f"v2-step-{optimizer_step}",
        )
        response = visible_response(
            processor.tokenizer,
            trajectory.completion_ids,
            model.im_end_token_id,
        )
        local_reward = reward_adapter.score(response, row)
        reward, advantage, group_reward_std = gather_reward_group(local_reward, group, device)

        # The cached incremental rollout and the full-prefix learner can use
        # different Hopper attention kernels.  Their BF16 log-probabilities are
        # therefore useful sampler telemetry, but are not a numerically exact
        # PPO old/current pair.  Re-score the sampled trajectory once with the
        # exact teacher-forcing path *before* the update and use that detached
        # tensor as old-policy log-probabilities.  The response distribution is
        # unchanged; this only makes the PPO ratio compare identical causal
        # factorizations and identical model versions.
        model.eval()
        engine.zero_grad()
        engine.set_gradient_accumulation_boundary(True)
        before_names, before_values = parameter_probe(model)
        sampler_logprobs = torch.tensor(
            trajectory.old_logprobs,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            old_logprobs = causal_teacher_forcing_logprobs(
                model,
                prompt,
                trajectory,
                config,
            ).detach()
        current_logprobs = causal_teacher_forcing_logprobs(model, prompt, trajectory, config)
        sampler_teacher_mean_abs = float(
            (sampler_logprobs - old_logprobs).abs().mean().item()
        )
        sampler_teacher_p99_abs = float(
            torch.quantile((sampler_logprobs - old_logprobs).abs(), 0.99).item()
        )
        loss, loss_telemetry = tracerl_token_loss(
            current_logprobs,
            old_logprobs,
            advantage,
            clip_epsilon=config.clip_epsilon,
        )
        engine.backward(loss)
        engine.step()
        grad_norm_raw = engine.get_global_grad_norm()
        if grad_norm_raw is None:
            raise RuntimeError("DeepSpeed did not publish DLM RL V2 gradient norm")
        grad_norm = float(grad_norm_raw)
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite DLM RL V2 gradient")
        after_names, after_values = parameter_probe(model)
        if after_names != before_names:
            raise RuntimeError("DLM RL V2 parameter probe ordering drift")
        parameter_delta = max(
            float((after - before).abs().max().item())
            for before, after in zip(before_values, after_values, strict=True)
        )

        ratio_mean = float(loss_telemetry["ratio_mean"].item())
        mean_abs = float(loss_telemetry["mean_abs_logprob_difference"].item())
        p99_abs = float(loss_telemetry["p99_abs_logprob_difference"].item())
        clip_fraction = float(loss_telemetry["clip_fraction"].item())
        ratio_error_max = distributed_max(abs(ratio_mean - 1.0), device)
        mean_abs_max = distributed_max(mean_abs, device)
        clip_max = distributed_max(clip_fraction, device)
        nonzero_delta_rate = distributed_mean(float(parameter_delta > 0.0), device)
        if optimizer_step == start_step:
            if ratio_error_max > 0.03 or mean_abs_max > 0.03 or clip_max > 0.10:
                raise RuntimeError(
                    "DLM RL V2 old/current ratio parity failed: "
                    f"ratio_error={ratio_error_max} mean_abs={mean_abs_max} clip={clip_max}"
                )
            if nonzero_delta_rate != 1.0:
                raise RuntimeError(
                    "DLM RL V2 first optimizer step did not update every ZeRO replica: "
                    f"rate={nonzero_delta_rate}"
                )
            if rank == 0:
                atomic_json(
                    args.stable_gate,
                    {
                        "status": "PASS",
                        "created_at_utc": utc_now(),
                        "optimizer_step": optimizer_step + 1,
                        "ratio_mean": ratio_mean,
                        "mean_abs_logprob_difference": mean_abs,
                        "p99_abs_logprob_difference": p99_abs,
                        "clip_fraction": clip_fraction,
                        "sampler_teacher_mean_abs_logprob_difference": sampler_teacher_mean_abs,
                        "sampler_teacher_p99_abs_logprob_difference": sampler_teacher_p99_abs,
                        "gradient_norm": grad_norm,
                        "parameter_max_delta": parameter_delta,
                        "run_id": os.environ.get("GAM_RUN_ID", ""),
                    },
                )

        metrics = {
            "timestamp_utc": utc_now(),
            "optimizer_step": optimizer_step + 1,
            "total_optimizer_steps": target_steps,
            "task_type": str(row.get("task_type")),
            "ocr_rank_fraction": distributed_mean(
                float(row.get("task_type") == "ocr_bbox_text_gam"), device
            ),
            "reward_mean": distributed_mean(reward, device),
            "policy_loss_mean": distributed_mean(float(loss.detach().item()), device),
            "absolute_advantage_mean": distributed_mean(abs(advantage), device),
            "group_reward_std_mean": distributed_mean(group_reward_std, device),
            "format_valid_rate": distributed_mean(float(local_reward.format_valid), device),
            "output_tokens_mean": distributed_mean(len(trajectory.completion_ids), device),
            "eos_rate": distributed_mean(float(trajectory.stop_reason == "eos"), device),
            "ratio_mean": ratio_mean,
            "clip_fraction": clip_fraction,
            "mean_abs_logprob_difference": mean_abs,
            "p99_abs_logprob_difference": p99_abs,
            "sampler_teacher_mean_abs_logprob_difference": sampler_teacher_mean_abs,
            "sampler_teacher_p99_abs_logprob_difference": sampler_teacher_p99_abs,
            "gradient_norm": grad_norm,
            "parameter_max_delta": parameter_delta,
            "learning_rate": config.learning_rate,
            "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        torch.cuda.reset_peak_memory_stats(device)
        if rank == 0:
            append_jsonl(log_path, metrics)
            print(json.dumps({"causal_justgrpo_v2": metrics}, sort_keys=True), flush=True)

        completed = optimizer_step + 1
        should_save = (
            args.run_kind == "smoke"
            or completed % config.save_steps == 0
            or completed == target_steps
        )
        if should_save:
            last_checkpoint = save_checkpoint(
                engine,
                args.output_dir,
                completed,
                {
                    "algorithm": "Causal-JustGRPO",
                    "version": "V2",
                    "config": asdict(config),
                    "metrics": metrics,
                    "optimizer_step": completed,
                },
            )

    if rank == 0:
        atomic_json(
            args.final_gate,
            {
                "status": "PASS",
                "completed_at_utc": utc_now(),
                "optimizer_steps": target_steps,
                "final_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                "run_id": os.environ.get("GAM_RUN_ID", ""),
            },
        )
    dist.barrier()


if __name__ == "__main__":
    main()
