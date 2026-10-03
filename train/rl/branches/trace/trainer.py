"""64-H800 fixed-scheduler GAM-TraceRL trainer.

The implementation intentionally has no AR completion path: rollout, replay
and loss all operate on the real B32 GAM diffusion commit trace.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
from torch.distributed import distributed_c10d
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoProcessor

from train.rl.config import TraceRLConfig, load_config
from train.rl.data import GroundingOCRMixture, GroundAnythingPromptEncoder
from train.rl.loss import group_normalized_advantage, tracerl_token_loss
from train.rl.replay import (
    ReplayTransition,
    ReplayPromptCache,
    bucket_replay_transitions,
    build_replay_transitions,
    differentiable_transition_batch_logprobs,
)
from train.rl.reward_adapter import JointGAMRewardAdapter, RewardResult, gdpo_total_rewards
from train.rl.resume import validate_resume_checkpoint, validate_resume_config
from train.rl.rollout import GAMTraceRollout
from models.dlm.hybrid import initialize_mask_token
from models.dlm.vlm import GAMQwen3DLM
from train.dlm.train_dlm import (
    configure_processor,
    configure_trainable_parameters,
    load_initial_dlm_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stable-gate", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--replay-token-budget", type=int, default=24576)
    parser.add_argument("--maximum-replay-batch-size", type=int, default=1)
    parser.add_argument("--temperature-adjustment-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--restart-index", type=int, choices=(0, 1), default=0)
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


def build_process_groups(
    group_size: int,
    *,
    timeout: timedelta,
) -> tuple[dist.ProcessGroup, int, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 64 or group_size != 8 or world_size % group_size:
        raise RuntimeError(f"TraceRL requires world64/group8, got world={world_size} group={group_size}")
    own_group = None
    group_index = rank // group_size
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        # ``init_process_group`` timeout applies only to the default group.
        # TraceRL creates eight generation/reward subgroups; leaving their
        # timeout implicit made the image's 600-second default kill healthy
        # ranks while one member replayed an extreme trajectory.  This changes
        # no rollout, replay, loss, optimizer, or collective ordering—it only
        # gives the existing collective the same operational timeout contract
        # as the default DeepSpeed group.
        process_group = dist.new_group(ranks=ranks, timeout=timeout)
        if rank in ranks:
            own_group = process_group
    if own_group is None:
        raise RuntimeError("failed to construct TraceRL process group")
    return own_group, group_index, rank % group_size, world_size // group_size


def align_process_group_timeouts(
    timeout: timedelta,
    *process_groups: dist.ProcessGroup | None,
) -> int:
    """Apply one timeout contract to already-created NCCL process groups.

    DeepSpeed ZeRO-1 clones WORLD with ``dist.new_group``.  That clone does not
    inherit the timeout passed to ``deepspeed.init_distributed`` and used the
    image default of 600 seconds in task 289.  Set the timeout on the actual
    engine groups after initialization; this is an operational liveness change
    only and leaves all tensors and collective ordering untouched.
    """

    setter = getattr(distributed_c10d, "_set_pg_timeout", None)
    if setter is None:
        raise RuntimeError("PyTorch runtime lacks distributed_c10d._set_pg_timeout")
    unique: list[dist.ProcessGroup] = []
    seen: set[int] = set()
    for process_group in process_groups:
        if process_group is None or process_group == dist.GroupMember.NON_GROUP_MEMBER:
            continue
        identity = id(process_group)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(process_group)
    if not unique:
        raise RuntimeError("no TraceRL process groups available for timeout alignment")
    for process_group in unique:
        setter(timeout, process_group)
    return len(unique)


def build_model(
    model_path: Path,
    initial_checkpoint: Path,
    device: torch.device,
) -> tuple[GAMQwen3DLM, Any]:
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
        raise RuntimeError("TraceRL loaded a non GroundAnything/Qwen3 checkpoint")
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
    audit = load_initial_dlm_checkpoint(model, initial_checkpoint)
    configure_trainable_parameters(model, freeze_vision_encoder=True, freeze_projector=True)
    # Only the custom hybrid language loop is checkpointed.  The frozen K3
    # ViT and projector stay outside the backward graph.
    model.gradient_checkpointing = True
    model._gradient_checkpointing_kwargs = {"use_reentrant": False}
    model.to(device)
    if dist.get_rank() == 0:
        print(json.dumps({"initial_dlm_checkpoint": audit}, sort_keys=True), flush=True)
    return model, processor


def optimizer_groups(model: torch.nn.Module, config: TraceRLConfig) -> list[dict[str, Any]]:
    no_decay_fragments = ("bias", "norm.weight", "layernorm.weight", "embed_tokens.weight")
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if any(fragment in name.lower() for fragment in no_decay_fragments) else decay
        target.append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("TraceRL optimizer parameter grouping is empty")
    return [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def normalize_loaded_optimizer_step_devices(engine: Any) -> int:
    """Move scalar Adam step tensors to their owning CUDA parameter.

    DeepSpeed Universal-checkpoint loading can leave AdamW's scalar ``step``
    entry on CPU even with optimizer offload disabled.  Fused CUDA AdamW
    requires the corresponding ``state_steps`` tensor to be on the same
    device as the parameter.  Only scalar step entries are moved here; the
    large moment tensors retain their normal DeepSpeed placement.
    """
    zero_optimizer = getattr(engine, "optimizer", None)
    base_optimizer = getattr(zero_optimizer, "optimizer", None)
    if base_optimizer is None or not hasattr(base_optimizer, "state"):
        raise RuntimeError("cannot inspect the underlying AdamW state after resume")
    moved = 0
    for parameter, state in base_optimizer.state.items():
        if not isinstance(state, dict) or parameter.device.type != "cuda":
            continue
        for key in ("step", "state_steps"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.device != parameter.device:
                if value.ndim != 0:
                    raise RuntimeError(
                        f"unexpected non-scalar Adam step state: key={key} shape={tuple(value.shape)}"
                    )
                state[key] = value.to(device=parameter.device, non_blocking=True)
                moved += 1
    return moved


def deepspeed_config(config: TraceRLConfig) -> dict[str, Any]:
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
        "gradient_accumulation_steps": 1,
        "gradient_clipping": config.max_grad_norm,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 64,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
    }


def visible_response(tokenizer: Any, output_ids: list[int], eos_id: int) -> str:
    try:
        end = output_ids.index(int(eos_id))
    except ValueError:
        end = len(output_ids)
    return tokenizer.decode(output_ids[:end], skip_special_tokens=False)


def gather_reward_group(
    local: RewardResult,
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[float, float, list[RewardResult]]:
    gathered: list[RewardResult | None] = [None] * 8
    dist.all_gather_object(gathered, local, group=group)
    rewards = [item for item in gathered if item is not None]
    if len(rewards) != 8:
        raise RuntimeError("incomplete TraceRL reward group")
    weights = rewards[0].weights
    if any(item.weights != weights for item in rewards):
        raise RuntimeError("reward weights drift within one prompt group")
    totals = gdpo_total_rewards([item.components for item in rewards], weights).to(device)
    advantages = group_normalized_advantage(totals)
    local_rank = dist.get_rank(group)
    return float(totals[local_rank].item()), float(advantages[local_rank].item()), rewards


def parameter_probe(model: torch.nn.Module) -> tuple[list[str], list[torch.Tensor]]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.numel()
    ]
    if not candidates:
        raise RuntimeError("no trainable TraceRL parameter")
    indices = sorted({0, len(candidates) // 4, len(candidates) // 2, 3 * len(candidates) // 4, len(candidates) - 1})
    names = [candidates[index][0] for index in indices]
    values = [
        candidates[index][1].detach().flatten()[:4096].float().cpu().clone()
        for index in indices
    ]
    return names, values


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


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0:
        raise RuntimeError("TraceRL must be launched by torchrun")
    distributed_timeout_seconds = int(os.environ.get("GAM_RL_DIST_TIMEOUT_SECONDS", "7200"))
    if distributed_timeout_seconds < 600:
        raise ValueError("GAM_RL_DIST_TIMEOUT_SECONDS must be at least 600")
    torch.cuda.set_device(local_rank)
    # TraceRL ranks replay variable-length diffusion trajectories before the
    # sole ZeRO-1 reduction.  The framework default (600 s) incorrectly kills
    # healthy ranks when another rank is replaying an extreme long trajectory.
    deepspeed.init_distributed(
        dist_backend="nccl",
        timeout=timedelta(seconds=distributed_timeout_seconds),
    )
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = load_config(args.config, temperature=args.temperature)
    resume = None
    if args.resume_checkpoint is not None:
        resume = validate_resume_checkpoint(args.resume_checkpoint, expected_world_size=world_size)
        validate_resume_config(resume.trainer_state.get("config", {}), asdict(config))
    group, group_index, group_local_rank, groups_per_step = build_process_groups(
        config.num_generations,
        timeout=timedelta(seconds=distributed_timeout_seconds),
    )
    seed = config.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    if rank == 0:
        atomic_json(
            args.output_dir / "contract.json",
            {
                "status": "PASS",
                "created_at_utc": utc_now(),
                "run_kind": args.run_kind,
                "world_size": world_size,
                "groups_per_step": groups_per_step,
                "group_size": config.num_generations,
                "config": asdict(config),
                "initial_checkpoint": str(args.initial_checkpoint),
                "resume_checkpoint": str(resume.path) if resume else None,
                "resume_optimizer_step": resume.optimizer_step if resume else 0,
                "model_path": str(args.model_path),
                "replay_token_budget": args.replay_token_budget,
                "maximum_replay_batch_size": args.maximum_replay_batch_size,
                "distributed_timeout_seconds": distributed_timeout_seconds,
            },
        )

    model, processor = build_model(args.model_path, args.initial_checkpoint, device)
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups,
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
    timeout_group_count = align_process_group_timeouts(
        timedelta(seconds=distributed_timeout_seconds),
        dist.group.WORLD,
        group,
        engine.data_parallel_group,
        engine.seq_data_parallel_group,
        getattr(engine.optimizer, "dp_process_group", None),
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "process_group_timeout_audit": {
                        "status": "PASS",
                        "timeout_seconds": distributed_timeout_seconds,
                        "unique_groups": timeout_group_count,
                        "deepspeed_data_parallel_clone_aligned": True,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
    start_optimizer_step = 0
    if resume is not None:
        load_path, client_state = engine.load_checkpoint(
            str(resume.path),
            tag=resume.tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
            load_module_only=False,
        )
        if load_path is None or client_state is None:
            raise RuntimeError(f"DeepSpeed failed to load resume checkpoint: {resume.path}")
        loaded_step = int(client_state.get("optimizer_step", -1))
        if loaded_step != resume.optimizer_step:
            raise RuntimeError(
                "DeepSpeed client-state step mismatch: "
                f"loaded={loaded_step} validated={resume.optimizer_step}"
            )
        validate_resume_config(client_state.get("config", {}), asdict(config))
        start_optimizer_step = loaded_step
        if os.environ.get("GAM_RL_NORMALIZE_OPTIMIZER_STEP_DEVICE", "0") == "1":
            moved_steps = normalize_loaded_optimizer_step_devices(engine)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "optimizer_state_device_audit": {
                                "status": "PASS",
                                "moved_scalar_step_tensors": moved_steps,
                                "target_device": str(device),
                                "fused_adamw": True,
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "resume_audit": {
                            "status": "PASS",
                            "checkpoint": str(resume.path),
                            "tag": resume.tag,
                            "optimizer_step": start_optimizer_step,
                            "deepspeed_global_steps": int(engine.global_steps),
                            "optimizer_state": "restored",
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model = engine.module
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    if rank == 0:
        print(
            json.dumps(
                {
                    "parameter_audit": {
                        "status": "PASS",
                        "trainable_language": trainable,
                        "frozen_vit_projector": frozen,
                        "dtype": "bfloat16",
                        "optimizer_state_dtype": "float32",
                        "deepspeed_zero_stage": 1,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    mixture = GroundingOCRMixture(config.seed)
    encoder = GroundAnythingPromptEncoder(processor)
    reward_adapter = JointGAMRewardAdapter()
    rollout = GAMTraceRollout(model, config, model.im_end_token_id)
    dataset_total_steps = math.ceil(len(mixture) / groups_per_step)
    total_steps = dataset_total_steps
    if args.max_optimizer_steps is not None:
        if args.max_optimizer_steps <= 0:
            raise ValueError("max optimizer steps must be positive")
        total_steps = min(total_steps, args.max_optimizer_steps)
    if args.run_kind == "smoke":
        # A bounded resumed run advances from its restored optimizer step.
        if resume is not None:
            # Compute the target from the checkpoint rather than an absolute step.
            extra_steps = int(os.environ.get("GAM_RL_SMOKE_EXTRA_STEPS", "1"))
            if extra_steps <= 0:
                raise ValueError("GAM_RL_SMOKE_EXTRA_STEPS must be positive")
            # ``--max-optimizer-steps 1`` is intentionally present in the
            # launcher for a fresh smoke.  Once resuming, it must not cap the
            # absolute target back to one; use the full dataset horizon and
            # stop after exactly ``extra_steps`` new steps.
            total_steps = min(dataset_total_steps, start_optimizer_step + extra_steps)
        else:
            total_steps = min(total_steps, 1)
    if start_optimizer_step >= total_steps:
        raise ValueError(
            f"resume step {start_optimizer_step} must be below target step {total_steps}"
        )

    start_time = time.monotonic()
    group_count = 0
    zero_variance_groups = 0
    invalid_rollouts = 0
    rollout_count = 0
    last_checkpoint: Path | None = resume.path if resume else None
    first_step_checked = False
    completed_steps = start_optimizer_step

    for optimizer_step in range(start_optimizer_step, total_steps):
        row = mixture.row_for_group(optimizer_step, group_index, groups_per_step)
        prompt = encoder(row).to(device)
        policy_version = f"restart-{args.restart_index}-step-{optimizer_step}"
        trace = rollout.generate(
            prompt,
            seed=config.seed + optimizer_step * 100_000 + group_index * 100 + group_local_rank,
            prompt_id=str(row.get("id", f"group-{group_index}-step-{optimizer_step}")),
            policy_version=policy_version,
        )
        response = visible_response(processor.tokenizer, trace.output_ids, model.im_end_token_id)
        local_reward = reward_adapter.score(response, row)
        reward_total, advantage, reward_group = gather_reward_group(local_reward, group, device)
        trace.reward = reward_total
        trace.reward_components = local_reward.components
        reward_values = gdpo_total_rewards(
            [item.components for item in reward_group], reward_group[0].weights
        )
        group_std = float(reward_values.std(unbiased=False).item())
        group_count += 1
        zero_variance_groups += int(group_std == 0.0)
        invalid_rollouts += int(not local_reward.format_valid)
        rollout_count += 1

        transitions = build_replay_transitions(
            trace,
            mask_token_id=model.mask_token_id,
            block_size=config.block_size,
            sub_block_size=config.sub_block_size,
        )
        batches = bucket_replay_transitions(
            prompt,
            transitions,
            token_budget=args.replay_token_budget,
            maximum_batch_size=args.maximum_replay_batch_size,
        ) if transitions else []
        prompt_cache = ReplayPromptCache(model, prompt)
        model.train()
        engine.zero_grad()
        action_total = max(trace.action_count, 1)
        telemetry_sums = {
            "ratio_sum": 0.0,
            "clip_sum": 0.0,
            "abs_logprob_sum": 0.0,
            "actions": 0,
            "p99_abs_logprob_difference": 0.0,
        }
        policy_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        if batches:
            for batch_index, transition_batch in enumerate(batches):
                final_batch = batch_index == len(batches) - 1
                # ZeRO-1 owns per-parameter reduction hooks.  ``no_sync`` only
                # suppresses the engine's traditional all-reduce path and is
                # therefore insufficient for multiple independent backward
                # graphs: the final graph can observe a parameter partition as
                # already reduced.  DeepSpeed's explicit accumulation boundary
                # is the supported mechanism here.  Non-final exact replay
                # states accumulate locally; the final state performs the sole
                # reduction before the single optimizer step below.
                engine.set_gradient_accumulation_boundary(final_batch)
                current, old = differentiable_transition_batch_logprobs(
                    model, prompt, prompt_cache, transition_batch, config
                )
                batch_loss, metrics = tracerl_token_loss(
                    current,
                    old,
                    advantage,
                    clip_epsilon=config.clip_epsilon,
                )
                actions = int(current.numel())
                action_weight = actions / action_total
                policy_loss_sum.add_(batch_loss.detach().float() * action_weight)
                engine.backward(batch_loss * action_weight)
                ratio = torch.exp(current.detach().float() - old.detach().float())
                absolute = current.detach().float().sub(old.detach().float()).abs()
                telemetry_sums["ratio_sum"] += float(ratio.sum().item())
                telemetry_sums["clip_sum"] += float(
                    ratio.sub(1.0).abs().gt(config.clip_epsilon).float().sum().item()
                )
                telemetry_sums["abs_logprob_sum"] += float(absolute.sum().item())
                telemetry_sums["actions"] += actions
                telemetry_sums["p99_abs_logprob_difference"] = max(
                    telemetry_sums["p99_abs_logprob_difference"],
                    float(torch.quantile(absolute, 0.99).item()),
                )
        else:
            # An immediate causal-anchor EOS has no diffusion action.  Replay
            # one masked dummy state through the full language graph and weight
            # it by zero so every rank executes the same collective hooks.
            dummy = ReplayTransition(
                step=1,
                state_response_ids=[trace.output_ids[0], model.mask_token_id],
                block_anchor_index=0,
                target_relative_indices=[0],
                target_ids=[trace.output_ids[0]],
                old_logprobs=[0.0],
            )
            current, _ = differentiable_transition_batch_logprobs(
                model, prompt, prompt_cache, [dummy], config
            )
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(current.sum() * 0.0)

        before_names, before_values = parameter_probe(model)
        engine.step()
        # ZeRO-1 partitions/reduces gradients and clears ``parameter.grad`` at
        # the accumulation boundary.  Reading module grads here therefore
        # reports a false zero.  DeepSpeed caches the true global norm while
        # executing its optimizer step; audit that value instead.
        deepspeed_grad_norm = engine.get_global_grad_norm()
        if deepspeed_grad_norm is None:
            raise RuntimeError("DeepSpeed did not publish the ZeRO-1 global gradient norm")
        grad_norm = float(deepspeed_grad_norm)
        local_finite = float(math.isfinite(grad_norm))
        if distributed_mean(local_finite, device) != 1.0:
            raise FloatingPointError("non-finite TraceRL gradient")
        after_names, after_values = parameter_probe(model)
        if before_names != after_names:
            raise RuntimeError("parameter probe ordering changed across optimizer step")
        parameter_delta = max(
            float((after - before).abs().max().item())
            for before, after in zip(before_values, after_values, strict=True)
        )
        action_count = int(telemetry_sums["actions"])
        ratio_mean = telemetry_sums["ratio_sum"] / max(action_count, 1)
        mean_abs_difference = telemetry_sums["abs_logprob_sum"] / max(action_count, 1)
        p99_difference = telemetry_sums["p99_abs_logprob_difference"]
        clip_fraction = telemetry_sums["clip_sum"] / max(action_count, 1)

        if not first_step_checked:
            ratio_error = distributed_max(abs(ratio_mean - 1.0), device)
            mean_abs_max = distributed_max(mean_abs_difference, device)
            p99_abs_max = distributed_max(p99_difference, device)
            clip_fraction_max = distributed_max(clip_fraction, device)
            grad_min_flag = distributed_mean(float(grad_norm > 0.0), device)
            delta_min_flag = distributed_mean(float(parameter_delta > 0.0), device)
            # Generation records old log-probabilities without gradients while
            # replay recomputes them through BF16 activation checkpointing.
            # Historical formal telemetry shows a sparse numerical tail up to
            # 0.186 in p99 although the mean error stays below 0.01 and PPO
            # clipping remains inactive. Gate on distribution-level agreement
            # and on the actual clipped-token rate instead of requiring an
            # arbitrary first sample to have p99 < 0.05.
            p99_limit = min(float(config.clip_epsilon), 0.20)
            if (
                ratio_error > 0.02
                or mean_abs_max >= 0.02
                or clip_fraction_max > 0.05
            ):
                raise RuntimeError(
                    "current=old replay parity failed: "
                    f"ratio_error={ratio_error} mean_abs={mean_abs_max} "
                    f"p99={p99_abs_max}/{p99_limit} clip_fraction={clip_fraction_max}"
                )
            if grad_min_flag <= 0.0 or delta_min_flag <= 0.0:
                raise RuntimeError(
                    f"first optimizer step did not update parameters: grad={grad_min_flag} delta={delta_min_flag}"
                )
            first_step_checked = True
            if rank == 0:
                atomic_json(
                    args.stable_gate,
                    {
                        "status": "PASS",
                        "created_at_utc": utc_now(),
                        "optimizer_step": optimizer_step + 1,
                        "temperature": config.temperature,
                        "ratio_mean": ratio_mean,
                        "mean_abs_logprob_difference": mean_abs_difference,
                        "p99_abs_logprob_difference": p99_difference,
                        "clip_fraction_max": clip_fraction_max,
                        "p99_abs_logprob_limit": p99_limit,
                        "gradient_norm": grad_norm,
                        "parameter_probe": before_names,
                        "parameter_max_delta": parameter_delta,
                        "run_id": os.environ.get("GAM_RUN_ID", ""),
                    },
                )

        metrics = {
            "timestamp_utc": utc_now(),
            "optimizer_step": optimizer_step + 1,
            "total_optimizer_steps": total_steps,
            "temperature": config.temperature,
            "reward": reward_total,
            "policy_loss": float(policy_loss_sum.item()),
            "advantage": advantage,
            "reward_group_std": group_std,
            "format_valid": local_reward.format_valid,
            "output_length": len(trace.visible_output_ids),
            "action_count": trace.action_count,
            "forced_one_ratio": trace.forced_one_ratio,
            "eos": model.im_end_token_id in trace.output_ids,
            "num_denoise_forwards": trace.num_denoise_forwards,
            "replay_microbatches": len(batches),
            "ratio_mean": ratio_mean,
            "clip_fraction": clip_fraction,
            "mean_abs_logprob_difference": mean_abs_difference,
            "p99_abs_logprob_difference": p99_difference,
            "gradient_norm": grad_norm,
            "parameter_max_delta": parameter_delta,
            "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        torch.cuda.reset_peak_memory_stats(device)
        if rank == 0:
            append_jsonl(log_path, metrics)
            print(json.dumps({"tracerl": metrics}, sort_keys=True), flush=True)

        completed_step = optimizer_step + 1
        completed_steps = completed_step
        if args.run_kind == "smoke" or completed_step % config.save_steps == 0:
            last_checkpoint = save_checkpoint(
                engine,
                args.output_dir,
                completed_step,
                {"config": asdict(config), "metrics": metrics, "optimizer_step": completed_step},
            )

        # Exactly one post-64-group decision.  A restart request is consumed by
        # the distributed runner, which relaunches from the untouched SFT1 checkpoint.
        if (
            args.run_kind == "formal"
            and args.temperature_adjustment_enabled
            and args.restart_index == 0
            and group_count * groups_per_step >= 64
        ):
            zero_rate = distributed_mean(zero_variance_groups / max(group_count, 1), device)
            invalid_rate = distributed_mean(invalid_rollouts / max(rollout_count, 1), device)
            target = 0.7 if zero_rate > 0.50 else (0.3 if invalid_rate > 0.30 else config.temperature)
            if rank == 0:
                atomic_json(
                    args.output_dir / "temperature-decision.json",
                    {
                        "status": "RESTART" if target != config.temperature else "KEEP",
                        "temperature_before": config.temperature,
                        "temperature_after": target,
                        "zero_reward_variance_group_rate": zero_rate,
                        "invalid_output_rate": invalid_rate,
                        "prompt_groups": group_count * groups_per_step,
                    },
                )
            dist.barrier()
            if target != config.temperature:
                raise SystemExit(75)
            args.temperature_adjustment_enabled = 0

    if completed_steps and (last_checkpoint is None or last_checkpoint.name != f"checkpoint-{completed_steps}"):
        last_checkpoint = save_checkpoint(
            engine,
            args.output_dir,
            completed_steps,
            {"config": asdict(config), "optimizer_step": completed_steps, "completed_at_utc": utc_now()},
        )
    if completed_steps != total_steps:
        raise RuntimeError(
            "TraceRL stopped before the target optimizer step: "
            f"completed={completed_steps} target={total_steps}"
        )
    if rank == 0:
        atomic_json(
            args.final_gate,
            {
                "status": "PASS",
                "completed_at_utc": utc_now(),
                "optimizer_steps": completed_steps,
                "temperature": config.temperature,
                "final_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                "run_id": os.environ.get("GAM_RUN_ID", ""),
            },
        )
    dist.barrier()


if __name__ == "__main__":
    main()
