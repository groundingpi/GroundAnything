#!/usr/bin/env python3
"""Audited entry point for DARE-backed standard causal GAM GRPO."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")


import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    import ray

    config_path = args.config.resolve(strict=True)
    config = OmegaConf.load(config_path)
    model_path = Path(str(config.actor_rollout_ref.model.path)).resolve(strict=True)
    train_path = Path(str(config.data.train_files[0])).resolve(strict=True)
    model_manifest = model_path / "rlv3_causal_model_manifest.json"
    data_manifest = train_path.parent / "build.json"
    data_audit = train_path.parent / "audit.json"
    for required in (model_manifest, data_manifest, data_audit):
        required.resolve(strict=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config.trainer.default_local_dir = str(output)
    config.trainer.experiment_name = f"dare_causal_grpo_h80056_{args.run_kind}"
    if args.run_kind == "smoke":
        config.trainer.total_training_steps = 1
        config.trainer.save_freq = 1
        config.trainer.max_actor_ckpt_to_keep = 1
        config.trainer.rollout_data_dir = str(output / "rollouts")
    else:
        config.trainer.total_training_steps = None
        config.trainer.resume_mode = "auto"

    # Fail closed on every mathematical and topology-sensitive setting.
    expected = {
        "algorithm": "grpo",
        "batch": 56,
        "rollouts": 8,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "policy": "sglang",
        "loss_agg": "seq-mean-token-mean",
        "lr": 2.0e-6,
        "nodes": 7,
        "gpus_per_node": 8,
    }
    actual = {
        "algorithm": str(config.algorithm.adv_estimator),
        "batch": int(config.data.train_batch_size),
        "rollouts": int(config.actor_rollout_ref.rollout.n),
        "temperature": float(config.actor_rollout_ref.rollout.temperature),
        "top_p": float(config.actor_rollout_ref.rollout.top_p),
        "top_k": int(config.actor_rollout_ref.rollout.top_k),
        "policy": str(config.actor_rollout_ref.rollout.name),
        "loss_agg": str(config.actor_rollout_ref.actor.loss_agg_mode),
        "lr": float(config.actor_rollout_ref.actor.optim.lr),
        "nodes": int(config.trainer.nnodes),
        "gpus_per_node": int(config.trainer.n_gpus_per_node),
    }
    if actual != expected:
        raise RuntimeError(f"RLV3 contract drift: expected={expected}, actual={actual}")
    if actual["batch"] * actual["rollouts"] % (actual["nodes"] * actual["gpus_per_node"]):
        raise RuntimeError("RLV3 rollout batch is not divisible by 56 GPUs")
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        raise RuntimeError("RLV3 must not construct a reference/KL path")
    if config.actor_rollout_ref.actor.fsdp_config.param_offload:
        raise RuntimeError("RLV3 parameter offload is forbidden")

    contract = {
        "status": "READY",
        "run_kind": args.run_kind,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "model": str(model_path),
        "model_manifest_sha256": sha256(model_manifest),
        "data": str(train_path),
        "data_manifest_sha256": sha256(data_manifest),
        "data_audit_sha256": sha256(data_audit),
        "unique_rows": 17700,
        "effective_rows": 17752,
        "deterministic_tail_repeats": 52,
        "optimizer_steps": 1 if args.run_kind == "smoke" else 317,
        "resume_mode": str(config.trainer.resume_mode),
        "initial_checkpoint": "posttrain-0826/chain1/formal/sft2/checkpoint-147",
        "policy": "exact causal rollout + exact causal teacher forcing",
        "decode_v4_used_for_training": False,
        "contract": expected,
    }
    (output / "rlv3-run-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "resolved-config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )

    runtime_env = {
        key: value
        for key, value in os.environ.items()
        if key == "PYTHONPATH"
        or key.startswith(("GAM_", "SGLANG_", "NCCL_", "TORCH_"))
        or key in {
            "HF_MODULES_CACHE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "PYTORCH_CUDA_ALLOC_CONF",
        }
    }
    address = os.environ.get("RAY_ADDRESS")
    if not address:
        raise RuntimeError("RAY_ADDRESS is required; refusing to create a local one-node cluster")
    ray.init(address=address, runtime_env={"env_vars": runtime_env})

    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
