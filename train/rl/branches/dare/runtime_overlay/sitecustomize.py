"""Opt-in runtime seam for GAM's special-token-preserving reward manager."""

from __future__ import annotations

import os


GAM_RLV3_OVERLAY_ACTIVE = False


if os.environ.get("GAM_RLV3_ENABLE_OVERLAY") == "1":
    import json
    import sys
    import types

    import torch

    # Install raising placeholders for optional SGLang-0.5.6 imports that
    # are absent from the Torch-2.8-compatible kernel wheel.  They are never
    # used by the BF16 dense Qwen3 route; an accidental call fails closed.
    from train.rl.shared.sgl_kernel_compat import install_sgl_kernel_compat

    _GAM_RLV3_SGL_KERNEL_AUDIT = install_sgl_kernel_compat()

    # DARE assumes Ray always narrows every GPU actor to one visible device.
    # On the H800 distributed control plane, Ray can instead leave all eight devices
    # visible while only recording the assigned accelerator in its runtime
    # context.  DARE then leaves LOCAL_RANK at zero and every rank enters NCCL
    # on physical GPU0.  Normalize the binding before ActorRolloutRefWorker
    # initializes its process group.  This is RLV3-only and does not mutate
    # DARE, V1/V2, or the container environment outside each Ray actor.
    from verl.single_controller.base.worker import Worker as _DAREWorker

    _original_worker_init = _DAREWorker.__init__

    def _rlv3_worker_init(self, cuda_visible_devices=None):
        import ray

        accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        logical_device = None
        visible_before = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if os.environ.get("WG_BACKEND") == "ray":
            if len(accelerator_ids) != 1:
                raise RuntimeError(
                    f"RLV3 Ray worker must own exactly one GPU, got {accelerator_ids!r}"
                )
            assigned = str(accelerator_ids[0])
            visible = [item.strip() for item in visible_before.split(",") if item.strip()]

            # CUDA is already initialized by the time this overlay is loaded.
            # Mutating CUDA_VISIBLE_DEVICES here therefore does *not* remap the
            # process to logical device zero.  On distributed/H800 the observed state
            # is, for example, assigned=['3'], CUDA_VISIBLE_DEVICES='3', while
            # torch still exposes all eight devices.  In that case device 3 is
            # the only correct NCCL binding.  A process that was genuinely
            # narrowed before CUDA initialization instead sees one device and
            # must use logical device zero.
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                raise RuntimeError("RLV3 Ray worker has no CUDA device")
            if device_count == 1:
                logical_device = 0
            else:
                try:
                    assigned_index = int(assigned)
                except ValueError:
                    assigned_index = -1
                if 0 <= assigned_index < device_count:
                    logical_device = assigned_index
                elif len(visible) == device_count and assigned in visible:
                    logical_device = visible.index(assigned)
                else:
                    raise RuntimeError(
                        "RLV3 cannot map Ray GPU assignment to a CUDA logical "
                        f"device: assigned={assigned!r}, visible={visible!r}, "
                        f"device_count={device_count}"
                    )

            # Preserve Ray's CUDA_VISIBLE_DEVICES verbatim.  Only LOCAL_RANK
            # and the CUDA current device are normalized before NCCL init.
            os.environ["LOCAL_RANK"] = str(logical_device)
            torch.cuda.set_device(logical_device)

        _original_worker_init(self, cuda_visible_devices=cuda_visible_devices)

        if os.environ.get("WG_BACKEND") == "ray":
            torch.cuda.set_device(logical_device)
            audit = {
                "status": "PASS",
                "rank": int(os.environ["RANK"]),
                "ray_local_rank": int(os.environ["RAY_LOCAL_RANK"]),
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "ray_accelerator_ids": accelerator_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "ray_node_id": ray.get_runtime_context().get_node_id(),
            }
            if audit["cuda_current_device"] != logical_device:
                raise RuntimeError(f"RLV3 GPU binding audit failed: {audit}")
            print(json.dumps({"gam_rlv3_gpu_binding": audit}, sort_keys=True), flush=True)
            audit_root = os.environ.get("GAM_RLV3_AUDIT_DIR")
            if audit_root:
                from pathlib import Path

                target = Path(audit_root) / f"gpu-binding-rank-{audit['rank']:02d}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                temporary.write_text(
                    json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary.replace(target)

    _DAREWorker.__init__ = _rlv3_worker_init
    _DAREWorker._gam_rlv3_gpu_binding_patch = True

    # DARE's generic reward registry eagerly imports its math/code evaluator
    # stack (pylatexenc, mmengine, tempdir, ...).  RLV3 never calls that stack:
    # every batch is handled by GAMBatchRewardManager.  Inject a temporary,
    # fail-closed placeholder only while importing DARE's reward dispatcher so
    # the isolated GAM path does not inherit dozens of unrelated dependencies.
    # The real module is restored immediately; the dispatcher keeps the
    # fail-closed function only for its unused generic fallback.
    reward_score_name = "verl.utils.reward_score"
    missing = object()
    previous_reward_score = sys.modules.get(reward_score_name, missing)
    reward_score_stub = types.ModuleType(reward_score_name)

    def _unused_default_compute_score(*args, **kwargs):
        raise RuntimeError("generic DARE reward is disabled in GAM RLV3")

    reward_score_stub.default_compute_score = _unused_default_compute_score
    sys.modules[reward_score_name] = reward_score_stub
    try:
        import verl.trainer.ppo.reward as reward_module
    finally:
        if previous_reward_score is missing:
            sys.modules.pop(reward_score_name, None)
        else:
            sys.modules[reward_score_name] = previous_reward_score

    import verl.trainer.ppo.ray_trainer as trainer_module

    original = reward_module.load_reward_manager

    def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
        if config.reward_model.get("reward_manager") == "gam_batch":
            from train.rl.branches.dare.gam_batch_reward_manager import GAMBatchRewardManager

            return GAMBatchRewardManager(
                tokenizer=tokenizer,
                num_examine=num_examine,
                **reward_kwargs,
            )
        return original(config, tokenizer, num_examine, **reward_kwargs)

    reward_module.load_reward_manager = load_reward_manager
    reward_module._gam_rlv3_reward_patch = True

    original_metrics = trainer_module.compute_data_metrics

    def compute_data_metrics(batch, use_critic=True):
        metrics = original_metrics(batch=batch, use_critic=use_critic)
        if {
            "rollout_log_probs",
            "old_log_probs",
            "response_mask",
        }.issubset(batch.batch.keys()):
            mask = batch.batch["response_mask"].bool()
            rollout = batch.batch["rollout_log_probs"]
            old = batch.batch["old_log_probs"]
            difference = torch.masked_select((rollout - old).abs(), mask)
            ratio = torch.masked_select(torch.exp(old - rollout), mask)
            if difference.numel() > 0:
                metrics.update(
                    {
                        "gam_rlv3/mean_abs_rollout_old_logp": difference.mean().item(),
                        "gam_rlv3/p99_abs_rollout_old_logp": torch.quantile(
                            difference.float(), 0.99
                        ).item(),
                        "gam_rlv3/ratio_mean": ratio.float().mean().item(),
                    }
                )
        return metrics

    trainer_module.compute_data_metrics = compute_data_metrics
    trainer_module._gam_rlv3_parity_metrics_patch = True

    # Persist the exact console metric dictionary as JSONL.  DARE's default
    # console formatter rounds every value to three decimals, which is not
    # sufficient for rollout/teacher-forcing parity gates.  This observer runs
    # after metrics have been computed and never mutates the training payload.
    import numbers
    import time
    from pathlib import Path
    from verl.utils.logger.aggregate_logger import LocalLogger

    original_local_log = LocalLogger.log

    def local_log(self, data, step):
        original_local_log(self, data, step)
        target_value = os.environ.get("GAM_RLV3_METRICS_JSONL")
        if not target_value:
            return
        metrics = {
            str(key): float(value)
            for key, value in data.items()
            if isinstance(value, numbers.Number) and not isinstance(value, bool)
        }
        payload = {"step": int(step), "wall_time": time.time(), "metrics": metrics}
        target = Path(target_value)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    LocalLogger.log = local_log
    LocalLogger._gam_rlv3_json_metrics_patch = True
    GAM_RLV3_OVERLAY_ACTIVE = True
