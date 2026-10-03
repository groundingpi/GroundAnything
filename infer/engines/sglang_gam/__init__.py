"""SGLang external route for the GroundAnything-VLM/Qwen3 GAM-DLM checkpoint.

The package is deliberately kept outside both the vendored Fast-dVLM fork and
the training implementation.  SGLang discovers model/processor classes by
importing this package through ``SGLANG_EXTERNAL_MODEL_PACKAGE`` and
``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Load the route-local old-sgl-kernel compatibility seam before SGLang's model
# registry imports CUDA-dependent model modules.  The operation is idempotent.
try:
    from .compat import (
        install_dllm_packed_mask_replay_patch,
        install_sgl_kernel_compat,
        install_triton_dllm_graph_patch,
    )

    install_sgl_kernel_compat()
    install_triton_dllm_graph_patch()
    # Do not import FlashInfer's updater while SGLang is still importing its
    # external model registry: that creates a registry -> external package ->
    # FlashInfer -> registry cycle on graph-enabled workers.  The graph patch
    # is installed lazily by GAMSpeculativeBlock immediately before its first
    # forward, after the registry/model runner are fully initialized.
except Exception:
    pass


def _mask_id_from_tokenizer(path: str | None) -> int | None:
    """Read the atomic GAM mask id without constructing a tokenizer."""

    if not path:
        return None
    try:
        data = json.loads((Path(path) / "tokenizer_config.json").read_text())
        for key, value in data.get("added_tokens_decoder", {}).items():
            if isinstance(value, dict) and value.get("content") == "|<MASK>|":
                return int(key)
    except (OSError, ValueError, TypeError):
        return None
    return None


def configure_dllm(cfg, server_args):
    """Called by the bundled engine after imports, in every worker process."""
    raw = os.environ.get("GAM_SGLANG_MASK_ID")
    mask_id = int(raw) if raw is not None else _mask_id_from_tokenizer(
        getattr(server_args, "tokenizer_path", None) or server_args.model_path)
    if mask_id is None:
        raise RuntimeError("GAM SGLang cannot determine the atomic mask token")
    expected = int(os.environ.get("GAM_SGLANG_BLOCK_SIZE", "32"))
    if cfg.block_size != expected:
        raise RuntimeError(f"GAM block size mismatch: {cfg.block_size} != {expected}")
    cfg.mask_id = mask_id
    register_algorithm(cfg.algorithm)
    patch_dllm_request_progress()
    patch_dllm_custom_params_transport()
    patch_dllm_single_target_completion()
    return cfg


def register_algorithm(name: str) -> None:
    """Register after SGLang's model registry has finished importing this package.

    Eager imports during external model discovery form a circular dependency;
    swallowing that exception left speculative absent in spawned workers.
    DllmConfig construction runs after discovery and must expose import errors.
    """
    import importlib
    from sglang.srt.dllm.algorithm import algo_name_to_cls
    modules = {
        "GAMSpeculativeBlock": (".gam_speculative", "GAMSpeculativeBlock"),
        "GAMHierarchyBlock": (".gam_hierarchy", "GAMHierarchyBlock"),
        "GAMReliableBlock": (".gam_hierarchy", "GAMReliableBlock"),
        "GAMMinerUBlock": (".gam_mineru", "GAMMinerUBlock"),
        "GAMDecodeV2Block": ("infer.decode.sglang_algorithm", "GAMDecodeV2Block"),
    }
    if name not in modules:
        return
    module, cls = modules[name]
    algo_name_to_cls[name] = getattr(importlib.import_module(module, __name__), cls)


def patch_dllm_request_progress() -> None:
    """Make dLLM block offsets follow committed output, including spec rejects.

    The vendored scheduler advances ``dllm_block_offset`` by a fixed B32 after
    every call.  That is valid only when HierarchyBlock accepts the entire
    block.  Linear speculative decoding can retain a shorter prefix; advancing
    by 32 then creates a positional hole and permanently skips the rejected
    tokens.  Rebuild each draft from the authoritative prompt + committed
    ``output_ids`` and start it at the actual response length instead.

    This monkeypatch is route-local and runs only in the dedicated GAM SGLang
    process.  Full-block hierarchy requests remain byte-for-byte equivalent:
    their committed output length advances by exactly 32.
    """

    try:
        from sglang.srt.managers import schedule_batch
    except Exception:
        return
    req_cls = schedule_batch.Req
    if getattr(req_cls, "_gam_dllm_progress_patch", False):
        return

    def _init_fill_ids_for_dllm(self):
        block_size = int(self.dllm_config.block_size)
        mask_id = int(self.dllm_config.mask_id)
        if not self.dllm_ids:
            self.dllm_ids = list(self.origin_input_ids)
            self.dllm_block_offset = 0
            self._dllm_prompt_prefilled = False
        elif not getattr(self, "_dllm_prompt_prefilled", True):
            # ``process_batch_result_dllm_prefill`` appends the sampled causal
            # anchor to output_ids.  It is a real response token and occupies
            # position zero of the first B32 block.
            committed = list(self.origin_input_ids) + list(self.output_ids)
            if self.output_ids:
                padding = block_size - 1
            else:
                # Defensive fallback when a backend returned no prefill token.
                padding = block_size
            self.dllm_ids = committed + [mask_id] * padding
            self.dllm_block_offset = 0
            self._dllm_prompt_prefilled = True
        else:
            committed = list(self.origin_input_ids) + list(self.output_ids)
            self.dllm_block_offset = len(self.output_ids)
            self.dllm_ids = committed + [mask_id] * block_size
        self.fill_ids = self.dllm_ids

    req_cls._init_fill_ids_for_dllm = _init_fill_ids_for_dllm
    req_cls._gam_dllm_progress_patch = True


def patch_dllm_custom_params_transport() -> None:
    """Retain request custom_params even without a custom logit processor.

    Vendored SGLang populates ``SamplingBatchInfo.custom_params`` only when a
    serialized custom logit processor is enabled.  GAMDecodeV2Block consumes
    ordinary request metadata instead, so that condition silently drops every
    decoder override.  The dataclass already supports filtering/merging this
    list; populate it unconditionally in this dedicated GAM server process.
    """

    try:
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    except Exception:
        return
    if getattr(SamplingBatchInfo, "_gam_dllm_custom_params_patch", False):
        return
    original = SamplingBatchInfo.from_schedule_batch.__func__

    @classmethod
    def wrapped(cls, batch, vocab_size):
        result = original(cls, batch, vocab_size)
        result.custom_params = [
            request.sampling_params.custom_params for request in batch.reqs
        ]
        return result

    SamplingBatchInfo.from_schedule_batch = wrapped
    SamplingBatchInfo._gam_dllm_custom_params_patch = True


def patch_dllm_single_target_completion() -> None:
    """Terminate Referring after its first complete coordinate tuple.

    A B32 diffusion step can predict several comma-separated boxes before the
    first ``<|box_end|>``. Static string stop therefore prevents cross-block
    loops but cannot enforce Referring's one-target protocol inside a block.
    This request-scoped patch observes tokens as the scheduler commits them,
    keeps the model's first 4-coordinate bbox (or 2-coordinate point), appends
    the canonical closing token, and finishes before later tokens in that B32
    result are exposed. Dense/Grounding requests carry no contract and are
    byte-for-byte unchanged.
    """

    try:
        from sglang.srt.managers import schedule_batch
    except Exception:
        return
    req_cls = schedule_batch.Req
    if getattr(req_cls, "_gam_dllm_single_target_patch", False):
        return
    original = req_cls.check_finished

    def check_finished(self, new_accepted_len=1):
        params = getattr(self.sampling_params, "custom_params", None)
        contract = (
            params.get("gam_dlm_single_target")
            if isinstance(params, dict)
            else None
        )
        if isinstance(contract, dict):
            try:
                box_start = int(contract["box_start_token_id"])
                box_end = int(contract["box_end_token_id"])
                coord_min = int(contract["coord_token_min_id"])
                coord_max = int(contract["coord_token_max_id"])
                coordinate_limit = int(contract["coordinate_limit"])
                expected_cardinality = int(contract.get("expected_cardinality", 1))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "invalid gam_dlm_single_target sampling contract"
                ) from exc
            if not (
                0 <= box_start < box_end < coord_min <= coord_max
                and coordinate_limit in (2, 4)
                and expected_cardinality >= 1
            ):
                raise RuntimeError(
                    "unsafe gam_dlm_single_target token ids or coordinate limit"
                )

            from infer.decode.structure import cardinality_cutoff

            output = self.output_ids
            cutoff = cardinality_cutoff(
                output,
                {
                    "box_start_token_id": box_start,
                    "box_end_token_id": box_end,
                    "coord_token_min_id": coord_min,
                    "coord_token_max_id": coord_max,
                    "coordinate_limit": coordinate_limit,
                    "expected_cardinality": expected_cardinality,
                },
            )
            if cutoff is not None:
                del output[cutoff:]
                output.append(box_end)
                self.finished_reason = schedule_batch.FINISH_MATCHED_STR(
                    matched="<|box_end|>"
                )
                self.finished_len = len(output)
                return
        return original(self, new_accepted_len)

    req_cls.check_finished = check_finished
    req_cls._gam_dllm_single_target_patch = True


# This import is intentionally cheap and makes the patch active before the
# scheduler creates its DllmConfig.  Model/processor modules are discovered
# lazily by SGLang's registry.
patch_dllm_request_progress()
patch_dllm_custom_params_transport()
patch_dllm_single_target_completion()

if os.environ.get("GAM_SGLANG_TOKEN_AUDIT_PATH"):
    from .token_audit import install_token_audit
    install_token_audit()
