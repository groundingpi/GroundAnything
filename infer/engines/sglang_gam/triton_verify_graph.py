"""Opt-in causal verifier graph, captured before any real prefill.

The draft and verifier own separate attention metadata, graph input buffers,
and graph allocation pools. Only model parameters and the real KV pool are
shared. The normal speculative acceptance policy is unchanged.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def use_backend(runner, backend):
    previous = runner.attn_backend
    runner.attn_backend = backend
    try:
        yield
    finally:
        runner.attn_backend = previous


def install_triton_verify_graph_patch():
    """Called by the loaded GAM model constructor, after registry imports."""
    if os.environ.get("GAM_SGLANG_TRITON_VERIFY_GRAPH", "0") != "1":
        return False
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor import cuda_graph_runner as graph_module
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    if getattr(ModelRunner, "_gam_verify_graph_patch", False):
        return True
    original = ModelRunner.init_device_graphs

    def init_device_graphs(self):
        if os.environ.get("GAM_SGLANG_TRITON_VERIFY_GRAPH", "0") != "1":
            return original(self)
        args = self.server_args
        if not (
            type(self.attn_backend) is TritonAttnBackend
            and os.environ.get("GAM_SGLANG_ALGORITHM") == "speculative"
            and os.environ.get("GAM_SGLANG_TRITON_GRAPH_METADATA_V2") == "1"
            and os.environ.get("GAM_SGLANG_ALLOW_TRITON_DLM_GRAPH") == "1"
            and not args.disable_cuda_graph
            and not args.enable_deterministic_inference
            and args.tp_size == args.pp_size == args.dp_size == 1
            and not args.enable_two_batch_overlap
            and not args.enable_pdmux
            and not args.enable_lora
            and not args.enable_return_hidden_states
            and int(os.environ.get("GAM_SGLANG_BLOCK_SIZE", "32")) == 32
            and self.hybrid_gdn_config is None
            and self.kimi_linear_config is None
        ):
            raise RuntimeError("Triton verifier graph requires isolated B32 normal-mode Strict Spec")
        original(self)
        if self.graph_runner is None or self.graph_runner.capture_bs != [1]:
            raise RuntimeError("Triton verifier graph requires a captured single-request draft graph")
        backend = TritonAttnBackend(self)
        backend._gam_triton_causal_graph = True
        # A second init_cuda_graph_state on the draft backend would replace
        # the indices allocation still referenced by its captured graph.
        previous_pool = graph_module.get_global_graph_memory_pool()
        try:
            graph_module.set_global_graph_memory_pool(None)
            with use_backend(self, backend):
                graph = graph_module.CudaGraphRunner(self)
        finally:
            graph_module.set_global_graph_memory_pool(previous_pool)
            graph_module.set_graph_pool_id(previous_pool)
        self._gam_verify_backend = backend
        self._gam_verify_graph = graph
        self._gam_verify_counts = {"replays": 0, "eager_shape_fallbacks": 0, "kv_audits": 0}
        atexit.register(lambda: logger.info("GAM Triton verifier graph counters %s", self._gam_verify_counts))
        logger.info("GAM Triton verifier graph ready: independent metadata/buffers/pool; causal=True; save_kv=True; before first prefill")

    ModelRunner.init_device_graphs = init_device_graphs
    ModelRunner._gam_verify_graph_patch = True
    return True


def _snapshot_kv(runner, locations):
    pool = runner.token_to_kv_pool
    return [
        (pool.get_key_buffer(layer.self_attn.attn.layer_id).index_select(0, locations),
         pool.get_value_buffer(layer.self_attn.attn.layer_id).index_select(0, locations))
        for layer in runner.model.model.layers
    ]


def _kv_equal(left, right):
    import torch
    return all(torch.equal(a, b) and torch.equal(c, d)
               for (a, c), (b, d) in zip(left, right))


def replay_verifier(runner, batch):
    """Return native LogitsProcessorOutput and whether a graph replay ran."""
    graph = getattr(runner, "_gam_verify_graph", None)
    if graph is None:
        raise RuntimeError("verifier graph was not initialized before real requests")
    # CudaGraphRunner.can_run does not check dLLM input token count. Initial
    # prompt/chunk batches must stay eager even when their batch size is one.
    supported = (batch.forward_mode.is_dllm_extend()
                 and batch.batch_size == 1 and batch.input_ids.numel() == 32
                 and graph.can_run(batch))
    if not supported:
        runner._gam_verify_counts["eager_shape_fallbacks"] += 1
        out = runner.forward_extend(batch, pp_proxy_tensors=None)
        return (out[0] if isinstance(out, tuple) else out), False

    audit_path = os.environ.get("GAM_SGLANG_VERIFY_KV_AUDIT_PATH")
    audit = bool(audit_path and runner._gam_verify_counts["kv_audits"] < int(
        os.environ.get("GAM_SGLANG_VERIFY_KV_AUDIT_BLOCKS", "16")))
    if audit:
        # Snapshot every prefix slot, not a sample. Current B32 must match the
        # eager causal write and prefix KV must remain bitwise unchanged.
        prefix_len = int(batch.seq_lens[0].item()) - 32
        request_slot = int(batch.req_pool_indices[0].item())
        prefix_locations = runner.req_to_token_pool.req_to_token[request_slot, :prefix_len].long()
        prefix_before = _snapshot_kv(runner, prefix_locations)
    with use_backend(runner, runner._gam_verify_backend):
        out = graph.replay(batch)
    runner._gam_verify_counts["replays"] += 1
    if runner._gam_verify_counts["replays"] == 1:
        logger.info("GAM Triton verifier graph first replay: B32 causal KV commit")
    if audit:
        import torch
        prefix_after = _snapshot_kv(runner, prefix_locations)
        current_locations = batch.out_cache_loc.long()
        graph_kv = _snapshot_kv(runner, current_locations)
        graph_logits = out.full_logits.clone()
        eager = runner.forward_extend(batch, pp_proxy_tensors=None)
        if isinstance(eager, tuple):
            eager = eager[0]
        eager_kv = _snapshot_kv(runner, current_locations)
        record = dict(
            block=runner._gam_verify_counts["kv_audits"], prefix_tokens=prefix_len,
            layers=len(graph_kv), prefix_unchanged=_kv_equal(prefix_before, prefix_after),
            committed_kv_equal=_kv_equal(graph_kv, eager_kv),
            full_logits_equal=torch.equal(graph_logits, eager.full_logits),
            argmax_equal=torch.equal(graph_logits.argmax(-1), eager.full_logits.argmax(-1)),
            maximum_logit_abs_error=float((graph_logits - eager.full_logits).abs().max().item()),
        )
        record["status"] = "PASS" if all(record[k] for k in (
            "prefix_unchanged", "committed_kv_equal", "full_logits_equal", "argmax_equal")) else "FAIL"
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        runner._gam_verify_counts["kv_audits"] += 1
        if record["status"] != "PASS":
            raise RuntimeError(f"Triton verifier KV/logit shadow gate failed: {record}")
    return out, True
