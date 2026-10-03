"""GAM-safe linear self-speculative decoding for the Fast-dLLM SGLang fork.

The two passes share SGLang's DLLM_EXTEND CUDA graph when FlashInfer is
available: a bidirectional custom mask drafts one B32 block, then a lower
triangular custom mask verifies it.  Eager/Triton remains a compatible
fallback.  No checkpoint or training tensor is changed.
"""

from __future__ import annotations

import json
import os
import atexit
from typing import Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from .compat import (
    fast_eager_context,
    install_dllm_packed_mask_replay_patch,
)
from infer.decode.decode_ops import shifted_argmax
from .execution_ops import longest_prefix_and_correction


class GAMSpeculativeBlock(DllmAlgorithm):
    """One bidirectional draft plus causal longest-prefix verification."""

    def __init__(self, config):
        super().__init__(config)
        self.token_shift = int(config.algorithm_config.get("token_shift", 1))
        self.last_inherited_token: Optional[int] = None
        self._request = 0
        self._blocks = 0
        self._accepted = 0
        # Trace is a calibration aid, not part of the decode contract.  The
        # old implementation opened/fsynced a JSONL file for every B32 block;
        # long responses therefore paid an avoidable filesystem round-trip on
        # every verifier.  Keep one buffered stream per algorithm worker and
        # flush periodically so a killed job still retains recent rows.
        self._trace_stream = None
        self._trace_path: str | None = None
        self._trace_rows_since_flush = 0
        try:
            self._trace_flush_every = max(
                1, int(os.environ.get("GAM_SGLANG_SPEC_TRACE_FLUSH_EVERY", "16"))
            )
        except ValueError:
            self._trace_flush_every = 16
        atexit.register(self._close_trace)
        B = self.block_size
        self._causal_mask = torch.tril(torch.ones(B, B, dtype=torch.uint8)).flatten()
        self._bidir_mask = torch.ones(B * B, dtype=torch.uint8)
        # FlashInfer's graph wrapper consumes bit-packed masks.  Keep packed
        # CPU templates as well as the legacy raw templates; the compatibility
        # layer selects the packed view without changing the eager fallback.
        self._causal_mask_packed = self._pack_mask(self._causal_mask)
        self._bidir_mask_packed = self._pack_mask(self._bidir_mask)
        # The CUDA-graph path must mutate a device-resident buffer that lives
        # on the same stream as graph replay.  Converting the CPU template in
        # every call (``src.to(device)``) can enqueue a separate H2D copy and
        # let replay observe the previous mask.  Keep lazy per-device copies;
        # this is only 2*B*B bytes and does not alter model state.
        self._mask_device_cache = {}
        if self.block_size != 32:
            raise ValueError(f"GAM DLM requires block_size=32, got {self.block_size}")
        if self.token_shift not in (0, 1):
            raise ValueError("token_shift must be 0 or 1")

    @staticmethod
    def _pack_mask(mask: torch.Tensor) -> torch.Tensor:
        """Pack uint8 0/1 values in FlashInfer's little-endian bit order."""

        if mask.numel() % 8:
            mask = torch.nn.functional.pad(mask, (0, 8 - mask.numel() % 8))
        bits = (1 << torch.arange(8, dtype=torch.int64)).view(1, 8)
        return (mask.reshape(-1, 8).to(torch.int64) * bits).sum(dim=1).to(torch.uint8)

    @staticmethod
    def _set_attention_type(model_runner: ModelRunner, attn_type: AttentionType) -> None:
        model = model_runner.model
        language = getattr(model, "model", model)
        layers = getattr(language, "layers", None)
        if layers is None:
            raise RuntimeError("GAM SGLang model exposes no language layers")
        # AttentionType is a Python attribute read by the backend at forward
        # time.  Avoid walking all transformer layers when the two passes use
        # the same type (the common draft-only-Graph case).
        if getattr(model_runner, "_gam_attention_type", None) == attn_type:
            return
        for layer in layers:
            attention = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attention is not None:
                attention.attn_type = attn_type
        model_runner._gam_attention_type = attn_type

    def _write_mask(self, model_runner: ModelRunner, *, causal: bool) -> None:
        """Update FlashInfer's preallocated dLLM mask without replacing storage."""

        backend = model_runner.attn_backend
        # In graph mode the vendored backend passes ``custom_mask_buf`` to
        # FlashInfer.  FlashInfer packs into that allocation during
        # ``plan``; it is therefore already the packed destination even
        # though the backend exposes it under the historical raw-mask name.
        # Use the wrapper-owned buffer directly instead of an import-time
        # monkey-patch (which is unsafe across SGLang's worker spawn).
        # ``_custom_mask_buf`` is a packed destination only for the explicit
        # CUDA-Graph compatibility route.  The same vendored FlashInfer
        # wrapper can be constructed in eager mode, where ``call_begin_forward``
        # still expects a raw mask and performs its own pack.  Do not write a
        # packed template into that eager buffer unless the replay adapter is
        # enabled; otherwise the eager FlashInfer diagnostic path would suffer
        # the exact double-pack bug this graph fix is meant to isolate.
        # Fast-dLLM's reference implementation writes a raw BxB mask and
        # lets FlashInfer pack it in ``begin_forward``.  The packed route is
        # kept as an opt-in optimization, but both representations must be
        # explicit: confusing a packed buffer with a raw mask silently gives
        # a very fast yet semantically invalid CUDA-Graph result.
        mask_mode = os.environ.get(
            "GAM_SGLANG_FLASHINFER_MASK_MODE",
            "packed"
            if os.environ.get("GAM_SGLANG_FLASHINFER_PACKED_MASK", "0") == "1"
            else "raw",
        ).lower()
        if mask_mode not in {"raw", "packed"}:
            raise RuntimeError(f"invalid GAM_SGLANG_FLASHINFER_MASK_MODE={mask_mode!r}")
        graph_packed_mode = mask_mode == "packed"
        wrapper = getattr(backend, "dllm_spec_wrapper_ragged", None)
        packed_buf = (
            getattr(backend, "dllm_ragged_packed_custom_mask", None)
            if graph_packed_mode
            else None
        )
        if packed_buf is None and graph_packed_mode and wrapper is not None:
            packed_buf = getattr(wrapper, "_custom_mask_buf", None)
        buf = getattr(backend, "dllm_ragged_custom_mask", None)
        if packed_buf is not None:
            src = self._causal_mask_packed if causal else self._bidir_mask_packed
            if src.numel() > packed_buf.numel():
                raise RuntimeError(
                    f"GAM speculative packed mask buffer too small: "
                    f"{packed_buf.numel()} < {src.numel()}"
                )
            key = (packed_buf.device.type, packed_buf.device.index)
            device_src = self._mask_device_cache.get((key, causal, "packed"))
            if device_src is None or device_src.device != packed_buf.device:
                device_src = src.to(device=packed_buf.device, non_blocking=False)
                self._mask_device_cache[(key, causal, "packed")] = device_src
            packed_buf[: src.numel()].copy_(device_src, non_blocking=False)
            return
        if buf is None:
            return
        src = self._causal_mask if causal else self._bidir_mask
        if src.numel() > buf.numel():
            raise RuntimeError(
                f"GAM speculative mask buffer too small: {buf.numel()} < {src.numel()}"
            )
        key = (buf.device.type, buf.device.index)
        device_src = self._mask_device_cache.get((key, causal))
        if device_src is None or device_src.device != buf.device:
            device_src = src.to(device=buf.device, non_blocking=False)
            self._mask_device_cache[(key, causal)] = device_src
        # Both tensors are on the current CUDA stream.  Keeping this copy
        # device-to-device avoids an implicit host staging stream and makes
        # the mask update ordered before the immediately-following replay.
        buf[: src.numel()].copy_(device_src, non_blocking=False)

    @staticmethod
    def _diagnostic_graph_sync() -> None:
        """Optional race diagnostic; never enabled by production defaults."""

        if os.environ.get("GAM_SGLANG_SPEC_GRAPH_SYNC", "0") == "1":
            # If an explicit synchronization repairs graph parity, the next
            # optimization should replace this with a stream/event dependency
            # rather than carrying a device-wide barrier into serving.
            torch.cuda.current_stream().synchronize()

    @staticmethod
    def _align_dllm_positions_before_graph(
        model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> None:
        """Materialize GAM's absolute Qwen3 positions before graph replay.

        ``FastDVLMForConditionalGeneration.forward`` normally repairs the
        scheduler's local ``0..B-1`` dLLM positions at the model boundary.
        CUDA Graph replay does not execute that Python boundary: its input
        buffers are populated first and only the captured CUDA operations are
        replayed.  The first draft therefore used local positions while the
        cached multimodal prefix used absolute positions, which collapsed the
        graph output to a constant token.  Move the same, idempotent position
        normalization ahead of ``ModelRunner.forward`` so eager and graph
        consume identical RoPE positions.  This changes no checkpoint,
        attention policy, sampling rule, or training path.
        """

        model = model_runner.model
        align = getattr(model, "_align_extend_positions", None)
        positions = getattr(forward_batch, "positions", None)
        if align is None or not isinstance(positions, torch.Tensor):
            return
        aligned = align(positions, forward_batch, forward_batch.input_ids)
        forward_batch.positions = aligned.to(
            device=positions.device,
            dtype=positions.dtype,
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        # Install only after SGLang's registry/model-runner imports have
        # completed.  Importing FlashInfer from the external package's
        # __init__ can deadlock graph-enabled workers during registry import;
        # this is the first safe point and still precedes graph capture/replay.
        if os.environ.get("GAM_SGLANG_DISABLE_CUDA_GRAPH", "1") != "1":
            mask_mode = os.environ.get(
                "GAM_SGLANG_FLASHINFER_MASK_MODE", "packed"
            ).lower()
            if mask_mode == "packed" and not install_dllm_packed_mask_replay_patch():
                raise RuntimeError(
                    "GAM packed-mask graph route requires FlashInfer replay "
                    "patch before the first DLM forward"
                )
        ids = forward_batch.input_ids
        total_len = int(ids.numel())
        initial_mask = ids.eq(self.mask_id)
        num_masked = int(initial_mask.sum().item())
        if num_masked == 0:
            raise RuntimeError("GAM speculative block contains no mask tokens")
        output_start = total_len - num_masked

        positions = getattr(forward_batch, "positions", None)
        flat_positions = (
            positions.reshape(-1) if isinstance(positions, torch.Tensor) else None
        )
        is_new_request = bool(
            flat_positions is not None
            and flat_positions.numel() > 0
            and int(flat_positions[0].item()) == 0
        )
        if is_new_request:
            self._flush_trace()
            self.last_inherited_token = None
            self._request += 1
            self._blocks = 0
            self._accepted = 0

        # After a partial or complete previous block, the causal verifier's
        # correction token becomes the anchor of the next draft.
        if bool(ids[0].eq(self.mask_id)) and self.last_inherited_token is not None:
            ids[0] = int(self.last_inherited_token)

        # ``initial_mask`` is intentionally retained above for output
        # accounting: on every block after the first, position zero starts as
        # a mask but is the causal correction token carried from the previous
        # verifier and therefore is a real output token.  It must *not* be
        # drafted again.  Recompute the writable mask after inheritance so the
        # correction anchor survives the encoder-only pass.  Reusing the
        # pre-inheritance mask here overwrote that token, shifted every later
        # block by one, and produced repetitive coordinate streams even though
        # the first block agreed with the exact Transformers verifier.
        draft_mask = ids.eq(self.mask_id)

        # This must happen outside the captured model body.  Eager execution
        # would repair positions inside the GAM model wrapper, but CUDA Graph
        # replay only copies ``forward_batch.positions`` into its persistent
        # input buffer and never re-enters that Python wrapper.
        self._align_dllm_positions_before_graph(model_runner, forward_batch)

        # Phase 1: full B32 bidirectional draft. ``forward`` is DLLM_EXTEND and
        # replays the captured graph when graph mode is enabled.
        self._set_attention_type(model_runner, AttentionType.ENCODER_ONLY)
        self._write_mask(model_runner, causal=False)
        # Keep the optimization at the forward boundary.  Unlike a class-level
        # ModelRunner patch this cannot affect worker initialization or other
        # routes, and the context restores CUDA synchronization immediately
        # after the enqueue.
        # Diagnostic reverse isolation: keep draft eager while allowing the
        # causal verifier to replay the graph. Production defaults remain the
        # original two-pass route until both sides pass semantic parity.
        draft_eager_only = (
            os.environ.get("GAM_SGLANG_SPEC_DRAFT_EAGER_VERIFY_GRAPH", "0") == "1"
        )
        if draft_eager_only:
            draft_out = model_runner.forward_extend(
                forward_batch, pp_proxy_tensors=None
            )
            if isinstance(draft_out, tuple):
                draft_out = draft_out[0]
            draft_logits_output = draft_out
            can_run_cuda_graph = False
        else:
            with fast_eager_context():
                draft_out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._diagnostic_graph_sync()
            draft_logits_output = draft_out.logits_output
            can_run_cuda_graph = bool(getattr(draft_out, "can_run_graph", False))
        draft_logits = getattr(draft_logits_output, "full_logits", None)
        if draft_logits is None:
            raise RuntimeError("GAM speculative draft requires full logits")
        draft_predictions = shifted_argmax(draft_logits, self.token_shift)
        ids[draft_mask] = draft_predictions[draft_mask]
        trace_tokens = bool(
            os.environ.get("GAM_SGLANG_SPEC_TRACE_PATH")
            and os.environ.get("GAM_SGLANG_SPEC_TRACE_TOKENS") == "1"
        )
        drafted_ids = ids.detach().cpu().tolist() if trace_tokens else None

        # Explicit upper-bound diagnostic: one eager bidirectional block
        # forward, without the causal verifier.  This is intentionally opt-in
        # and fail-closed in the launcher defaults.  It estimates the maximum
        # throughput available if a future trained verifier/distillation path
        # makes the draft self-consistent; it is not claimed as strict
        # speculative decoding because no AR parity check is performed here.
        draft_only_eager = (
            os.environ.get("GAM_SGLANG_SPEC_DRAFT_ONLY_EAGER", "0") == "1"
        )
        if draft_only_eager:
            output_count = max(total_len - output_start, 0)
            next_token_ids = ids[output_start : output_start + output_count]
            if output_count:
                self.last_inherited_token = int(ids[-1].item())
            else:
                self.last_inherited_token = None
            self._blocks += 1
            self._accepted += int(output_count)
            self._trace(
                total_len=total_len,
                output_start=output_start,
                keep_positions=total_len,
                output_count=output_count,
                nfe_per_block=1,
                drafted_ids=drafted_ids,
                causal_predictions=None,
                output_ids=(
                    next_token_ids.detach().cpu().tolist() if trace_tokens else None
                ),
            )
            self._set_attention_type(model_runner, AttentionType.DECODER)
            return draft_logits_output, next_token_ids, False

        # Phase 2: causal verification through the same DLLM_EXTEND graph when
        # FlashInfer's custom-mask buffer exists.  Triton/eager has no mutable
        # mask storage, so retain its proven decoder forward_extend fallback.
        if getattr(model_runner.attn_backend, "dllm_ragged_custom_mask", None) is not None:
            # A CUDA graph can safely accelerate the bidirectional draft even
            # on FlashInfer builds where mutable custom-mask replay has not
            # yet demonstrated token parity.  The opt-in split mode keeps the
            # causal verifier on the proven eager path; it is deliberately
            # disabled by default until a graph/eager parity gate passes.
            draft_only_graph = os.environ.get(
                "GAM_SGLANG_SPEC_DRAFT_ONLY_GRAPH", "0"
            ) == "1"
            self._set_attention_type(
                model_runner,
                (
                    AttentionType.DECODER
                    if (
                        draft_only_graph
                        or os.environ.get("GAM_SGLANG_SPEC_GRAPH_VERIFY_DECODER", "0")
                        == "1"
                    )
                    else AttentionType.ENCODER_ONLY
                ),
            )
            self._write_mask(model_runner, causal=True)
            self._diagnostic_graph_sync()
            draft_graph_ok = bool(getattr(draft_out, "can_run_graph", False))
            if draft_only_graph:
                verify_out = model_runner.forward_extend(
                    forward_batch, pp_proxy_tensors=None
                )
                if isinstance(verify_out, tuple):
                    verify_out = verify_out[0]
                # ``forward_extend`` returns LogitsProcessorOutput directly,
                # whereas graph replay returns ModelRunnerOutput.  Keep the
                # two API shapes explicit so draft-only Graph mode cannot
                # fail after a successful draft capture.
                verify_logits_output = verify_out
            else:
                with fast_eager_context():
                    verify_out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                verify_logits_output = verify_out.logits_output
            can_run_cuda_graph = draft_graph_ok and (
                draft_only_graph or bool(getattr(verify_out, "can_run_graph", False))
            )
            verify_logits = getattr(verify_logits_output, "full_logits", None)
        else:
            self._set_attention_type(model_runner, AttentionType.DECODER)
            if os.environ.get("GAM_SGLANG_TRITON_VERIFY_GRAPH", "0") == "1":
                from .triton_verify_graph import replay_verifier
                verify_out, can_run_cuda_graph = replay_verifier(model_runner, forward_batch)
            else:
                verify_out = model_runner.forward_extend(forward_batch, pp_proxy_tensors=None)
                can_run_cuda_graph = False
            if isinstance(verify_out, tuple):
                verify_out = verify_out[0]
            verify_logits = getattr(verify_out, "full_logits", None)
            verify_logits_output = verify_out
        if verify_logits is None:
            raise RuntimeError("GAM speculative verifier requires full logits")
        # The causal verifier's rows are already aligned by the scheduler;
        # unlike the bidirectional draft, it intentionally uses the direct
        # row-wise argmax.  Keep this indexing unchanged for token parity.
        causal_predictions = verify_logits.argmax(dim=-1)

        # The causal row at i predicts token i+1.  Retain the longest matching
        # draft prefix, then carry the first correction prediction into the
        # next block.  At least the anchor/correction position advances.
        # Comparing one token at a time with ``.item()`` forces a host sync
        # for every position (up to 31 synchronizations per B32 verify).
        # Keep the exact longest-prefix rule, but perform it on-device and
        # transfer only the first mismatch index.
        if os.environ.get("GAM_SGLANG_SPEC_PREFIX_REDUCTION", "0") == "1":
            keep_positions, self.last_inherited_token = longest_prefix_and_correction(
                ids, causal_predictions
            )
        else:
            equal = causal_predictions[:-1].eq(ids[1:])
            mismatch = (~equal).nonzero(as_tuple=False)
            matched = (
                int(mismatch[0, 0].item())
                if mismatch.numel()
                else total_len - 1
            )
            keep_positions = min(matched + 1, total_len)
            self.last_inherited_token = int(causal_predictions[keep_positions - 1].item())
        output_count = max(keep_positions - output_start, 0)
        next_token_ids = ids[output_start : output_start + output_count]

        self._blocks += 1
        self._accepted += int(output_count)
        self._trace(
            total_len=total_len,
            output_start=output_start,
            keep_positions=keep_positions,
            output_count=output_count,
            nfe_per_block=2,
            drafted_ids=drafted_ids,
            causal_predictions=(
                causal_predictions.detach().cpu().tolist() if trace_tokens else None
            ),
            output_ids=(next_token_ids.detach().cpu().tolist() if trace_tokens else None),
        )
        # Request-boundary invariant: never leak encoder-only mode into the
        # next text/image prefill.
        self._set_attention_type(model_runner, AttentionType.DECODER)
        return verify_logits_output, next_token_ids, can_run_cuda_graph

    def _trace(
        self,
        *,
        total_len: int,
        output_start: int,
        keep_positions: int,
        output_count: int,
        nfe_per_block: int,
        drafted_ids: Optional[list[int]],
        causal_predictions: Optional[list[int]],
        output_ids: Optional[list[int]],
    ) -> None:
        path = os.environ.get("GAM_SGLANG_SPEC_TRACE_PATH")
        if not path:
            return
        if self._trace_path != path or self._trace_stream is None:
            self._close_trace()
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._trace_stream = open(
                path, "a", encoding="utf-8", buffering=1024 * 1024
            )
            self._trace_path = path
        row = {
            "request": self._request,
            "block": self._blocks,
            "total_len": total_len,
            "output_start": output_start,
            "keep_positions": keep_positions,
            "accepted_output_tokens": output_count,
            "cumulative_output_tokens": self._accepted,
            "nfe_per_block": int(nfe_per_block),
        }
        if (
            drafted_ids is not None
            and causal_predictions is not None
            and output_ids is not None
        ):
            row.update(
                {
                    "drafted_ids": drafted_ids,
                    "causal_predictions": causal_predictions,
                    "output_ids": output_ids,
                    "correction_token": self.last_inherited_token,
                }
            )
        self._trace_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._trace_rows_since_flush += 1
        if self._trace_rows_since_flush >= self._trace_flush_every:
            self._flush_trace()

    def _flush_trace(self) -> None:
        stream = self._trace_stream
        if stream is None:
            return
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
        self._trace_rows_since_flush = 0

    def _close_trace(self) -> None:
        stream = self._trace_stream
        self._trace_stream = None
        self._trace_path = None
        self._trace_rows_since_flush = 0
        if stream is None:
            return
        try:
            stream.flush()
            stream.close()
        except (OSError, ValueError):
            pass


Algorithm = GAMSpeculativeBlock
