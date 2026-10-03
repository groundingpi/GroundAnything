"""KV-cache-compatible HierarchyBlock decoding for GAM Direct Conversion."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from infer.decode.reliability import raw_reliability
from infer.dlm.config import DLMInferenceConfig
from infer.dlm.prefix_cache import Qwen35PrefixDraftRunner
from infer.dlm.selection import (
    dynamic_acceptance,
    entropy_acceptance,
    predictions_and_confidence,
    step_acceptance,
)
from infer.dlm.stop_contract import first_token_sequence_match


def _trace_hierarchy_logits(
    step: int,
    *,
    generated: torch.Tensor,
    block_tokens: torch.Tensor,
    sub_start: int,
    sub_end: int,
    logits: torch.Tensor,
    block_start: int,
) -> int:
    """Opt-in compact trajectory trace used to audit SGLang parity.

    Only top-k CPU scalars are persisted, never the full vocabulary tensor.
    The hook is disabled unless an output path is explicitly supplied.
    """
    path = os.environ.get("GAM_DLM_TRACE_PATH")
    limit = int(os.environ.get("GAM_DLM_TRACE_MAX_STEPS", "24"))
    if not path or step >= limit:
        return step
    step += 1
    local = logits[sub_start:sub_end]
    top_k = min(5, local.shape[-1])
    values, indices = torch.topk(local.float(), k=top_k, dim=-1)
    row = {
        "step": step,
        "phase": "denoise",
        "block_start": block_start,
        "sub_start": sub_start,
        "sub_end": sub_end,
        "generated": generated.detach().cpu().tolist(),
        "block_tokens": block_tokens.detach().cpu().tolist(),
        "top_ids": indices.detach().cpu().tolist(),
        "top_logits": values.detach().cpu().tolist(),
    }
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_path = os.environ.get("GAM_DLM_DUMP_LOGITS_PATH")
    if dump_path and not os.path.exists(dump_path):
        # One full [B32-1, vocab] snapshot is intentionally opt-in.  It is
        # used only to compare the SGLang adapter's first denoise NFE against
        # the canonical Transformers implementation; ordinary evaluation
        # keeps the compact top-k trace above.
        torch.save(
            {
                "step": step,
                "block_start": block_start,
                "sub_start": sub_start,
                "sub_end": sub_end,
                "generated": generated.detach().cpu(),
                "block_tokens": block_tokens.detach().cpu(),
                "logits": logits.detach().float().cpu(),
            },
            dump_path,
        )
    return step


def _enforce_single_bbox_grammar(
    logits: torch.Tensor,
    *,
    response_prefix: list[int],
    block_tokens: torch.Tensor,
    mask_token_id: int,
    structural_token_ids: dict[str, int],
    coordinate_token_id_range: tuple[int, int],
    coordinate_group_width: int = 4,
) -> tuple[int, int | None]:
    """Constrain an open box wrapper to one task-specific coordinate group.

    The state machine advances over resolved tokens and the current constrained
    argmax trajectory.  It never calls the causal model.  The returned position
    marks the first token whose acceptance depends on this grammar trajectory.
    """

    if logits.ndim != 2 or block_tokens.ndim != 1:
        raise ValueError("single-bbox grammar expects [positions,vocab] logits")
    if logits.shape[0] != block_tokens.shape[0]:
        raise ValueError("single-bbox grammar position count mismatch")
    coordinate_min, coordinate_max = coordinate_token_id_range
    if coordinate_group_width not in {2, 4}:
        raise ValueError("coordinate_group_width must be 2 (point) or 4 (bbox)")
    if not 0 <= coordinate_min <= coordinate_max < logits.shape[1]:
        raise ValueError("invalid coordinate token range")
    box_start_id = structural_token_ids.get("box_start")
    box_end_id = structural_token_ids.get("box_end")
    if box_start_id is None or box_end_id is None:
        raise ValueError("single-bbox grammar requires box start/end token IDs")
    if not (
        0 <= box_start_id < logits.shape[1]
        and 0 <= box_end_id < logits.shape[1]
    ):
        raise ValueError("box start/end token IDs are outside the vocabulary")

    # ``box_width`` is None outside a box, otherwise the number of token slots
    # already consumed after the unmatched box start.  Counting slots rather
    # than provisional coordinate values makes the four-coordinate shape exact.
    box_width: int | None = None
    for token in response_prefix:
        if box_width is None:
            if token == box_start_id:
                box_width = 0
        elif token == box_end_id:
            box_width = None
        else:
            box_width += 1

    sensitive_from = 0 if box_width is not None else None
    constrained_positions = 0
    provisional = logits.argmax(dim=-1).tolist()
    resolved_tokens = block_tokens.tolist()
    first_unresolved = next(
        (
            index
            for index, token in enumerate(resolved_tokens)
            if token == mask_token_id
        ),
        len(resolved_tokens),
    )
    for row, resolved in enumerate(resolved_tokens):
        if resolved == mask_token_id:
            row_logits = logits[row]
            if box_width is not None:
                if box_width < coordinate_group_width:
                    row_logits[:coordinate_min] = -torch.inf
                    row_logits[coordinate_max + 1 :] = -torch.inf
                    # The exact coordinate value does not affect the FSM.
                    token = coordinate_min
                else:
                    box_end_logit = row_logits[box_end_id].clone()
                    row_logits.fill_(-torch.inf)
                    row_logits[box_end_id] = box_end_logit
                    token = box_end_id
                constrained_positions += 1
            else:
                token = int(provisional[row])
        else:
            token = int(resolved)

        if box_width is None:
            if token == box_start_id:
                box_width = 0
                if sensitive_from is None:
                    # The transition cannot freeze ahead of an unresolved
                    # predecessor that may change the grammar state.
                    sensitive_from = min(first_unresolved, row)
        elif token == box_end_id:
            box_width = None
        else:
            box_width += 1

    return constrained_positions, sensitive_from


def _prefix_safe_grammar_acceptance(
    accepted: torch.Tensor,
    masked: torch.Tensor,
    *,
    sensitive_from: int | None,
    step_mode: bool,
) -> torch.Tensor:
    """Make grammar-dependent acceptance an ordered unresolved prefix.

    Step mode preserves its selected-token count by moving selections to the
    left edge of the grammar span.  Dynamic mode keeps only its accepted prefix;
    when that prefix is empty it advances the leftmost unresolved token.
    """

    if accepted.ndim != 1 or masked.shape != accepted.shape:
        raise ValueError("accepted and masked must be equal one-dimensional tensors")
    if sensitive_from is None:
        return accepted
    if not 0 <= sensitive_from <= accepted.numel():
        raise ValueError("grammar-sensitive start is outside the sub-block")

    output = accepted.clone()
    indices = torch.nonzero(masked, as_tuple=False).flatten()
    indices = indices[indices.ge(sensitive_from)]
    if indices.numel() == 0:
        return output

    original = output[indices].clone()
    output[indices] = False
    if step_mode:
        count = max(int(original.sum().item()), 1)
        output[indices[:count]] = True
        return output

    prefix_count = 0
    for selected in original.tolist():
        if not selected:
            break
        prefix_count += 1
    # Dynamic decoding must still make progress when the left edge is below
    # threshold; accepting a later high-confidence token would break the FSM.
    output[indices[: max(prefix_count, 1)]] = True
    return output


def _suppress_invalid_terminations(
    logits: torch.Tensor,
    *,
    response_prefix: list[int],
    block_tokens: torch.Tensor,
    mask_token_id: int,
    stop_token_ids: set[int],
    semantic_stop_token_ids: set[int],
    structural_token_ids: dict[str, int],
    coordinate_token_id_range: tuple[int, int],
    coordinate_separator_token_ids: set[int],
    coordinate_group_width: int = 4,
) -> tuple[int, int | None]:
    """Apply the unlimited-group grounding FSM to every unresolved position.

    An open box/quad accepts complete coordinate groups separated by a comma;
    the number of groups is deliberately unlimited.  Other open structures
    suppress EOS and impossible nesting.  The trajectory is recomputed after
    every masked row so later positions never inherit a stale argmax token.
    ``sensitive_from`` makes acceptance ordered whenever an unresolved token
    can change the FSM state.
    """

    if logits.ndim != 2 or block_tokens.ndim != 1:
        raise ValueError("structured termination expects [positions,vocab] logits")
    if logits.shape[0] != block_tokens.shape[0]:
        raise ValueError("structured termination position count mismatch")
    coordinate_min, coordinate_max = coordinate_token_id_range
    if coordinate_group_width not in {2, 4}:
        raise ValueError("coordinate_group_width must be 2 (point) or 4 (bbox)")
    if not 0 <= coordinate_min <= coordinate_max:
        raise ValueError("invalid coordinate token range")
    if not coordinate_separator_token_ids or any(
        token < 0 or token >= logits.shape[1]
        for token in coordinate_separator_token_ids
    ):
        raise ValueError("invalid coordinate separator token contract")
    required = {
        "object_ref_start",
        "object_ref_end",
        "box_start",
        "box_end",
        "quad_start",
        "quad_end",
    }
    if set(structural_token_ids) != required:
        raise ValueError("incomplete structural token contract")

    # Structural end tokens must be grammar-checked even when they are not
    # generation stop tokens.  Formal dense/multi-object tasks deliberately
    # continue after the first box/quad, whereas single-object tasks may also
    # list these IDs in ``semantic_stop_token_ids``.
    candidates = stop_token_ids | semantic_stop_token_ids | set(
        structural_token_ids.values()
    )
    suppressed = 0
    trajectory: list[int] = []
    resolved_tokens = [int(token) for token in block_tokens.tolist()]
    first_unresolved = next(
        (
            index
            for index, token in enumerate(resolved_tokens)
            if token == mask_token_id
        ),
        len(resolved_tokens),
    )
    # Structured validity is prefix-dependent even while no wrapper is open:
    # an unresolved predecessor may later become a wrapper start and make any
    # already accepted successor (including EOS) illegal.  Therefore every
    # structured sub-block must freeze an ordered unresolved prefix.  Merely
    # marking predicted structural tokens is insufficient because a later
    # high-confidence EOS can otherwise be accepted before a lower-confidence
    # ``box_start`` and leave a response ending at the open wrapper.
    sensitive_from: int | None = (
        first_unresolved if first_unresolved < len(resolved_tokens) else None
    )

    def unmatched_start(tokens: list[int], start_id: int, end_id: int) -> int | None:
        stack: list[int] = []
        for index, token in enumerate(tokens):
            if token == start_id:
                stack.append(index)
            elif token == end_id and stack:
                stack.pop()
        return stack[-1] if stack else None

    def state(tokens_before: list[int]) -> tuple[int | None, int | None, int | None]:
        ids = structural_token_ids
        box_start = unmatched_start(tokens_before, ids["box_start"], ids["box_end"])
        quad_start = unmatched_start(tokens_before, ids["quad_start"], ids["quad_end"])
        object_ref_start = unmatched_start(
            tokens_before, ids["object_ref_start"], ids["object_ref_end"]
        )
        return box_start, quad_start, object_ref_start

    def invalid(candidate: int, tokens_before: list[int]) -> bool:
        ids = structural_token_ids
        box_start, quad_start, object_ref_start = state(tokens_before)
        # The canonical GAM entry grammar has no optional gap here:
        # ``REF_END BOX_START`` must be adjacent.  Without this transition,
        # EOS can win immediately after a completed label and silently turn a
        # visually correct prediction into an unparsable empty result.
        awaiting_box_start = bool(tokens_before) and (
            tokens_before[-1] == ids["object_ref_end"]
        )
        if awaiting_box_start:
            return candidate != ids["box_start"]
        any_open = any(
            value is not None for value in (box_start, quad_start, object_ref_start)
        )
        if candidate in {
            ids["object_ref_start"],
            ids["box_start"],
            ids["quad_start"],
        }:
            return any_open
        if candidate == ids["object_ref_end"]:
            return object_ref_start is None or box_start is not None or quad_start is not None
        if candidate == ids["box_end"]:
            if box_start is None:
                return True
            coordinates = sum(
                coordinate_min <= token <= coordinate_max
                for token in tokens_before[box_start + 1 :]
            )
            return coordinates == 0 or coordinates % coordinate_group_width != 0
        if candidate == ids["quad_end"]:
            if quad_start is None:
                return True
            coordinates = sum(
                coordinate_min <= token <= coordinate_max
                for token in tokens_before[quad_start + 1 :]
            )
            return coordinates == 0 or coordinates % 8 != 0
        return candidate in stop_token_ids and any_open

    for row in range(logits.shape[0]):
        tokens_before = response_prefix + trajectory
        box_start, quad_start, _ = state(tokens_before)
        open_start = box_start if box_start is not None else quad_start
        group_width = coordinate_group_width if box_start is not None else 8
        end_id = (
            structural_token_ids["box_end"]
            if box_start is not None
            else structural_token_ids["quad_end"]
        )
        row_logits = logits[row]

        awaiting_box_start = bool(tokens_before) and (
            tokens_before[-1] == structural_token_ids["object_ref_end"]
        )
        if awaiting_box_start:
            box_start_id = structural_token_ids["box_start"]
            box_start_logit = row_logits[box_start_id].clone()
            row_logits.fill_(-torch.inf)
            row_logits[box_start_id] = box_start_logit
            suppressed += 1
            if sensitive_from is None:
                sensitive_from = min(first_unresolved, row)
        elif open_start is not None:
            # All histories produced by this FSM alternate a fixed-width
            # coordinate group with exactly one separator.  Counting since the
            # latest separator is therefore sufficient and preserves an
            # unlimited number of dense-task groups.
            inside = tokens_before[open_start + 1 :]
            last_separator = max(
                (
                    index
                    for index, token in enumerate(inside)
                    if token in coordinate_separator_token_ids
                ),
                default=-1,
            )
            current_group = inside[last_separator + 1 :]
            coordinate_count = sum(
                coordinate_min <= token <= coordinate_max
                for token in current_group
            )
            allowed: list[int]
            if coordinate_count < group_width:
                allowed = list(range(coordinate_min, coordinate_max + 1))
            else:
                allowed = [end_id, *sorted(coordinate_separator_token_ids)]
            allowed_logits = row_logits[allowed].clone()
            row_logits.fill_(-torch.inf)
            row_logits[allowed] = allowed_logits
            suppressed += 1
            if sensitive_from is None:
                sensitive_from = min(first_unresolved, row)
        else:
            for candidate in candidates:
                if 0 <= candidate < logits.shape[1] and invalid(candidate, tokens_before):
                    if torch.isfinite(row_logits[candidate]):
                        row_logits[candidate] = -torch.inf
                        suppressed += 1

        resolved = resolved_tokens[row]
        token = int(row_logits.argmax().item()) if resolved == mask_token_id else resolved
        trajectory.append(token)
        if token in {
            structural_token_ids["object_ref_start"],
            structural_token_ids["box_start"],
            structural_token_ids["quad_start"],
        } and sensitive_from is None:
            sensitive_from = min(first_unresolved, row)

    return suppressed, sensitive_from


def _suppress_repeated_ngrams(
    logits: torch.Tensor,
    *,
    response_prefix: list[int],
    block_tokens: torch.Tensor,
    mask_token_id: int,
    ngram_size: int,
) -> tuple[int, int | None, int]:
    """Ban exact repeated n-grams without emptying the candidate set.

    Structured grammar is a hard constraint while no-repeat is a soft decoding
    preference.  Callers therefore apply grammar first.  If every currently
    finite (and hence grammar-valid) candidate would be banned, retain the
    highest-logit candidate so sampling always has a legal continuation.
    """

    if ngram_size < 2:
        return 0, None, 0
    trajectory: list[int] = []
    resolved_tokens = [int(token) for token in block_tokens.tolist()]
    suppressed = 0
    dead_end_escapes = 0
    sensitive_from: int | None = None
    first_unresolved = next(
        (
            index
            for index, token in enumerate(resolved_tokens)
            if token == mask_token_id
        ),
        len(resolved_tokens),
    )
    for row, resolved in enumerate(resolved_tokens):
        history = response_prefix + trajectory
        if len(history) >= ngram_size - 1:
            suffix = tuple(history[-(ngram_size - 1) :])
            banned = {
                history[index + ngram_size - 1]
                for index in range(len(history) - ngram_size + 1)
                if tuple(history[index : index + ngram_size - 1]) == suffix
            }
            finite_tokens = torch.nonzero(
                torch.isfinite(logits[row]), as_tuple=False
            ).flatten()
            banned_finite = {
                token
                for token in banned
                if 0 <= token < logits.shape[1]
                and torch.isfinite(logits[row, token])
            }
            keep_token: int | None = None
            if finite_tokens.numel() and len(banned_finite) == finite_tokens.numel():
                keep_token = int(logits[row].argmax().item())
                dead_end_escapes += 1
            for token in banned_finite:
                if token != keep_token:
                    logits[row, token] = -torch.inf
                    suppressed += 1
                    if sensitive_from is None:
                        sensitive_from = min(first_unresolved, row)
        token = int(logits[row].argmax().item()) if resolved == mask_token_id else resolved
        trajectory.append(token)
    return suppressed, sensitive_from, dead_end_escapes


class HierarchyBlockDecoder:
    """Decode response blocks with fixed-step or confidence-based remasking."""

    def __init__(self, model: Any, config: DLMInferenceConfig):
        self.model = model
        self.config = config.validate(int(model.block_size))
        if config.mode not in {"hierarchy_step", "hierarchy_dynamic"}:
            raise ValueError(f"HierarchyBlockDecoder cannot run mode={config.mode!r}")

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...] = (),
        stop_token_sequences: tuple[tuple[int, ...], ...] = (),
        semantic_stop_token_ids: tuple[int, ...] = (),
        structural_token_ids: dict[str, int] | None = None,
        coordinate_token_id_range: tuple[int, int] | None = None,
        coordinate_separator_token_ids: tuple[int, ...] = (),
        coordinate_group_width: int = 4,
        no_repeat_ngram_size: int = 32,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cuda_cache_cleanup_interval_tokens: int = 0,
        cuda_cache_cleanup_fraction: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("HierarchyBlock decoding currently requires batch size 1")
        if max_new_tokens < 1:
            return input_ids[:, :0]
        if cuda_cache_cleanup_interval_tokens < 0:
            raise ValueError("cuda_cache_cleanup_interval_tokens must be non-negative")
        if not 0.0 <= cuda_cache_cleanup_fraction <= 1.0:
            raise ValueError("cuda_cache_cleanup_fraction must be in [0, 1]")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")
        if no_repeat_ngram_size == 1 or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be 0 or at least 2")
        if coordinate_group_width not in {2, 4}:
            raise ValueError("coordinate_group_width must be 2 (point) or 4 (bbox)")

        model = self.model
        prompt_length = int(input_ids.shape[1])
        generated = input_ids
        generated_attention = attention_mask
        stop_ids = set(int(value) for value in stop_token_ids)
        semantic_stop_ids = set(int(value) for value in semantic_stop_token_ids)
        termination_ids = stop_ids | semantic_stop_ids
        coordinate_separator_ids = set(
            int(value) for value in coordinate_separator_token_ids
        )
        structured_termination_suppressions = 0
        repeated_ngram_suppressions = 0
        no_repeat_dead_end_escapes = 0
        trace_step = 0
        if self.config.enforce_structured_termination and (
            structural_token_ids is None
            or coordinate_token_id_range is None
            or not coordinate_separator_ids
        ):
            raise ValueError("structured termination requires tokenizer token IDs")
        structural_boundary_ids = (
            {
                int((structural_token_ids or {})["box_end"]),
                int((structural_token_ids or {})["quad_end"]),
            }
            if self.config.commit_structural_boundaries
            else set()
        )

        def request_stop_match() -> tuple[int, int] | None:
            return first_token_sequence_match(
                generated[0, prompt_length:].tolist(), stop_token_sequences
            )

        def constrain_next_logits(
            logits: torch.Tensor, response_prefix: list[int]
        ) -> None:
            nonlocal structured_termination_suppressions
            nonlocal repeated_ngram_suppressions, no_repeat_dead_end_escapes
            unresolved = torch.full(
                (logits.shape[0],),
                model.mask_token_id,
                dtype=torch.long,
                device=logits.device,
            )
            if self.config.enforce_structured_termination:
                count, _ = _suppress_invalid_terminations(
                    logits,
                    response_prefix=response_prefix,
                    block_tokens=unresolved,
                    mask_token_id=model.mask_token_id,
                    stop_token_ids=stop_ids,
                    semantic_stop_token_ids=semantic_stop_ids,
                    structural_token_ids=structural_token_ids or {},
                    coordinate_token_id_range=coordinate_token_id_range or (0, -1),
                    coordinate_separator_token_ids=coordinate_separator_ids,
                    coordinate_group_width=coordinate_group_width,
                )
                structured_termination_suppressions += count
            count, _, escapes = _suppress_repeated_ngrams(
                logits,
                response_prefix=response_prefix,
                block_tokens=unresolved,
                mask_token_id=model.mask_token_id,
                ngram_size=no_repeat_ngram_size,
            )
            repeated_ngram_suppressions += count
            no_repeat_dead_end_escapes += escapes

        prompt_clean_embeds, prompt_position_ids = model._embed_clean(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
        )
        rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        position_axes = int(prompt_position_ids.shape[0])
        batch_size = int(input_ids.shape[0])

        # Qwen3.5 consumes three-axis mRoPE positions, while the plain-Qwen3
        # wrapper keeps a singleton compatibility axis internally and must
        # remove it at the native language-model boundary.  Draft-block calls
        # already normalize positions inside ``GAMQwen3DLM``; causal prefill
        # and cache extension come through this decoder and need the same
        # conversion here.
        def causal_position_ids(positions: torch.Tensor) -> torch.Tensor:
            normalizer = getattr(model, "_text_positions", None)
            return normalizer(positions) if callable(normalizer) else positions

        prompt_cache_positions = torch.arange(prompt_length, device=input_ids.device)
        causal_outputs = model.language_model(
            input_ids=None,
            inputs_embeds=prompt_clean_embeds,
            position_ids=causal_position_ids(prompt_position_ids),
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            cache_position=prompt_cache_positions,
            return_dict=True,
        )
        causal_cache = causal_outputs.past_key_values
        if causal_cache is None:
            raise RuntimeError("Qwen3.5 causal prefill did not return a cache")
        first_logits = model.lm_head(causal_outputs.last_hidden_state[:, -1])
        prefill_dump_path = os.environ.get("GAM_DLM_DUMP_PREFILL_PATH")
        if prefill_dump_path and not os.path.exists(prefill_dump_path):
            torch.save(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "position_ids": prompt_position_ids.detach().cpu(),
                    "first_logits": first_logits.detach().float().cpu(),
                },
                prefill_dump_path,
            )
        first_logits = model._apply_repetition_penalty(
            first_logits, generated, repetition_penalty
        )
        constrain_next_logits(first_logits, [])
        first_prediction, _ = predictions_and_confidence(
            first_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        first_token = first_prediction.unsqueeze(1)
        generated = torch.cat([generated, first_token], dim=1)
        generated_attention = model._extend_optional_sequence(generated_attention, 1, fill=1)

        causal_nfe = 1
        denoise_nfe = 0
        blocks = 0
        accepted_per_step: list[int] = []
        confidence_per_step: list[float] = []
        entropy_per_step: list[float] = []
        terminated = int(first_token[0, 0]) in termination_ids
        prefix_runner = (
            Qwen35PrefixDraftRunner(model, self.config.block_size)
            if self.config.use_prefix_cache
            else None
        )
        prefix_cache_builds = 0
        prefix_cache_draft_calls = 0
        prefix_cache_current_tokens = 0
        prefix_cache_borrowed_bytes_peak = 0
        structured_grammar_constrained_positions = 0
        structural_boundary_commits = 0
        cuda_cache_cleanups = 0
        last_cache_check_tokens = 0
        peak_cuda_reserved_bytes = 0
        def update_progress(active: bool = True) -> None:
            model._inflight_generation_stats = {
                "active": active,
                "decoding": self.config.mode,
                "block_size": self.config.block_size,
                "sub_block_size": self.config.sub_block_size,
                "output_tokens": int(generated.shape[1] - prompt_length),
                "blocks": blocks,
                "nfe": causal_nfe + denoise_nfe,
                "causal_nfe": causal_nfe,
                "denoise_nfe": denoise_nfe,
                "commit_structural_boundaries": self.config.commit_structural_boundaries,
                "structural_boundary_commits": structural_boundary_commits,
                "no_repeat_dead_end_escapes": no_repeat_dead_end_escapes,
                "cuda_cache_cleanups": cuda_cache_cleanups,
                "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
            }

        def maybe_release_fragmented_cuda_cache() -> None:
            """Release inactive blocks created by growing Hierarchy prefixes."""

            nonlocal cuda_cache_cleanups, last_cache_check_tokens, peak_cuda_reserved_bytes
            if cuda_cache_cleanup_interval_tokens == 0 or not generated.is_cuda:
                return
            output_tokens = int(generated.shape[1] - prompt_length)
            if output_tokens - last_cache_check_tokens < cuda_cache_cleanup_interval_tokens:
                return
            last_cache_check_tokens = output_tokens
            reserved = int(torch.cuda.memory_reserved(generated.device))
            allocated = int(torch.cuda.memory_allocated(generated.device))
            peak_cuda_reserved_bytes = max(peak_cuda_reserved_bytes, reserved)
            total = int(torch.cuda.get_device_properties(generated.device).total_memory)
            if reserved < int(total * cuda_cache_cleanup_fraction) or reserved <= 2 * allocated:
                return
            torch.cuda.empty_cache()
            cuda_cache_cleanups += 1

        def cached_causal_logits(tokens: torch.Tensor) -> torch.Tensor:
            nonlocal causal_nfe
            cache_start = int(causal_cache.get_seq_length())
            cache_position = torch.arange(
                cache_start,
                cache_start + tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            )
            position_ids = model._generation_position_ids(
                cache_position, position_axes, batch_size, rope_deltas
            )
            outputs = model.language_model(
                input_ids=tokens,
                attention_mask=None,
                position_ids=causal_position_ids(position_ids),
                past_key_values=causal_cache,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )
            causal_nfe += 1
            return model.lm_head(outputs.last_hidden_state)

        update_progress()
        while not terminated and generated.shape[1] - prompt_length < max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - prompt_length)
            continuation_width = min(self.config.block_size - 1, remaining)
            if continuation_width <= 0:
                break
            blocks += 1
            block_start = generated.shape[1] - 1
            block_tokens = torch.full(
                (1, continuation_width),
                model.mask_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )

            block_prefix_cache = None
            if prefix_runner is not None:
                if block_start == prompt_length:
                    prefix_position_ids = prompt_position_ids
                else:
                    prefix_cache_positions = torch.arange(
                        prompt_length,
                        block_start,
                        dtype=torch.long,
                        device=generated.device,
                    )
                    prefix_suffix_positions = model._generation_position_ids(
                        prefix_cache_positions,
                        position_axes,
                        batch_size,
                        rope_deltas,
                    )
                    prefix_position_ids = torch.cat(
                        [prompt_position_ids, prefix_suffix_positions], dim=2
                    )
                block_prefix_cache = prefix_runner.build(
                    prefix_input_ids=generated[:, :block_start],
                    prefix_position_ids=prefix_position_ids,
                    clean_cache=causal_cache,
                )
                prefix_cache_builds += 1
                prefix_cache_borrowed_bytes_peak = max(
                    prefix_cache_borrowed_bytes_peak,
                    block_prefix_cache.borrowed_bytes,
                )

            resolved_stop_at: int | None = None
            resolved_boundary_at: int | None = None
            if self.config.hierarchy_full_block:
                # A full-length training B32 assigns all response positions to
                # one bidirectional N2N turn.  Keep that complete turn visible
                # for the training-alignment experiment.  This matches turn
                # visibility, not training's random-mask distribution or its
                # naturally short, unpadded tail blocks.
                segments = ((0, continuation_width),)
            else:
                segments = tuple(
                    (
                        sub_start,
                        min(sub_start + self.config.sub_block_size, continuation_width),
                    )
                    for sub_start in range(
                        0, continuation_width, self.config.sub_block_size
                    )
                )
            for sub_start, sub_end in segments:
                sub_width = sub_end - sub_start
                max_iterations = (
                    self.config.denoise_steps
                    if self.config.mode == "hierarchy_step"
                    else sub_width
                )
                for step_index in range(max_iterations):
                    sub_masked = block_tokens[0, sub_start:sub_end].eq(model.mask_token_id)
                    if not bool(sub_masked.any()):
                        break
                    # The default compatibility path exposes only completed
                    # earlier sub-blocks and the current sub-block.  The
                    # explicit training-aligned path exposes the complete B32
                    # turn, matching a full-length training turn's visibility.
                    visible_end = (
                        continuation_width
                        if self.config.hierarchy_full_block
                        else sub_end
                    )
                    visible_block = block_tokens[:, :visible_end]
                    draft_input = torch.cat([generated, visible_block], dim=1)
                    draft_attention = model._extend_optional_sequence(
                        generated_attention, sub_end, fill=1
                    )
                    suffix_cache_positions = torch.arange(
                        prompt_length,
                        draft_input.shape[1],
                        dtype=torch.long,
                        device=draft_input.device,
                    )
                    suffix_positions = model._generation_position_ids(
                        suffix_cache_positions, position_axes, batch_size, rope_deltas
                    )
                    position_ids = torch.cat([prompt_position_ids, suffix_positions], dim=2)
                    if prefix_runner is None:
                        suffix_embeds = model.language_model.embed_tokens(
                            draft_input[:, prompt_length:]
                        )
                        clean_embeds = torch.cat([prompt_clean_embeds, suffix_embeds], dim=1)
                        logits = model.draft_block_logits(
                            draft_input,
                            draft_attention,
                            block_start,
                            precomputed_clean_embeds=clean_embeds,
                            precomputed_position_ids=position_ids,
                        )
                    else:
                        if block_prefix_cache is None:
                            raise RuntimeError("Hierarchy prefix cache was not built")
                        current_ids = draft_input[:, block_start:]
                        current_position_ids = position_ids[:, :, block_start:]
                        logits = prefix_runner.draft_logits(
                            block_prefix_cache,
                            current_ids=current_ids,
                            current_position_ids=current_position_ids,
                        )
                        prefix_cache_draft_calls += 1
                        prefix_cache_current_tokens += int(current_ids.shape[1])
                    trace_step = _trace_hierarchy_logits(
                        trace_step,
                        generated=generated,
                        block_tokens=block_tokens,
                        sub_start=sub_start,
                        sub_end=sub_end,
                        logits=logits,
                        block_start=block_start,
                    )
                    # DecodeV3 calibrates acceptance at T=1 on untouched model
                    # logits, before repetition, grammar and sampling filters.
                    raw_stats = raw_reliability(logits)
                    # Only completed earlier sub-blocks are a sequential
                    # repetition-penalty prefix. Unresolved peers remain truly
                    # bidirectional and are intentionally excluded.
                    penalty_history = (
                        generated
                        if self.config.hierarchy_full_block
                        else torch.cat(
                            [generated, block_tokens[:, :sub_start]], dim=1
                        )
                    )
                    logits = model._apply_repetition_penalty(
                        logits,
                        penalty_history.expand(logits.shape[0], -1),
                        repetition_penalty,
                    )
                    # Apply hard structure constraints before the soft
                    # no-repeat preference.  The latter must never erase all
                    # grammar-valid candidates at a position.
                    grammar_sensitive_from: int | None = None
                    if self.config.structured_max_coordinate_groups == 1:
                        constrained, single_sensitive_from = (
                            _enforce_single_bbox_grammar(
                                logits,
                                response_prefix=generated[
                                    0, prompt_length:
                                ].tolist(),
                                block_tokens=block_tokens[0, :visible_end],
                                mask_token_id=model.mask_token_id,
                                structural_token_ids=structural_token_ids or {},
                                coordinate_token_id_range=(
                                    coordinate_token_id_range or (0, -1)
                                ),
                                coordinate_group_width=coordinate_group_width,
                            )
                        )
                        structured_grammar_constrained_positions += constrained
                        if single_sensitive_from is not None:
                            grammar_sensitive_from = (
                                single_sensitive_from
                                if grammar_sensitive_from is None
                                else min(grammar_sensitive_from, single_sensitive_from)
                            )
                    if self.config.enforce_structured_termination:
                        suppressed, structural_sensitive_from = (
                            _suppress_invalid_terminations(
                                logits,
                                response_prefix=generated[0, prompt_length:].tolist(),
                                block_tokens=block_tokens[0, :visible_end],
                                mask_token_id=model.mask_token_id,
                                stop_token_ids=stop_ids,
                                semantic_stop_token_ids=semantic_stop_ids,
                                structural_token_ids=structural_token_ids or {},
                                coordinate_token_id_range=(
                                    coordinate_token_id_range or (0, -1)
                                ),
                                coordinate_separator_token_ids=(
                                    coordinate_separator_ids
                                ),
                                coordinate_group_width=coordinate_group_width,
                            )
                        )
                        structured_termination_suppressions += suppressed
                        if structural_sensitive_from is not None:
                            grammar_sensitive_from = (
                                structural_sensitive_from
                                if grammar_sensitive_from is None
                                else min(
                                    grammar_sensitive_from,
                                    structural_sensitive_from,
                                )
                            )
                    repeated, repeated_sensitive_from, escapes = (
                        _suppress_repeated_ngrams(
                            logits,
                            response_prefix=generated[
                                0, prompt_length:
                            ].tolist(),
                            block_tokens=block_tokens[0, :visible_end],
                            mask_token_id=model.mask_token_id,
                            ngram_size=no_repeat_ngram_size,
                        )
                    )
                    repeated_ngram_suppressions += repeated
                    no_repeat_dead_end_escapes += escapes
                    if repeated_sensitive_from is not None:
                        grammar_sensitive_from = (
                            repeated_sensitive_from
                            if grammar_sensitive_from is None
                            else min(
                                grammar_sensitive_from,
                                repeated_sensitive_from,
                            )
                        )
                    predictions, confidence = predictions_and_confidence(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    sub_confidence = confidence[sub_start:sub_end]
                    if self.config.mode == "hierarchy_step":
                        sub_accepted = step_acceptance(
                            sub_confidence,
                            sub_masked,
                            step_index=step_index,
                            total_steps=max_iterations,
                        )
                    else:
                        if self.config.acceptance_policy == "entropy":
                            sub_entropy = raw_stats.entropy[sub_start:sub_end]
                            sub_accepted = entropy_acceptance(
                                sub_entropy,
                                sub_masked,
                                threshold=self.config.entropy_threshold,
                            )
                        else:
                            sub_accepted = dynamic_acceptance(
                                sub_confidence,
                                sub_masked,
                                threshold=self.config.confidence_threshold,
                            )
                    local_sensitive_from = None
                    if grammar_sensitive_from is not None:
                        if grammar_sensitive_from < sub_end:
                            local_sensitive_from = max(
                                grammar_sensitive_from - sub_start, 0
                            )
                    sub_accepted = _prefix_safe_grammar_acceptance(
                        sub_accepted,
                        sub_masked,
                        sensitive_from=local_sensitive_from,
                        step_mode=self.config.mode == "hierarchy_step",
                    )
                    accepted_indices = torch.nonzero(sub_accepted, as_tuple=False).flatten()
                    absolute_indices = accepted_indices + sub_start
                    block_tokens[0, absolute_indices] = predictions[absolute_indices]
                    accepted_per_step.append(int(sub_accepted.sum().item()))
                    confidence_per_step.append(
                        float(sub_confidence[sub_accepted].mean().item())
                    )
                    entropy_per_step.append(
                        float(raw_stats.entropy[sub_start:sub_end][sub_accepted].mean().item())
                    )
                    denoise_nfe += 1

                if bool(block_tokens[0, sub_start:sub_end].eq(model.mask_token_id).any()):
                    raise RuntimeError(
                        "HierarchyBlock exhausted denoising budget with unresolved sub-block masks"
                    )
                for index, token in enumerate(block_tokens[0, :sub_end].tolist()):
                    if int(token) in termination_ids:
                        resolved_stop_at = index
                        break
                    if int(token) in structural_boundary_ids:
                        resolved_boundary_at = index
                        break
                if resolved_stop_at is not None or resolved_boundary_at is not None:
                    break

            if (
                resolved_stop_at is None
                and resolved_boundary_at is None
                and bool(block_tokens.eq(model.mask_token_id).any())
            ):
                raise RuntimeError("HierarchyBlock exhausted denoising budget with unresolved masks")

            resolved_at = (
                resolved_stop_at
                if resolved_stop_at is not None
                else resolved_boundary_at
            )
            append = (
                block_tokens[:, : resolved_at + 1]
                if resolved_at is not None
                else block_tokens
            )
            terminated = resolved_stop_at is not None
            if resolved_boundary_at is not None:
                structural_boundary_commits += 1
            generated = torch.cat([generated, append], dim=1)
            generated_attention = model._extend_optional_sequence(
                generated_attention, append.shape[1], fill=1
            )
            sequence_match = request_stop_match()
            if sequence_match is not None:
                generated = generated[:, : prompt_length + sequence_match[1]]
                generated_attention = generated_attention[:, : generated.shape[1]]
                terminated = True
            # The prefix runner owns no cross-block mutable state.  Drop the
            # per-block clone before allocator cleanup so only live model/cache
            # tensors remain referenced.
            block_prefix_cache = None
            if terminated or generated.shape[1] - prompt_length >= max_new_tokens:
                maybe_release_fragmented_cuda_cache()
                update_progress()
                break

            # Commit the completed block in one causal NFE. The last row
            # predicts the inherited first token of the next diffusion block.
            completed_block = generated[:, block_start:]
            next_logits = cached_causal_logits(completed_block)[:, -1]
            next_logits = model._apply_repetition_penalty(
                next_logits, generated, repetition_penalty
            )
            constrain_next_logits(
                next_logits, generated[0, prompt_length:].tolist()
            )
            next_prediction, _ = predictions_and_confidence(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            next_token = next_prediction.unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            generated_attention = model._extend_optional_sequence(
                generated_attention, 1, fill=1
            )
            terminated = int(next_token[0, 0]) in termination_ids
            sequence_match = request_stop_match()
            if sequence_match is not None:
                generated = generated[:, : prompt_length + sequence_match[1]]
                generated_attention = generated_attention[:, : generated.shape[1]]
                terminated = True
            maybe_release_fragmented_cuda_cache()
            update_progress()

        response = generated[:, prompt_length : prompt_length + max_new_tokens]
        model._last_generation_stats = {
            "decoding": self.config.mode,
            "block_size": self.config.block_size,
            "sub_block_size": self.config.sub_block_size,
            "denoise_steps": self.config.denoise_steps,
            "confidence_threshold": self.config.confidence_threshold,
            "acceptance_policy": self.config.acceptance_policy,
            "entropy_threshold": self.config.entropy_threshold,
            "prefix_cache_enabled": self.config.use_prefix_cache,
            "hierarchy_full_block": self.config.hierarchy_full_block,
            "structured_termination_enabled": self.config.enforce_structured_termination,
            "structured_termination_suppressions": structured_termination_suppressions,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repeated_ngram_suppressions": repeated_ngram_suppressions,
            "no_repeat_dead_end_escapes": no_repeat_dead_end_escapes,
            "structured_max_coordinate_groups": (
                self.config.structured_max_coordinate_groups
            ),
            "coordinate_group_width": coordinate_group_width,
            "commit_structural_boundaries": self.config.commit_structural_boundaries,
            "structural_boundary_commits": structural_boundary_commits,
            "structured_grammar_constrained_positions": (
                structured_grammar_constrained_positions
            ),
            "prefix_cache_builds": prefix_cache_builds,
            "prefix_cache_draft_calls": prefix_cache_draft_calls,
            "prefix_cache_current_tokens": prefix_cache_current_tokens,
            "prefix_cache_borrowed_bytes_peak": prefix_cache_borrowed_bytes_peak,
            "semantic_stop_token_ids": sorted(semantic_stop_ids),
            "output_tokens": int(response.shape[1]),
            "blocks": blocks,
            "nfe": causal_nfe + denoise_nfe,
            "causal_nfe": causal_nfe,
            "denoise_nfe": denoise_nfe,
            "accepted_per_step": accepted_per_step,
            "mean_accepted_confidence": (
                sum(confidence_per_step) / len(confidence_per_step)
                if confidence_per_step
                else 0.0
            ),
            "mean_accepted_raw_entropy": (
                sum(entropy_per_step) / len(entropy_per_step)
                if entropy_per_step
                else 0.0
            ),
            "tokens_per_nfe": float(response.shape[1]) / max(causal_nfe + denoise_nfe, 1),
            "causal_cache_tokens": int(causal_cache.get_seq_length()),
            "cuda_cache_cleanups": cuda_cache_cleanups,
            "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "logits_processing_order": [
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
                "sample_or_argmax",
            ],
        }
        model._inflight_generation_stats = {
            **model._last_generation_stats,
            "active": False,
        }
        return response
