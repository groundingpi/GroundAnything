"""Exact B32 state replay and differentiable current-policy log-probability."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from train.rl.config import TraceRLConfig
from train.rl.data import EncodedPrompt
from train.rl.trajectory import GAMTraceTrajectory


@dataclass(frozen=True)
class ReplayTransition:
    step: int
    state_response_ids: list[int]
    block_anchor_index: int
    target_relative_indices: list[int]
    target_ids: list[int]
    old_logprobs: list[float]


def build_replay_transitions(
    trajectory: GAMTraceTrajectory,
    *,
    mask_token_id: int,
    block_size: int = 32,
    sub_block_size: int = 4,
) -> list[ReplayTransition]:
    trajectory.validate(block_size=block_size)
    if sub_block_size <= 0 or sub_block_size >= block_size:
        raise ValueError("invalid TraceRL sub-block size")
    unique_steps = sorted(
        {step for step, action in zip(trajectory.commit_step, trajectory.action_mask, strict=True) if action}
    )
    transitions: list[ReplayTransition] = []
    for step in unique_steps:
        actions = [
            index
            for index, (value, action) in enumerate(
                zip(trajectory.commit_step, trajectory.action_mask, strict=True)
            )
            if value == step and action
        ]
        anchor = (actions[0] // block_size) * block_size
        if any((index // block_size) * block_size != anchor for index in actions):
            raise ValueError("one commit step cannot span physical B32 blocks")
        maximum_relative = max(index - anchor - 1 for index in actions)
        sub_block_end = ((maximum_relative // sub_block_size) + 1) * sub_block_size
        block_end = min(
            anchor + 1 + sub_block_end,
            anchor + block_size,
            len(trajectory.output_ids),
        )
        current = []
        for index in range(anchor + 1, block_end):
            committed_before = trajectory.commit_step[index] < step
            current.append(trajectory.output_ids[index] if committed_before else int(mask_token_id))
        response = trajectory.output_ids[: anchor + 1] + current
        transitions.append(
            ReplayTransition(
                step=step,
                state_response_ids=response,
                block_anchor_index=anchor,
                target_relative_indices=[index - anchor - 1 for index in actions],
                target_ids=[trajectory.output_ids[index] for index in actions],
                old_logprobs=[trajectory.old_logprobs[index] for index in actions],
            )
        )
    if sum(len(item.target_ids) for item in transitions) != trajectory.action_count:
        raise RuntimeError("replay action conservation failed")
    return transitions


class ReplayPromptCache:
    """Cache only frozen vision features; language embeddings remain trainable."""

    def __init__(self, model, prompt: EncodedPrompt):
        self.prompt = prompt
        self.image_features: torch.Tensor | None = None
        if prompt.pixel_values is not None:
            with torch.no_grad():
                features = model.multimodal_model.get_image_features(
                    prompt.pixel_values,
                    prompt.image_grid_thw,
                    patch_positions=prompt.patch_positions,
                )
                self.image_features = torch.cat(features, dim=0).detach()

    def clean_embeds_and_positions(self, model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.ones_like(input_ids, dtype=torch.long)
        embeds = model.multimodal_model.get_input_embeddings()(input_ids)
        if self.image_features is not None:
            image_embeds = self.image_features.to(embeds.device, embeds.dtype)
            image_mask, _ = model.multimodal_model.get_placeholder_mask(
                input_ids,
                embeds,
                image_features=image_embeds,
            )
            embeds = embeds.masked_scatter(image_mask, image_embeds)
        positions = attention.cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 1)
        return embeds, positions.unsqueeze(0)


def transition_workload_tokens(prompt: EncodedPrompt, transition: ReplayTransition) -> int:
    """Conservative clean+noisy token workload used for lossless micro-batching."""

    sequence = int(prompt.input_ids.shape[1]) + len(transition.state_response_ids)
    return 2 * sequence


def bucket_replay_transitions(
    prompt: EncodedPrompt,
    transitions: list[ReplayTransition],
    *,
    token_budget: int,
    maximum_batch_size: int,
) -> list[list[ReplayTransition]]:
    """Length-bucket exact states without changing their loss or scheduler semantics."""

    if token_budget <= 0 or maximum_batch_size <= 0:
        raise ValueError("replay token budget and maximum batch size must be positive")
    ordered = sorted(transitions, key=lambda item: len(item.state_response_ids))
    batches: list[list[ReplayTransition]] = []
    current: list[ReplayTransition] = []
    current_max = 0
    for transition in ordered:
        length = transition_workload_tokens(prompt, transition)
        proposed_max = max(current_max, length)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > maximum_batch_size
            or proposed_max * proposed_size > token_budget
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(transition)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def differentiable_transition_batch_logprobs(
    model,
    prompt: EncodedPrompt,
    prompt_cache: ReplayPromptCache,
    transitions: list[ReplayTransition],
    config: TraceRLConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay multiple exact states in one BF16 forward.

    States are merely right-padded for kernel efficiency.  Each sample retains
    its own absolute positions, B32 turn boundary and target action set.  The
    returned current/old vectors are concatenated in transition order.
    """

    if not transitions:
        raise ValueError("empty replay transition batch")
    device = prompt.input_ids.device
    prompt_ids = prompt.input_ids[0]
    prompt_length = int(prompt_ids.numel())
    cfg = model.config
    vision_ids = {
        int(cfg.image_token_id),
        int(cfg.video_token_id),
        int(cfg.vision_start_token_id),
    }

    sequences = [
        torch.cat(
            [
                prompt_ids,
                torch.tensor(item.state_response_ids, dtype=prompt_ids.dtype, device=device),
            ]
        )
        for item in transitions
    ]
    lengths = [int(item.numel()) for item in sequences]
    maximum = max(lengths)
    batch_size = len(sequences)
    input_ids = torch.zeros(
        (batch_size, maximum), dtype=prompt_ids.dtype, device=device
    )
    clean_valid = torch.zeros((batch_size, maximum), dtype=torch.bool, device=device)
    clean_turn = torch.zeros((batch_size, maximum), dtype=torch.long, device=device)
    clean_rows: list[torch.Tensor] = []
    clean_positions = torch.ones((batch_size, maximum), dtype=torch.long, device=device)
    noisy_rows: list[torch.Tensor] = []
    noisy_turn_rows: list[torch.Tensor] = []
    noisy_position_rows: list[torch.Tensor] = []
    target_compact_rows: list[torch.Tensor] = []

    for batch_index, (sequence, transition) in enumerate(zip(sequences, transitions, strict=True)):
        length = lengths[batch_index]
        input_ids[batch_index, :length] = sequence
        clean_valid[batch_index, :length] = True
        block_start = prompt_length + int(transition.block_anchor_index)
        clean_turn[batch_index, block_start:length] = 1
        positions = torch.arange(length, dtype=torch.long, device=device)
        clean_positions[batch_index, :length] = positions

        embeds = model.multimodal_model.get_input_embeddings()(sequence.unsqueeze(0))[0]
        if prompt_cache.image_features is not None:
            image_embeds = prompt_cache.image_features.to(embeds.device, embeds.dtype)
            image_mask, _ = model.multimodal_model.get_placeholder_mask(
                sequence.unsqueeze(0),
                embeds.unsqueeze(0),
                image_features=image_embeds,
            )
            embeds = embeds.masked_scatter(image_mask[0], image_embeds)
        clean_rows.append(embeds)

        text = torch.ones(length, dtype=torch.bool, device=device)
        for token_id in vision_ids:
            text &= sequence.ne(token_id)
        text_indices = torch.nonzero(text, as_tuple=False).flatten()
        noisy_ids = sequence[text]
        noisy_rows.append(model.language_model.embed_tokens(noisy_ids))
        noisy_turn_rows.append(clean_turn[batch_index, :length][text])
        noisy_position_rows.append(positions[text])

        target_absolute = block_start + 1 + torch.tensor(
            transition.target_relative_indices, dtype=torch.long, device=device
        )
        compact = torch.searchsorted(text_indices, target_absolute)
        if not torch.equal(text_indices[compact], target_absolute):
            raise RuntimeError("TraceRL replay target contains a vision placeholder")
        if bool(compact.eq(0).any()):
            raise RuntimeError("TraceRL token-shift target has no predecessor")
        target_compact_rows.append(compact - config.token_shift)

    hidden_size = int(clean_rows[0].shape[-1])
    clean = clean_rows[0].new_zeros((batch_size, maximum, hidden_size))
    noisy_maximum = max(int(row.shape[0]) for row in noisy_rows)
    noisy = noisy_rows[0].new_zeros((batch_size, noisy_maximum, hidden_size))
    noisy_valid = torch.zeros((batch_size, noisy_maximum), dtype=torch.bool, device=device)
    noisy_turn = torch.zeros((batch_size, noisy_maximum), dtype=torch.long, device=device)
    noisy_positions = torch.ones((batch_size, noisy_maximum), dtype=torch.long, device=device)
    for index, (clean_row, noisy_row) in enumerate(zip(clean_rows, noisy_rows, strict=True)):
        clean[index, : clean_row.shape[0]] = clean_row
        noisy[index, : noisy_row.shape[0]] = noisy_row
        noisy_valid[index, : noisy_row.shape[0]] = True
        noisy_turn[index, : noisy_row.shape[0]] = noisy_turn_rows[index]
        noisy_positions[index, : noisy_row.shape[0]] = noisy_position_rows[index]

    streams = {
        "noisy": noisy,
        "noisy_valid": noisy_valid,
        "noisy_turn": noisy_turn,
        "noisy_positions": noisy_positions.unsqueeze(0),
        "clean": clean,
        "clean_valid": clean_valid,
        "clean_turn": clean_turn,
        "clean_positions": clean_positions.unsqueeze(0),
    }
    noisy_hidden, _ = model._hybrid_language_forward(streams)
    selected_hidden = torch.cat(
        [
            noisy_hidden[index].index_select(0, compact)
            for index, compact in enumerate(target_compact_rows)
        ],
        dim=0,
    )
    targets = torch.tensor(
        [token for item in transitions for token in item.target_ids],
        dtype=torch.long,
        device=device,
    )
    old = torch.tensor(
        [value for item in transitions for value in item.old_logprobs],
        dtype=torch.float32,
        device=device,
    )
    logits = model.lm_head(selected_hidden).float() / config.temperature
    current = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return current, old


def differentiable_transition_logprobs(
    model,
    prompt: EncodedPrompt,
    prompt_cache: ReplayPromptCache,
    transition: ReplayTransition,
    config: TraceRLConfig,
) -> torch.Tensor:
    response = torch.tensor(
        transition.state_response_ids,
        dtype=prompt.input_ids.dtype,
        device=prompt.input_ids.device,
    ).unsqueeze(0)
    input_ids = torch.cat([prompt.input_ids, response], dim=1)
    attention = torch.ones_like(input_ids, dtype=prompt.attention_mask.dtype)
    block_start = int(prompt.input_ids.shape[1]) + transition.block_anchor_index
    clean_embeds, position_ids = prompt_cache.clean_embeds_and_positions(model, input_ids)
    streams = model._build_inference_streams(
        input_ids,
        attention,
        clean_embeds,
        position_ids,
        block_start,
    )
    noisy_hidden, _ = model._hybrid_language_forward(streams)

    cfg = model.config
    text = torch.ones_like(input_ids[0], dtype=torch.bool)
    for token_id in (
        int(cfg.image_token_id),
        int(cfg.video_token_id),
        int(cfg.vision_start_token_id),
    ):
        text &= input_ids[0].ne(token_id)
    text_indices = torch.nonzero(text, as_tuple=False).flatten()
    target_positions = torch.arange(
        block_start + 1,
        input_ids.shape[1],
        device=input_ids.device,
    )
    compact = torch.searchsorted(text_indices, target_positions)
    if not torch.equal(text_indices[compact], target_positions):
        raise RuntimeError("TraceRL replay target contains a vision placeholder")
    prediction_hidden = noisy_hidden[0, compact - config.token_shift]
    logits = model.lm_head(prediction_hidden).float() / config.temperature
    selected_positions = torch.tensor(
        transition.target_relative_indices,
        dtype=torch.long,
        device=logits.device,
    )
    targets = torch.tensor(transition.target_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(0, selected_positions)
    return torch.log_softmax(selected_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
