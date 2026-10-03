"""Ordinary causal KV-cache rollout used by Causal-JustGRPO."""

from __future__ import annotations

import torch

from train.rl.data import EncodedPrompt
from train.rl.current.config import CausalJustGRPOConfig
from train.rl.current.trajectory import CausalTrajectory


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = logits.float() / float(temperature)
    logprobs = torch.log_softmax(scaled, dim=-1)
    token = torch.multinomial(logprobs.exp(), 1, generator=generator).squeeze(-1)
    chosen = logprobs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    return token, chosen


class CausalRollout:
    def __init__(self, model, config: CausalJustGRPOConfig, eos_token_id: int):
        self.model = model
        self.config = config.validate()
        self.eos_token_id = int(eos_token_id)

    @torch.no_grad()
    def generate(
        self,
        prompt: EncodedPrompt,
        *,
        seed: int,
        prompt_id: str,
        policy_version: str,
    ) -> CausalTrajectory:
        model = self.model
        config = self.config
        model.eval()
        if prompt.input_ids.shape[0] != 1 or not bool(prompt.attention_mask.bool().all()):
            raise ValueError("DLM RL V2 requires one unpadded prompt per rank")
        generator = torch.Generator(device=prompt.input_ids.device)
        generator.manual_seed(int(seed) % (2**63 - 1))

        clean_embeds, position_ids = model._embed_clean(
            prompt.input_ids,
            prompt.attention_mask,
            prompt.pixel_values,
            prompt.image_grid_thw,
            None,
            None,
            prompt.mm_token_type_ids,
            prompt.patch_positions,
        )
        axes = int(position_ids.shape[0])
        language_positions = position_ids[0] if axes == 1 else position_ids
        cache_positions = torch.arange(prompt.input_ids.shape[1], device=prompt.input_ids.device)
        outputs = model.language_model(
            input_ids=None,
            inputs_embeds=clean_embeds,
            position_ids=language_positions,
            attention_mask=prompt.attention_mask,
            past_key_values=None,
            use_cache=True,
            cache_position=cache_positions,
            return_dict=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("DLM RL V2 causal prefill returned no KV cache")
        logits = model.lm_head(outputs.last_hidden_state[:, -1])
        rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()

        completion_ids: list[int] = []
        old_logprobs: list[float] = []
        for _ in range(config.max_completion_tokens):
            token, logprob = _sample_token(
                logits,
                temperature=config.temperature,
                generator=generator,
            )
            token_id = int(token.item())
            completion_ids.append(token_id)
            old_logprobs.append(float(logprob.item()))
            if token_id == self.eos_token_id:
                break

            cache_start = int(cache.get_seq_length())
            token_position = torch.arange(
                cache_start,
                cache_start + 1,
                dtype=torch.long,
                device=token.device,
            )
            next_positions = model._generation_position_ids(
                token_position,
                axes,
                1,
                rope_deltas,
            )
            if axes == 1:
                next_positions = next_positions[0]
            outputs = model.language_model(
                input_ids=token[:, None],
                attention_mask=None,
                position_ids=next_positions,
                past_key_values=cache,
                use_cache=True,
                cache_position=token_position,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = model.lm_head(outputs.last_hidden_state[:, -1])

        stop_reason = "eos" if completion_ids[-1] == self.eos_token_id else "length"
        return CausalTrajectory(
            completion_ids=completion_ids,
            response_mask=[1] * len(completion_ids),
            old_logprobs=old_logprobs,
            prompt_id=prompt_id,
            policy_version=policy_version,
            stop_reason=stop_reason,
        ).validate(self.eos_token_id)


def causal_teacher_forcing_logprobs(
    model,
    prompt: EncodedPrompt,
    trajectory: CausalTrajectory,
    config: CausalJustGRPOConfig,
) -> torch.Tensor:
    """Exact shifted causal log-probabilities over completion tokens only."""

    completion = torch.tensor(
        trajectory.completion_ids,
        dtype=prompt.input_ids.dtype,
        device=prompt.input_ids.device,
    ).unsqueeze(0)
    full_ids = torch.cat([prompt.input_ids, completion], dim=1)
    full_attention = torch.cat(
        [
            prompt.attention_mask,
            torch.ones_like(completion, dtype=prompt.attention_mask.dtype),
        ],
        dim=1,
    )
    full_mm_types = torch.cat(
        [
            prompt.mm_token_type_ids,
            torch.zeros_like(completion, dtype=prompt.mm_token_type_ids.dtype),
        ],
        dim=1,
    )
    embeds, position_ids = model._embed_clean(
        full_ids,
        full_attention,
        prompt.pixel_values,
        prompt.image_grid_thw,
        None,
        None,
        full_mm_types,
        prompt.patch_positions,
    )
    axes = int(position_ids.shape[0])
    language_positions = position_ids[0] if axes == 1 else position_ids
    outputs = model.language_model(
        input_ids=None,
        inputs_embeds=embeds,
        position_ids=language_positions,
        attention_mask=full_attention,
        past_key_values=None,
        use_cache=False,
        cache_position=torch.arange(full_ids.shape[1], device=full_ids.device),
        return_dict=True,
    )
    prompt_length = int(prompt.input_ids.shape[1])
    prediction_hidden = outputs.last_hidden_state[:, prompt_length - 1 : -1]
    logits = model.lm_head(prediction_hidden).float() / float(config.temperature)
    targets = completion.long()
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = torch.tensor(
        trajectory.response_mask,
        dtype=torch.bool,
        device=logprobs.device,
    ).unsqueeze(0)
    selected = logprobs.masked_select(mask)
    if selected.numel() != len(trajectory.completion_ids):
        raise RuntimeError("DLM RL V2 teacher-forcing shift/mask alignment failed")
    return selected
