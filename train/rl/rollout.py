"""Real fixed-scheduler GAM DecodeV4 rollout with token commit tracing."""

from __future__ import annotations

import torch

from train.rl.config import TraceRLConfig
from train.rl.data import EncodedPrompt
from train.rl.trajectory import GAMTraceTrajectory
from infer.decode.reliability import raw_reliability


def _sample_and_logprob(
    raw_logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = raw_logits.float() / float(temperature)
    logprobs = torch.log_softmax(logits, dim=-1)
    probabilities = logprobs.exp()
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    chosen = logprobs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, chosen


class GAMTraceRollout:
    def __init__(self, model, config: TraceRLConfig, eos_token_id: int):
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
    ) -> GAMTraceTrajectory:
        model = self.model
        config = self.config
        model.eval()
        input_ids = prompt.input_ids
        attention_mask = prompt.attention_mask
        if input_ids.shape[0] != 1 or not bool(attention_mask.bool().all()):
            raise ValueError("GAM TraceRL rollout requires one unpadded prompt")
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(int(seed) % (2**63 - 1))

        prompt_length = int(input_ids.shape[1])
        generated = input_ids
        generated_attention = attention_mask
        prompt_clean_embeds, prompt_position_ids = model._embed_clean(
            input_ids,
            attention_mask,
            prompt.pixel_values,
            prompt.image_grid_thw,
            None,
            None,
            prompt.mm_token_type_ids,
            prompt.patch_positions,
        )
        position_axes = int(prompt_position_ids.shape[0])
        batch_size = int(input_ids.shape[0])
        causal_prompt_position_ids = (
            prompt_position_ids[0] if position_axes == 1 else prompt_position_ids
        )
        rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()

        cache_positions = torch.arange(prompt_length, device=input_ids.device)
        causal = model.language_model(
            input_ids=None,
            inputs_embeds=prompt_clean_embeds,
            position_ids=causal_prompt_position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            cache_position=cache_positions,
            return_dict=True,
        )
        causal_cache = causal.past_key_values
        if causal_cache is None:
            raise RuntimeError("TraceRL causal prefill returned no KV cache")

        first_logits = model.lm_head(causal.last_hidden_state[:, -1])
        first_token, _ = _sample_and_logprob(
            first_logits,
            temperature=config.temperature,
            generator=generator,
        )
        generated = torch.cat([generated, first_token[:, None]], dim=1)
        generated_attention = model._extend_optional_sequence(generated_attention, 1, fill=1)

        output_ids = [int(first_token.item())]
        commit_step = [-1]
        old_logprobs = [0.0]
        action_mask = [0]
        forced_flags = [0]
        denoise_forwards = 0
        global_commit_step = 0
        terminated = output_ids[0] == self.eos_token_id

        def cached_causal_logits(tokens: torch.Tensor) -> torch.Tensor:
            cache_start = int(causal_cache.get_seq_length())
            positions = torch.arange(
                cache_start,
                cache_start + tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            )
            position_ids = model._generation_position_ids(
                positions, position_axes, batch_size, rope_deltas
            )
            if position_axes == 1:
                position_ids = position_ids[0]
            outputs = model.language_model(
                input_ids=tokens,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=causal_cache,
                use_cache=True,
                cache_position=positions,
                return_dict=True,
            )
            return model.lm_head(outputs.last_hidden_state)

        while not terminated and len(output_ids) < config.max_new_tokens:
            width = min(config.diffusion_tokens_per_block, config.max_new_tokens - len(output_ids))
            if width <= 0:
                break
            block_start = int(generated.shape[1] - 1)
            block_tokens = torch.full(
                (1, width),
                model.mask_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            block_steps = [-2] * width
            block_logprobs = [0.0] * width
            block_forced = [0] * width

            for sub_start in range(0, width, config.sub_block_size):
                sub_end = min(sub_start + config.sub_block_size, width)
                for _ in range(sub_end - sub_start):
                    masked = block_tokens[0, sub_start:sub_end].eq(model.mask_token_id)
                    if not bool(masked.any()):
                        break
                    visible_block = block_tokens[:, :sub_end]
                    draft_input = torch.cat([generated, visible_block], dim=1)
                    draft_attention = model._extend_optional_sequence(
                        generated_attention, sub_end, fill=1
                    )
                    suffix_ids = draft_input[:, prompt_length:]
                    suffix_embeds = model.language_model.embed_tokens(suffix_ids)
                    clean_embeds = torch.cat([prompt_clean_embeds, suffix_embeds], dim=1)
                    suffix_positions = torch.arange(
                        prompt_length,
                        draft_input.shape[1],
                        dtype=torch.long,
                        device=draft_input.device,
                    )
                    position_ids = torch.cat(
                        [
                            prompt_position_ids,
                            model._generation_position_ids(
                                suffix_positions, position_axes, batch_size, rope_deltas
                            ),
                        ],
                        dim=2,
                    )
                    logits = model.draft_block_logits(
                        draft_input,
                        draft_attention,
                        block_start,
                        precomputed_clean_embeds=clean_embeds,
                        precomputed_position_ids=position_ids,
                    )
                    local_logits = logits[sub_start:sub_end]
                    reliability = raw_reliability(local_logits)
                    predictions, chosen_logprobs = _sample_and_logprob(
                        local_logits,
                        temperature=config.temperature,
                        generator=generator,
                    )
                    accepted = masked & reliability.entropy.le(config.entropy_threshold)
                    forced = not bool(accepted.any())
                    if forced:
                        scores = reliability.entropy.masked_fill(~masked, torch.inf)
                        accepted[scores.argmin()] = True
                    global_commit_step += 1
                    denoise_forwards += 1
                    for local_index in torch.nonzero(accepted, as_tuple=False).flatten().tolist():
                        absolute = sub_start + int(local_index)
                        token = int(predictions[local_index].item())
                        block_tokens[0, absolute] = token
                        block_steps[absolute] = global_commit_step
                        block_logprobs[absolute] = float(chosen_logprobs[local_index].item())
                        block_forced[absolute] = int(forced)

                if bool(block_tokens[0, sub_start:sub_end].eq(model.mask_token_id).any()):
                    raise RuntimeError("TraceRL forced-one schedule left an unresolved sub-block")

            if any(step < 0 for step in block_steps):
                raise RuntimeError("TraceRL block has an unrecorded diffusion commit")
            block_list = [int(token) for token in block_tokens[0].tolist()]
            generated = torch.cat([generated, block_tokens], dim=1)
            generated_attention = model._extend_optional_sequence(generated_attention, width, fill=1)
            output_ids.extend(block_list)
            commit_step.extend(block_steps)
            old_logprobs.extend(block_logprobs)
            action_mask.extend([1] * width)
            forced_flags.extend(block_forced)

            terminated = self.eos_token_id in block_list
            if terminated or len(output_ids) >= config.max_new_tokens:
                break

            completed_block = generated[:, block_start:]
            anchor_logits = cached_causal_logits(completed_block)[:, -1]
            anchor, _ = _sample_and_logprob(
                anchor_logits,
                temperature=config.temperature,
                generator=generator,
            )
            generated = torch.cat([generated, anchor[:, None]], dim=1)
            generated_attention = model._extend_optional_sequence(generated_attention, 1, fill=1)
            output_ids.append(int(anchor.item()))
            commit_step.append(-1)
            old_logprobs.append(0.0)
            action_mask.append(0)
            forced_flags.append(0)
            terminated = output_ids[-1] == self.eos_token_id

        first_eos = next(
            (index for index, token in enumerate(output_ids) if token == self.eos_token_id),
            None,
        )
        visible_mask = [
            int(first_eos is None or index <= first_eos) for index in range(len(output_ids))
        ]
        action_mask = [
            int(action and visible)
            for action, visible in zip(action_mask, visible_mask, strict=True)
        ]
        forced_one_tokens = sum(
            forced and action
            for forced, action in zip(forced_flags, action_mask, strict=True)
        )
        normal_accept_tokens = sum(action_mask) - forced_one_tokens
        trajectory = GAMTraceTrajectory(
            output_ids=output_ids,
            commit_step=commit_step,
            old_logprobs=old_logprobs,
            action_mask=action_mask,
            visible_mask=visible_mask,
            policy_version=policy_version,
            prompt_id=prompt_id,
            forced_one_tokens=forced_one_tokens,
            normal_accept_tokens=normal_accept_tokens,
            num_denoise_forwards=denoise_forwards,
        )
        return trajectory.validate(block_size=config.block_size)
