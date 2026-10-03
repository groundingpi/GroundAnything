"""Cached causal baseline for wall-clock comparisons with DLM decoding."""

from __future__ import annotations

from typing import Any

import torch

from infer.dlm.selection import predictions_and_confidence


@torch.inference_mode()
def causal_generate_cached(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...] = (),
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Native AR decoding with Qwen3.5 hybrid KV/recurrent cache."""

    if input_ids.shape[0] != 1:
        raise ValueError("cached causal decoding currently requires batch size 1")
    if max_new_tokens < 1:
        return input_ids[:, :0]
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be strictly positive")

    prompt_length = int(input_ids.shape[1])
    generated = input_ids
    stop_ids = set(int(value) for value in stop_token_ids)
    prompt_embeds, prompt_positions = model._embed_clean(
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
    axes = int(prompt_positions.shape[0])
    batch_size = int(input_ids.shape[0])
    cache_position = torch.arange(prompt_length, device=input_ids.device)
    outputs = model.language_model(
        input_ids=None,
        inputs_embeds=prompt_embeds,
        position_ids=model._native_position_ids(prompt_positions),
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )
    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("Qwen3.5 causal prefill did not return a cache")
    logits = model.lm_head(outputs.last_hidden_state[:, -1])
    nfe = 1

    while generated.shape[1] - prompt_length < max_new_tokens:
        logits = model._apply_repetition_penalty(logits, generated, repetition_penalty)
        prediction, _ = predictions_and_confidence(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token = prediction.unsqueeze(1)
        generated = torch.cat([generated, token], dim=1)
        if int(token[0, 0]) in stop_ids:
            break
        if generated.shape[1] - prompt_length >= max_new_tokens:
            break
        cache_start = int(cache.get_seq_length())
        token_position = torch.arange(
            cache_start, cache_start + 1, dtype=torch.long, device=input_ids.device
        )
        position_ids = model._generation_position_ids(
            token_position, axes, batch_size, rope_deltas
        )
        outputs = model.language_model(
            input_ids=token,
            attention_mask=None,
            position_ids=model._native_position_ids(position_ids),
            past_key_values=cache,
            use_cache=True,
            cache_position=token_position,
            return_dict=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1])
        nfe += 1

    response = generated[:, prompt_length : prompt_length + max_new_tokens]
    model._last_generation_stats = {
        "decoding": "causal_cached",
        "output_tokens": int(response.shape[1]),
        "nfe": nfe,
        "causal_nfe": nfe,
        "tokens_per_nfe": float(response.shape[1]) / max(nfe, 1),
        "causal_cache_tokens": int(cache.get_seq_length()),
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
    model._inflight_generation_stats = {**model._last_generation_stats, "active": False}
    return response
