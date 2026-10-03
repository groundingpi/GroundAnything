"""Token-level clipped TraceRL objective."""

from __future__ import annotations

import torch


def group_normalized_advantage(rewards: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    values = rewards.float()
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("GRPO rewards must be a one-dimensional group")
    std = values.std(unbiased=False)
    if float(std) == 0.0:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


def tracerl_token_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: torch.Tensor | float,
    *,
    clip_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if current_logprobs.ndim != 1 or current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current and old log-probabilities must be aligned vectors")
    if current_logprobs.numel() == 0:
        raise ValueError("TraceRL loss requires at least one action")
    current = current_logprobs.float()
    old = old_logprobs.float()
    adv = torch.as_tensor(advantage, dtype=torch.float32, device=current.device).detach()
    ratio = torch.exp(current - old)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    loss = -surrogate.mean()
    telemetry = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_p95": torch.quantile(ratio.detach(), 0.95),
        "clip_fraction": ratio.detach().sub(1.0).abs().gt(clip_epsilon).float().mean(),
        "mean_abs_logprob_difference": current.detach().sub(old).abs().mean(),
        "p99_abs_logprob_difference": torch.quantile(current.detach().sub(old).abs(), 0.99),
    }
    return loss, telemetry
