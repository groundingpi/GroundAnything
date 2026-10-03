"""Register the audited GAM GroundAnything/Qwen3 model for causal RL rollout."""

from __future__ import annotations

from train.rl.branches.dare.sglang_causal_adapter import GAMCausalSGLangModel


# SGLang's external registry keys classes by ``__name__``.
GAMCausalSGLangModel.__name__ = "Fast_dVLMForConditionalGeneration"
EntryClass = GAMCausalSGLangModel
