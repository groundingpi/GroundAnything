"""Single dispatch point for every supported GAM DLM inference mode."""

from __future__ import annotations

from typing import Any

import torch

from infer.dlm.config import DLMInferenceConfig
from infer.dlm.hierarchy import HierarchyBlockDecoder


class DLMInferenceEngine:
    def __init__(self, model: Any):
        self.model = model

    @torch.inference_mode()
    def generate(self, config: DLMInferenceConfig, **generation_args: Any) -> torch.Tensor:
        config.validate(int(self.model.block_size))
        if config.mode in {"hierarchy_step", "hierarchy_dynamic"}:
            return HierarchyBlockDecoder(self.model, config).generate(**generation_args)
        temperature = float(generation_args.pop("temperature", 0.0))
        top_p = float(generation_args.pop("top_p", 1.0))
        top_k = generation_args.pop("top_k", None)
        if temperature != 0.0 or top_p != 1.0 or top_k is not None:
            raise ValueError(
                "speculative decoding does not support sampling; use hierarchy_step, "
                "hierarchy_dynamic, or causal_cached"
            )
        # The exact Speculative implementation currently accepts only atomic
        # stop IDs.  The HTTP layer still applies decoded stop-sequence
        # trimming to its output, while Hierarchy can terminate sequences
        # during generation.
        generation_args.pop("stop_token_sequences", None)
        response = self.model.speculative_generate(
            **generation_args,
            inference_block_size=config.block_size,
        )
        self.model._last_generation_stats.update(
            {
                "decoding": "speculative",
                "block_size": config.block_size,
            }
        )
        return response
