"""Make causal HF state names loadable by the audited GAM SGLang model."""

from __future__ import annotations

from infer.engines.sglang_gam.gam_qwen3 import FastDVLMForConditionalGeneration


class GAMCausalSGLangModel(FastDVLMForConditionalGeneration):
    def load_weights(self, weights):
        def normalized():
            for name, tensor in weights:
                if name.startswith("base_model."):
                    yield name, tensor
                elif name == "lm_head.weight" or name.startswith("model."):
                    yield "base_model." + name, tensor
                else:
                    yield name, tensor

        return super().load_weights(normalized())


GAMCausalSGLangModel.__name__ = "Fast_dVLMForConditionalGeneration"
EntryClass = GAMCausalSGLangModel

