"""Pure source transforms for GroundAnything checkpoint serving compatibility."""

import ast
from functools import lru_cache, wraps
from pathlib import Path


def _accept_prepared_source(module):
    """Accept a complete canonical overlay without applying its patches twice."""
    def decorate(transform):
        @lru_cache(maxsize=1)
        def canonical():
            original = (Path(__file__).parent / "vlm" / module).read_text()
            # Some releases already ship the complete serving-compatible model
            # sources. Do not patch that trusted canonical a second time. Remote
            # input is accepted only when its full AST matches this canonical.
            prepared_markers = {
                "configuration_groundinganything.py": (
                    "@dataclass(init=False)", "from transformers.models.qwen3.configuration_qwen3 import Qwen3Config",
                    "PreTrainedConfig.__post_init__(self, **kwargs)",
                ),
                "modeling_groundinganything.py": (
                    "    def is_flash_attention_requested(config):", "    _supports_attention_backend = True",
                ),
                "modeling_groundinganything_vision.py": (
                    "nn.RMSNorm(hidden_dim, eps=1e-6)", '"sdpa": sdpa_attention', "def sdpa_attention(",
                ),
                "processing_groundinganything.py": (
                    "class GroundAnythingVLMProcessor(ProcessorMixin):", "def _get_num_multimodal_tokens(",
                    'text_inputs["mm_token_type_ids"] = mm_token_type_ids',
                ),
            }
            if all(marker in original for marker in prepared_markers[module]):
                return ast.dump(ast.parse(original))
            return ast.dump(ast.parse(transform(original)))

        @wraps(transform)
        def apply(source):
            try:
                parsed = ast.dump(ast.parse(source))
            except SyntaxError as exc:
                raise ValueError(f"invalid model source: {module}") from exc
            if parsed == canonical():
                return source
            result = transform(source)
            try:
                supported = ast.dump(ast.parse(result)) == canonical()
            except SyntaxError as exc:
                raise ValueError(f"unsupported model source: {module}") from exc
            if not supported:
                raise ValueError(f"unsupported model source: {module}")
            return result

        return apply
    return decorate


CONFIG_MODULE = "configuration_groundinganything.py"
KIMI_MODEL_MODULE = "modeling_groundinganything_vision.py"
MODEL_MODULE = "modeling_groundinganything.py"
PROCESSOR_MODULE = "processing_groundinganything.py"


@_accept_prepared_source(CONFIG_MODULE)
def _patched_config_source(source: str) -> str:
    import_line = "from huggingface_hub.dataclasses import strict"
    if import_line not in source:
        raise ValueError(f"{CONFIG_MODULE} does not contain the expected strict import")
    source = source.replace(import_line, "from dataclasses import dataclass", 1)
    transformers_import = "from transformers import CONFIG_MAPPING, AutoConfig"
    if transformers_import not in source:
        raise ValueError(f"{CONFIG_MODULE} does not contain the expected transformers import")
    source = source.replace(
        transformers_import,
        "from transformers.models.qwen3.configuration_qwen3 import Qwen3Config",
        1,
    )
    config_base_import = "from transformers.configuration_utils import PreTrainedConfig"
    if config_base_import not in source:
        raise ValueError(f"{CONFIG_MODULE} does not contain the expected config base import")
    source = source.replace(
        config_base_import,
        "try:\n"
        "    from transformers.configuration_utils import PreTrainedConfig\n"
        "except ImportError:\n"
        "    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig",
        1,
    )
    expected_classes = ("GroundAnythingVLMVisionConfig", "GroundAnythingVLMConfig")
    for class_name in expected_classes:
        original = f"@strict\nclass {class_name}"
        replacement = f"@dataclass(init=False)\nclass {class_name}"
        if original not in source:
            raise ValueError(f"cannot locate strict declaration for {class_name}")
        source = source.replace(original, replacement, 1)

    vision_marker = '    base_config_key = "vision_config"\n'
    vision_init = (
        "\n    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
    )
    if vision_marker not in source:
        raise ValueError("cannot locate vision config fields")
    source = source.replace(vision_marker, vision_marker + vision_init, 1)

    config_marker = '    keys_to_ignore_at_inference = ["past_key_values"]\n'
    config_init = """

    def __init__(self, **kwargs):
        self.text_config = kwargs.pop("text_config", None)
        self.vision_config = kwargs.pop("vision_config", None)
        self.image_token_id = kwargs.pop("image_token_id", 151655)
        self.video_token_id = kwargs.pop("video_token_id", 151656)
        self.vision_start_token_id = kwargs.pop("vision_start_token_id", 151652)
        self.vision_end_token_id = kwargs.pop("vision_end_token_id", 151653)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        if isinstance(self.vision_config, dict):
            self.vision_config = GroundAnythingVLMVisionConfig(**self.vision_config)
        if isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "qwen3")
            text_config_cls = CONFIG_MAPPING[text_model_type]
            self.sub_configs["text_config"] = text_config_cls
            self.text_config = text_config_cls(**self.text_config)
        # Transformers 5.x makes PreTrainedConfig a dataclass whose generated
        # __init__ dispatches to self.__post_init__.  GroundAnythingVLMConfig overrides
        # __post_init__, so super().__init__ would skip the base initializer and
        # leave fields such as _output_attentions unset.  Invoke the base hook
        # directly on 5.x while retaining the 4.x path used by training.
        if hasattr(PreTrainedConfig, "__post_init__"):
            PreTrainedConfig.__post_init__(self, **kwargs)
        else:
            super().__init__(**kwargs)
        self.__post_init__()
"""
    if config_marker not in source:
        raise ValueError("cannot locate top-level config fields")
    source = source.replace(config_marker, config_marker + config_init, 1)
    source = source.replace(
        '    sub_configs = {"vision_config": GroundAnythingVLMVisionConfig, "text_config": AutoConfig}\n',
        '    sub_configs = {"vision_config": GroundAnythingVLMVisionConfig, "text_config": Qwen3Config}\n',
        1,
    )
    source = source.replace(
        '        self.text_config = kwargs.pop("text_config", None)\n',
        '        self.text_config = kwargs.pop("text_config", None)\n',
        1,
    )
    source = source.replace(
        "            text_config_cls = CONFIG_MAPPING[text_model_type]\n",
        '            if text_model_type != "qwen3":\n'
        '                raise ValueError(f"unsupported text model type: {text_model_type}")\n'
        "            text_config_cls = Qwen3Config\n",
    )
    source = source.replace(
        '            text_config_cls = CONFIG_MAPPING["qwen3"]\n',
        "            text_config_cls = Qwen3Config\n",
    )
    source = source.replace(
        "        super().__post_init__(**kwargs)\n",
        "        del kwargs\n",
        1,
    )
    if "CONFIG_MAPPING[" in source or "AutoConfig}" in source:
        raise ValueError("lazy Transformers config mapping remains in patched source")
    return source


@_accept_prepared_source(MODEL_MODULE)
def _patched_model_source(source: str) -> str:
    flash_import = "from transformers.utils.generic import is_flash_attention_requested"
    if flash_import not in source:
        raise ValueError(f"cannot locate flash-attention helper import in {MODEL_MODULE}")
    source = source.replace(
        flash_import,
        "try:\n"
        "    from transformers.utils.generic import is_flash_attention_requested\n"
        "except ImportError:\n"
        "    def is_flash_attention_requested(config):\n"
        "        return getattr(config, \"_attn_implementation\", None) == \"flash_attention_2\"",
        1,
    )
    marker = "class GroundAnythingVLMPreTrainedModel(PreTrainedModel):\n"
    replacement = marker + "    _supports_attention_backend = True\n"
    if marker not in source:
        raise ValueError(f"cannot locate {marker.strip()} declaration")
    return source.replace(marker, replacement, 1)


@_accept_prepared_source(KIMI_MODEL_MODULE)
def _patched_kimi_model_source(source: str) -> str:
    marker = "nn.RMSNorm(hidden_dim)"
    count = source.count(marker)
    if count != 3:
        raise ValueError(
            f"expected 3 implicit RMSNorm epsilon values in {KIMI_MODEL_MODULE}, found {count}"
        )
    source = source.replace(marker, "nn.RMSNorm(hidden_dim, eps=1e-6)")
    attention_map = '''VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "eager": eager_attention,
}
'''
    sdpa_support = '''def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    del k_cu_seqlens, kwargs
    outputs = []
    for index in range(1, len(q_cu_seqlens)):
        start = int(q_cu_seqlens[index - 1])
        end = int(q_cu_seqlens[index])
        query = q[start:end].transpose(0, 1)
        key = k[start:end].transpose(0, 1)
        value = v[start:end].transpose(0, 1)
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        outputs.append(output.transpose(0, 1))
    return torch.cat(outputs, dim=0).flatten(start_dim=-2)


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "eager": eager_attention,
    "sdpa": sdpa_attention,
}
'''
    if attention_map not in source:
        raise ValueError(f"cannot locate vision attention map in {KIMI_MODEL_MODULE}")
    return source.replace(attention_map, sdpa_support, 1)


@_accept_prepared_source(PROCESSOR_MODULE)
def _patched_processor_source(source: str) -> str:
    import_marker = "from transformers.feature_extraction_utils import BatchFeature\n"
    if import_marker not in source:
        raise ValueError(f"cannot locate BatchFeature import in {PROCESSOR_MODULE}")
    source = source.replace(
        import_marker,
        import_marker + "from transformers.processing_utils import ProcessorMixin\n",
        1,
    )
    class_marker = "class GroundAnythingVLMProcessor:\n"
    if class_marker not in source:
        raise ValueError(f"cannot locate GroundAnythingVLMProcessor in {PROCESSOR_MODULE}")
    source = source.replace(
        class_marker,
        "class GroundAnythingVLMProcessor(ProcessorMixin):\n",
        1,
    )
    original_init = """        del kwargs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)
"""
    patched_init = """        del kwargs
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template or getattr(tokenizer, "chat_template", None),
        )
"""
    if original_init not in source:
        raise ValueError(f"cannot locate GroundAnythingVLMProcessor.__init__ in {PROCESSOR_MODULE}")
    source = source.replace(original_init, patched_init, 1)

    register_marker = "    @classmethod\n    def register_for_auto_class"
    protocol_methods = '''    @property
    def image_token(self):
        return "<|image_pad|>"

    @property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        del kwargs
        num_image_tokens = []
        num_image_patches = []
        for height, width in image_sizes or ():
            image_stub = type("ImageSize", (), {"size": (width, height)})()
            resize = self.image_processor.get_resize_config(
                {"type": "image", "image": image_stub}
            )
            tokens = int(resize["num_tokens"])
            num_image_tokens.append(tokens)
            num_image_patches.append(tokens * self.image_processor.merge_size**2)
        return {
            "num_image_tokens": num_image_tokens,
            "num_image_patches": num_image_patches,
        }

'''
    if register_marker not in source:
        raise ValueError(f"cannot locate auto-class registration in {PROCESSOR_MODULE}")
    source = source.replace(register_marker, protocol_methods + register_marker, 1)

    call_start = source.find("    def __call__(\n")
    call_end = source.find("    def batch_decode", call_start)
    if call_start < 0 or call_end < 0:
        raise ValueError(f"cannot locate GroundAnythingVLMProcessor.__call__ in {PROCESSOR_MODULE}")
    patched_call = '''    def __call__(
        self,
        text=None,
        images=None,
        return_tensors="pt",
        padding=False,
        **kwargs,
    ):
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", False)
        if isinstance(text, str):
            text = [text]

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                return_tensors=return_tensors,
            )
            text = list(text)
            image_index = 0
            merge_length = self.image_processor.merge_size**2
            for batch_index, prompt in enumerate(text):
                while self.image_token in prompt:
                    grid = image_inputs["image_grid_thw"][image_index]
                    num_tokens = int(grid.prod().item()) // merge_length
                    prompt = prompt.replace(
                        self.image_token, "<|image_placeholder|>" * num_tokens, 1
                    )
                    image_index += 1
                text[batch_index] = prompt.replace(
                    "<|image_placeholder|>", self.image_token
                )
            if image_index != len(image_inputs["image_grid_thw"]):
                raise ValueError(
                    "number of image placeholders does not match image inputs"
                )

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )
        if return_mm_token_type_ids:
            input_ids = text_inputs["input_ids"]
            if hasattr(input_ids, "new_zeros"):
                mm_token_type_ids = input_ids.new_zeros(input_ids.shape)
                mm_token_type_ids[input_ids == self.image_token_id] = 1
            else:
                mm_token_type_ids = [
                    [int(token == self.image_token_id) for token in row]
                    for row in input_ids
                ]
            text_inputs["mm_token_type_ids"] = mm_token_type_ids

        return BatchFeature(data={**text_inputs, **image_inputs})

'''
    return source[:call_start] + patched_call + source[call_end:]
