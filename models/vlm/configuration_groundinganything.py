from dataclasses import dataclass

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
try:
    from transformers.configuration_utils import PreTrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig


@dataclass(init=False)
class GroundAnythingVLMVisionConfig(PreTrainedConfig):
    model_type = "groundinganything_vision"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    layer_norm_type: str = "layer_norm"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    use_head: bool = False
    out_hidden_size: int = 1024
    spatial_merge_size: int = 2
    tokens_per_second: int = 1
    frame_windows_size: int = 4
    use_patch_position_encoding: bool = False
    patch_position_encoding_type: str = "absolute"
    max_position_embeddings: int = 8192
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_type: str = "divided_fixed"
    merge_kernel_size: list | None = None
    merge_type: str = "sd2_tpool"
    qkv_hidden_size: int = 1536
    norm_type: str = "rmsnorm"
    attn_bias: bool = False
    patch_embed_proj_bias: bool = False
    mlp_type: str = "mlp2"
    linear_bias: bool = False
    activation_func: str = "gelu_pytorch_tanh"
    pos_emb_interpolation_mode: str = "bilinear"
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-5


@dataclass(init=False)
class GroundAnythingVLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroundAnythingVLMBaseModel`]. It is used to instantiate a
    GroundAnythingVLMBaseModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a GroundAnything-VLM configuration.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `GroundAnythingVLMVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.
    """

    model_type = "groundinganything_vlm"
    # `text_config` is resolved dynamically based on its `model_type` (defaults to `qwen3`),
    # so we use `AutoConfig` here as a placeholder; `__post_init__` swaps it for the
    # concrete config class via `CONFIG_MAPPING`.
    sub_configs = {"vision_config": GroundAnythingVLMVisionConfig, "text_config": Qwen3Config}
    keys_to_ignore_at_inference = ["past_key_values"]


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
            if text_model_type != "qwen3":
                raise ValueError(f"unsupported text model type: {text_model_type}")
            text_config_cls = Qwen3Config
            self.sub_configs["text_config"] = text_config_cls
            self.text_config = text_config_cls(**self.text_config)
        # Transformers 5.x uses a generated dataclass __init__ that dispatches
        # to this subclass' __post_init__. Call the base hook explicitly so its
        # private attention/config state is initialized. Keep 4.x compatible.
        if hasattr(PreTrainedConfig, "__post_init__"):
            PreTrainedConfig.__post_init__(self, **kwargs)
        else:
            super().__init__(**kwargs)
        self.__post_init__()

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    # Generation-related token ids are mirrored from `text_config` in `__post_init__`
    # so downstream tools (e.g. `generate`, vLLM) that read them at the top level keep working.
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        # Resolve vision_config
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # Resolve text_config dynamically via CONFIG_MAPPING (defaults to qwen3)
        if isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "qwen3")
            self.text_config["model_type"] = text_model_type
            if text_model_type != "qwen3":
                raise ValueError(f"unsupported text model type: {text_model_type}")
            text_config_cls = Qwen3Config
            self.sub_configs["text_config"] = text_config_cls
            self.text_config = text_config_cls(**self.text_config)
        elif self.text_config is None:
            text_config_cls = Qwen3Config
            self.sub_configs["text_config"] = text_config_cls
            self.text_config = text_config_cls()

        # Mirror generation-related token ids from text_config to the top level so
        # downstream tools (e.g. `generate`, chat templates, vLLM) that read them
        # from the top-level config keep working.
        for tok_key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            text_val = getattr(self.text_config, tok_key, None)
            if text_val is not None and getattr(self, tok_key, None) is None:
                setattr(self, tok_key, text_val)

        del kwargs


__all__ = ["GroundAnythingVLMConfig", "GroundAnythingVLMVisionConfig", "GroundAnythingConfig"]


class GroundAnythingConfig(GroundAnythingVLMConfig):
    """DLM release identity for the shared Qwen3 VLM configuration."""

    model_type = "groundinganything"
