"""Processor glue for GroundAnything-VLM with Kimi-K3 MoonViT preprocessing."""

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from .media_utils import MediaInput
from .image_processing_groundinganything import GroundAnythingVLMImageProcessor


class GroundAnythingVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        del kwargs
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template or getattr(tokenizer, "chat_template", None),
        )

    @property
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

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        cls._auto_class = auto_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        import os
        from transformers import AutoTokenizer

        kwargs.pop("_from_auto", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("code_revision", None)
        with open(os.path.join(pretrained_model_name_or_path, "preprocessor_config.json"), encoding="utf-8") as f:
            processor_config = json.load(f)
        image_processor = GroundAnythingVLMImageProcessor(
            media_proc_cfg=processor_config["media_proc_cfg"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def apply_chat_template(self, messages, **kwargs):
        if self.chat_template and "chat_template" not in kwargs:
            kwargs["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(
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

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["GroundAnythingVLMProcessor"]


class GroundAnythingProcessor(GroundAnythingVLMProcessor):
    """DLM release processor identity."""
