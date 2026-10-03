"""Kimi-K3 image processor bridge for the external SGLang model."""

from __future__ import annotations

import re
from typing import Any, List, Union

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from .gam_qwen3 import FastDVLMForConditionalGeneration


class GAMQwen3ImageProcessor(BaseMultimodalProcessor):
    """Keep the original K3 patchify/resize path; only adapt SGLang's item API."""

    models = [FastDVLMForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, transport_mode, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, transport_mode, *args, **kwargs)
        self.hf_config = hf_config
        self.IM_START_TOKEN_ID = int(hf_config.vision_start_token_id)
        self.IM_END_TOKEN_ID = int(hf_config.vision_end_token_id)
        self.IM_TOKEN_ID = int(hf_config.image_token_id)
        self.vision_start_token_id = self.IM_START_TOKEN_ID
        self.vision_end_token_id = self.IM_END_TOKEN_ID
        # ``None`` is meaningful to SGLang's multimodal token builder: this
        # route is image-only and must not turn a missing video token into
        # token id ``-1`` (which would be converted through the tokenizer and
        # can accidentally match the last vocabulary row).
        raw_video_token_id = getattr(hf_config, "video_token_id", None)
        self.video_token_id = (
            None if raw_video_token_id is None else int(raw_video_token_id)
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.video_token_id,
        ).build(_processor)

    def _tokenize(self, text: str) -> list[int]:
        return self._processor.tokenizer.encode(text, add_special_tokens=True)

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        del videos, audios, kwargs
        if images:
            image_processor = self._processor.image_processor
            out = image_processor(images=images, return_tensors="pt")
            grid = out["image_grid_thw"]
            prompt_ids = self._tokenize(input_text)
            input_ids, _ = self.build_input_ids(prompt_ids, grid)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "pixel_values": out["pixel_values"],
                "image_grid_thw": grid,
            }
        return {
            "input_ids": torch.tensor(self._tokenize(input_text), dtype=torch.long).unsqueeze(0)
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        del args, kwargs
        if getattr(request_obj, "video_data", None):
            raise RuntimeError("GAM Qwen3 SGLang route does not support video inputs")
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=None,
            audio_data=None,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        # Plain Qwen3 uses 1-D RoPE.  Supplying no mRoPE fields makes the
        # scheduler use its ordinary position sequence instead of accidentally
        # applying Qwen-VL's 3-axis positions.
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.IM_TOKEN_ID,
            "video_token_id": self.video_token_id,
            "mrope_positions": None,
            "mrope_position_delta": None,
        }


EntryClass = GAMQwen3ImageProcessor
