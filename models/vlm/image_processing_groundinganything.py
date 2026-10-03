"""Image processor class for Kimi-K3.
"""

import json
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import (BaseImageProcessor,
                                                 BatchFeature)
from transformers.utils import TensorType

from .media_utils import (MediaInput, TransparentBgConfig, _to_tensor,
                          ensure_media_type, image_to_np, navit_patchify,
                          navit_resize_image, normalize)


class GroundAnythingVLMImageProcessor(BaseImageProcessor):
    model_type = "groundinganything_backbone"

    def __init__(
        self,
        media_proc_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.media_proc_cfg = dict(media_proc_cfg)
        self.patch_size = int(self.media_proc_cfg.get("patch_size", 14))
        self.merge_size = int(self.media_proc_cfg.get("merge_kernel_size", 2))
        max_output_tokens = os.getenv("IMAGE_MAX_TOKEN_NUM")
        if max_output_tokens:
            max_input_patches = int(max_output_tokens) * self.merge_size**2
            self.media_proc_cfg["in_patch_limit"] = min(
                int(self.media_proc_cfg["in_patch_limit"]), max_input_patches
            )
        self.max_output_tokens = (
            int(self.media_proc_cfg["in_patch_limit"]) // self.merge_size**2
        )

    @property
    def _transparent_bg_config(self) -> Optional[TransparentBgConfig]:
        cfg = self.media_proc_cfg.get("transparent_bg_config")
        if cfg is None:
            return None
        if isinstance(cfg, TransparentBgConfig):
            return cfg
        return TransparentBgConfig(**cfg)

    @property
    def _transparent_bg_fill_stage(self) -> str:
        return self.media_proc_cfg.get("transparent_bg_fill_stage",
                                       "before_resize")

    def media_tokens_calculator(self, media: MediaInput):
        media = ensure_media_type(
            media,
            transparent_bg_config=self._transparent_bg_config,
            transparent_bg_fill_stage=self._transparent_bg_fill_stage,
        )
        ret = self.get_resize_config(media)
        return ret['num_tokens']

    @classmethod
    def make_image_prompt(cls, width: int, height: int) -> str:
        """Build the K3 image placeholder with resolution info."""
        return (f"<|media_begin|>image {width}x{height}"
                f"<|media_content|><|media_pad|><|media_end|>")

    def get_resize_config(self, media_input: MediaInput) -> dict:
        if media_input['type'] == 'image':
            w, h = media_input['image'].size
            input_patch_limit = int(self.media_proc_cfg['in_patch_limit'])
            while True:
                ret = navit_resize_image(
                    w, h, self.media_proc_cfg['patch_size'],
                    self.media_proc_cfg['merge_kernel_size'],
                    input_patch_limit,
                    self.media_proc_cfg['patch_limit_on_one_side'],
                    self.media_proc_cfg['fixed_output_tokens'])
                if ret['num_tokens'] <= self.max_output_tokens:
                    return ret
                reduced_limit = int(
                    input_patch_limit * self.max_output_tokens /
                    ret['num_tokens'])
                input_patch_limit = min(input_patch_limit - 1,
                                        reduced_limit)
        else:
            raise ValueError("Unsupported type: {}".format(
                media_input['type']))

    def resize_image(self, image: Image.Image, new_width: int, new_height: int,
                     pad_width: int, pad_height: int) -> np.ndarray:
        image_np = image_to_np(
            image,
            (new_width, new_height),
            "resize",
            transparent_bg_config=self._transparent_bg_config,
            transparent_bg_fill_stage=self._transparent_bg_fill_stage,
        )
        image_np = np.pad(
            image_np,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return image_np

    def preprocess(
        self,
        medias: Optional[list[MediaInput]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        images=None,
        do_resize: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess a atom vision input (images) into model-ready tensors.

        Args:
            medias: List of MediaInput.
            return_tensors: Desired output format ('pt', 'np', 'tf', or None).

        Returns:
            BatchFeature containing 'pixel_values' and 'grid_thws' tensors.
        """
        del do_resize, kwargs
        if medias is None:
            medias = images
        if medias is None:
            medias = []
        if not isinstance(medias, list):
            medias = [medias]
        medias = [item if isinstance(item, dict) else {"type": "image", "image": item} for item in medias]
        if medias:
            pixel_values = []
            for item in medias:
                item = ensure_media_type(
                    item,
                    transparent_bg_config=self._transparent_bg_config,
                    transparent_bg_fill_stage=self._transparent_bg_fill_stage,
                )
                resize_config = self.get_resize_config(item)
                new_width, new_height, pad_width, pad_height = resize_config[
                    'new_width'], resize_config['new_height'], resize_config[
                        'pad_width'], resize_config['pad_height']
                if item['type'] == 'image':
                    image = item['image']
                    image_np = self.resize_image(image, new_width, new_height,
                                                 pad_width, pad_height)
                    pixel_values.append(np.expand_dims(image_np, axis=0))
                else:
                    raise ValueError("Unsupported type: {}".format(
                        item['type']))
            normalized_pixel_values = []
            image_std_inv = 1.0 / np.array(self.media_proc_cfg['image_std'])
            image_mean = np.array(self.media_proc_cfg['image_mean'])
            for pixels in pixel_values:
                pixels = normalize(pixels, image_mean, image_std_inv)
                pixels_and_thw = navit_patchify(
                    pixels,
                    self.media_proc_cfg['patch_size'],
                )
                normalized_pixel_values.append(pixels_and_thw)

            pixel_values = torch.cat([
                _to_tensor(pixel_value['pixel_values'])
                for pixel_value in normalized_pixel_values
            ])
            grid_thws = torch.cat([
                _to_tensor(pixel_value['grid_thw'],
                           dtype=torch.int64).unsqueeze(0)
                for pixel_value in normalized_pixel_values
            ])

            data = {
                'pixel_values': pixel_values,
                'image_grid_thw': grid_thws,
            }

        else:
            data = {}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __repr__(self):
        return f"GroundAnythingVLMImageProcessor(media_proc_cfg={self.media_proc_cfg})"

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["media_proc_cfg"] = self.media_proc_cfg
        if "media_processor" in output:
            del output["media_processor"]
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        media_proc_cfg = config.pop("media_proc_cfg", {})
        return cls(media_proc_cfg=media_proc_cfg, **config, **kwargs)

    def to_json_string(self):
        dictionary = self.to_dict()
        for key, value in dictionary.items():
            if hasattr(value, 'tolist'):
                dictionary[key] = value.tolist()
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"
