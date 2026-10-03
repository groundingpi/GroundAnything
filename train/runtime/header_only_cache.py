"""Header-only Qwen3.5 cached-dataset export support.

Swift's regular VLM cache exporter decodes and resizes every image only to
derive ``image_grid_thw`` and the resulting sequence length.  GAM inputs are
image paths (ordinary files or ``.tar@offset=...`` virtual members), so the
same grid can be derived from width/height read from the image header.  This
module deliberately patches only the fixed Docker Swift method whose source
hash is recorded below; an image/video/template change fails closed.
"""

from __future__ import annotations

import hashlib
import inspect
from io import BytesIO
import os
from pathlib import Path
import stat
from typing import Any, Callable, Iterable, Tuple

from PIL import Image, ImageFile

from .finevision_text_media import (
    EXTREME_ASPECT_MAX_RATIO,
    PLACEHOLDER_ABSOLUTE,
    record_extreme_aspect_image,
    rewrite_export_images,
    validate_placeholder_asset,
)
from .tar_uri import parse_tar_uri


ENABLE_ENV = "GAM_FAST_CACHE_HEADER_ONLY"
EXPECTED_QWEN3VL_ENCODE_SHA256 = (
    "3995e0b0783a4722358e8ec2e146d353bf1585664d3e3096833681848042319d"
)
EXPECTED_TEMPLATE_PREPROCESS_INPUTS_SHA256 = (
    "4df1bdf0aaebcf1f37fa47b0859927868e2d2e058d44cc9894fb4e532ac928c3"
)
INITIAL_HEADER_BYTES = 4096
# A valid JPEG can legally place many APP/EXIF segments before its SOF marker.
# The General Data V1 LLaVA inventory contains a strictly decodable 5,159,426
# byte JPEG whose 1200x800 SOF is beyond 4 MiB.  Keep a hard bound, but make it
# large enough to preserve such images instead of letting Swift delete rows.
MAX_HEADER_BYTES = 64 << 20


class HeaderOnlyCacheError(RuntimeError):
    """Raised when header-only export cannot prove an exact length."""


def _positive_size(size: Iterable[Any], label: str) -> Tuple[int, int]:
    values = tuple(size)
    if len(values) != 2:
        raise HeaderOnlyCacheError(f"{label} image size is not width/height: {values!r}")
    width, height = values
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise HeaderOnlyCacheError(f"{label} image size is invalid: {values!r}")
    return width, height


def _parse_header(
    read_at: Callable[[int, int], bytes], total_bytes: int, label: str
) -> Tuple[int, int, int]:
    """Return width, height and bytes consumed without decoding pixels."""

    if type(total_bytes) is not int or total_bytes <= 0:
        raise HeaderOnlyCacheError(f"{label} has invalid byte size: {total_bytes!r}")
    parser = ImageFile.Parser()
    offset = 0
    chunk_size = INITIAL_HEADER_BYTES
    limit = min(total_bytes, MAX_HEADER_BYTES)
    while offset < limit:
        requested = min(chunk_size, limit - offset)
        chunk = read_at(offset, requested)
        if not chunk:
            break
        parser.feed(chunk)
        offset += len(chunk)
        if parser.image is not None:
            width, height = _positive_size(parser.image.size, label)
            return width, height, offset
        if len(chunk) != requested:
            break
        chunk_size = min(chunk_size * 2, 256 << 10)
    raise HeaderOnlyCacheError(
        f"cannot determine image size from at most {limit} header bytes: {label}"
    )


def _regular_file_size(path: Path) -> Tuple[int, int, int]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise HeaderOnlyCacheError(f"image path is not a regular file: {path}")
        return _parse_header(
            lambda offset, size: os.pread(descriptor, size, offset),
            info.st_size,
            str(path),
        )
    finally:
        os.close(descriptor)


def _tar_member_size(value: str) -> Tuple[int, int, int]:
    parsed = parse_tar_uri(value)
    if parsed is None:
        raise HeaderOnlyCacheError(f"not a GAM tar image URI: {value!r}")
    tar_path, member_offset, member_bytes, member = parsed
    descriptor = os.open(tar_path, os.O_RDONLY)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or member_offset + member_bytes > info.st_size:
            raise HeaderOnlyCacheError(
                f"tar member range is invalid: {tar_path} offset={member_offset} "
                f"size={member_bytes} file_size={info.st_size}"
            )
        return _parse_header(
            lambda offset, size: os.pread(
                descriptor, size, member_offset + offset
            ),
            member_bytes,
            f"{tar_path}:{member}",
        )
    finally:
        os.close(descriptor)


def _bytes_size(payload: bytes, label: str) -> Tuple[int, int, int]:
    return _parse_header(
        lambda offset, size: payload[offset : offset + size], len(payload), label
    )


def image_header_size(image: Any) -> Tuple[int, int, int]:
    """Return ``(width, height, bytes_read)`` for one supported image value."""

    if isinstance(image, Image.Image):
        width, height = _positive_size(image.size, "PIL")
        return width, height, 0
    try:
        import torch

        if isinstance(image, torch.Tensor):
            shape = tuple(int(value) for value in image.shape)
            if len(shape) == 2:
                height, width = shape
            elif len(shape) == 3 and shape[0] in (1, 3, 4):
                _, height, width = shape
            elif len(shape) == 3:
                height, width, _ = shape
            else:
                raise HeaderOnlyCacheError(f"unsupported image tensor shape: {shape}")
            width, height = _positive_size((width, height), "tensor")
            return width, height, 0
    except ImportError:
        pass
    if isinstance(image, dict):
        path = image.get("path")
        payload = image.get("bytes")
        usable_path = False
        if isinstance(path, str) and path:
            local_path = path[7:] if path.startswith("file://") else path
            usable_path = (
                parse_tar_uri(path) is not None or Path(local_path).is_file()
            )
        if usable_path:
            image = path
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            image = bytes(payload)
        elif isinstance(path, str) and path:
            # Preserve the original value so the eventual error names the
            # missing path instead of degrading into an unhelpful type error.
            image = path
        else:
            raise HeaderOnlyCacheError("image mapping has neither path nor bytes")
    if isinstance(image, (bytearray, memoryview)):
        image = bytes(image)
    if isinstance(image, bytes):
        return _bytes_size(image, "in-memory image")
    if not isinstance(image, str) or not image.strip():
        raise HeaderOnlyCacheError(f"unsupported image value: {type(image).__name__}")
    value = image.strip()
    if parse_tar_uri(value) is not None:
        return _tar_member_size(value)
    if value.startswith("file://"):
        value = value[7:]
    if value.startswith(("http://", "https://", "data:")):
        raise HeaderOnlyCacheError(
            "formal GAM header-only export forbids remote/base64 images"
        )
    return _regular_file_size(Path(value).expanduser())


def _method_sha256(method: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(method).encode("utf-8")).hexdigest()


def install_header_only_cache_patch() -> None:
    """Install a fail-closed Qwen3/3.5 patch for the fixed export process."""

    if os.environ.get(ENABLE_ENV) != "1":
        raise HeaderOnlyCacheError(f"{ENABLE_ENV}=1 is required")
    validate_placeholder_asset()

    import torch
    from qwen_vl_utils import vision_process
    from swift.template.base import Template
    from swift.template.templates.qwen import Qwen3VLTemplate
    from swift.template.templates import qwen as qwen_module

    current_encode = Qwen3VLTemplate._encode
    if getattr(current_encode, "_gam_header_only_cache_patch", False):
        return
    actual_sha256 = _method_sha256(current_encode)
    if actual_sha256 != EXPECTED_QWEN3VL_ENCODE_SHA256:
        raise HeaderOnlyCacheError(
            "Docker Swift Qwen3VLTemplate._encode drifted: "
            f"expected={EXPECTED_QWEN3VL_ENCODE_SHA256} actual={actual_sha256}"
        )
    original_replace_tag = Qwen3VLTemplate.replace_tag
    current_preprocess_inputs = Template._preprocess_inputs
    actual_preprocess_sha256 = _method_sha256(current_preprocess_inputs)
    if (
        actual_preprocess_sha256
        != EXPECTED_TEMPLATE_PREPROCESS_INPUTS_SHA256
    ):
        raise HeaderOnlyCacheError(
            "Docker Swift Template._preprocess_inputs drifted: "
            f"expected={EXPECTED_TEMPLATE_PREPROCESS_INPUTS_SHA256} "
            f"actual={actual_preprocess_sha256}"
        )

    def header_only_replace_tag(self, media_type, index, inputs):
        if media_type == "image":
            # Preserve the path/ref and never call qwen_vl_utils.fetch_image.
            # The patched _encode probes the header exactly once per image.
            return ["<|vision_start|><|image_pad|><|vision_end|>"]
        return original_replace_tag(self, media_type, index, inputs)

    def header_only_preprocess_inputs(self, inputs):
        """Mirror the fixed Swift preprocessor without loading image pixels."""

        if inputs.videos:
            raise HeaderOnlyCacheError(
                "formal GAM fast cache does not support video"
            )
        if inputs.audios:
            raise HeaderOnlyCacheError(
                "formal GAM fast cache does not support audio"
            )
        if inputs.objects:
            raise HeaderOnlyCacheError(
                "header-only cache requires coordinates inline in messages"
            )
        if self.max_pixels is not None:
            raise HeaderOnlyCacheError(
                "header-only cache does not support Template.max_pixels"
            )
        if inputs.images:
            rewritten, _ = rewrite_export_images(inputs.images)
            inputs.images[:] = rewritten
        self._preprocess_function_call(inputs)
        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)
        # Do not call Template._load_image.  The patched _encode consumes the
        # untouched str / datasets Image mapping with image_header_size().
        if inputs.is_multimodal:
            self._add_default_tags(inputs)

    def header_only_encode(self, inputs):
        if inputs.videos:
            raise HeaderOnlyCacheError("formal GAM fast cache does not support video")
        encoded = Template._encode(self, inputs)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale")
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor

            def scalar(value, label):
                if isinstance(value, (tuple, list)):
                    if not value or any(item != value[0] for item in value):
                        raise HeaderOnlyCacheError(
                            f"non-uniform {label} is unsupported: {value!r}"
                        )
                    value = value[0]
                result = int(value)
                if result <= 0:
                    raise HeaderOnlyCacheError(f"invalid {label}: {value!r}")
                return result

            patch_size = scalar(getattr(image_processor, "patch_size", 16), "patch_size")
            merge_size = scalar(getattr(image_processor, "merge_size", 2), "merge_size")
            factor = patch_size * merge_size
            min_pixels = int(vision_process.IMAGE_MIN_TOKEN_NUM) * factor**2
            max_pixels = int(vision_process.IMAGE_MAX_TOKEN_NUM) * factor**2
            grids = []
            for image_index, image in enumerate(images):
                width, height, _ = image_header_size(image)
                ratio = max(width, height) / min(width, height)
                if ratio > EXTREME_ASPECT_MAX_RATIO:
                    record_extreme_aspect_image(
                        image, image_index, width, height
                    )
                    width, height, _ = image_header_size(PLACEHOLDER_ABSOLUTE)
                try:
                    resized_height, resized_width = vision_process.smart_resize(
                        height,
                        width,
                        factor=factor,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                except ValueError as exc:
                    raise HeaderOnlyCacheError(
                        "Qwen smart_resize rejected image: "
                        f"index={image_index} width={width} height={height} "
                        f"image={image!r}: {exc}"
                    ) from exc
                grids.append(
                    [1, resized_height // patch_size, resized_width // patch_size]
                )
            image_grid_thw = torch.tensor(grids, dtype=torch.long)
            indices = qwen_module.findall(input_ids, self.image_token_id)
            if len(indices) != len(images):
                raise HeaderOnlyCacheError(
                    f"image placeholder mismatch: tokens={len(indices)} images={len(images)}"
                )

            def image_tokens(index):
                token_length = int(image_grid_thw[index].prod().item()) // (
                    merge_size**2
                )
                return [self.image_token_id] * token_length

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, indices, image_tokens
            )
            encoded["image_grid_thw"] = image_grid_thw
        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        return encoded

    header_only_replace_tag._gam_header_only_cache_patch = True
    header_only_preprocess_inputs._gam_header_only_cache_patch = True
    header_only_preprocess_inputs._gam_original_sha256 = (
        actual_preprocess_sha256
    )
    header_only_encode._gam_header_only_cache_patch = True
    header_only_encode._gam_original_sha256 = actual_sha256
    Qwen3VLTemplate.replace_tag = header_only_replace_tag
    Qwen3VLTemplate._preprocess_inputs = header_only_preprocess_inputs
    Qwen3VLTemplate._encode = header_only_encode
