"""Audited media repairs used only during cached-dataset export.

FineVision parts 67--96 contain text corpora represented as VLM rows.  Their
single ``images`` entry is a synthetic parquet locator rather than an image.
Keeping that locator makes both real training and strict cache export fail.
This module maps only that exact locator family to one pinned neutral image;
messages and row membership remain byte-for-byte unchanged.

Qwen additionally rejects images whose absolute aspect ratio is greater than
200.  The header-only encoder records every such image in a per-row context;
the pinned AddLengthPreprocessor wrapper then replaces only those exact image
entries with the same neutral image and emits a temporary JSON audit column.
The cache normalizer validates and removes that column before publication.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, List, Mapping, Sequence, Tuple


FINEVISION_SOURCE_ID = "finevision_all_cache_tar"
PLACEHOLDER_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "finevision_text_placeholder_32x32.png"
)
PLACEHOLDER_ABSOLUTE = str(PLACEHOLDER_PATH.resolve())
PLACEHOLDER_SHA256 = (
    "8426a28ecc2706c999bce2abd6789b8ec5759f59aa11ad174f1f0c24c7ea66dd"
)
POLICY_NAME = "finevision_text_parquet_locator_to_pinned_neutral_png"
EXTREME_ASPECT_POLICY_NAME = (
    "qwen_absolute_aspect_ratio_gt_200_to_pinned_neutral_png"
)
EXTREME_ASPECT_MAX_RATIO = 200
MEDIA_REPAIR_COLUMN = "gam_media_repair"
MEDIA_REPAIR_SCHEMA_VERSION = 1
EXPECTED_ADD_LENGTH_PREPROCESS_SHA256 = (
    "6bcb9b6719aaab10617da8e67e648aca1575573c21f4e3edf64f7a067e683d16"
)
_FINEVISION_TEXT_LOCATOR = re.compile(
    r"^data/train/"
    r"FineVision/text_[^/]+/[^/@]+\.parquet"
    r"@row_idx=[0-9]+&img_idx=[0-9]+$"
)
_FINEVISION_TEXT_PREFIX = (
    "data/train/"
    "FineVision/text_"
)


class FineVisionTextMediaError(RuntimeError):
    """Raised when the narrow media-repair contract is violated."""


_ACTIVE_EXTREME_ASPECT_REPAIRS: ContextVar[List[dict] | None] = ContextVar(
    "gam_active_extreme_aspect_repairs", default=None
)


def _image_path(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        path = value.get("path")
        payload = value.get("bytes")
        if isinstance(path, str) and path and payload in (None, b""):
            return path
    raise FineVisionTextMediaError(
        "extreme-aspect repair requires a path-backed image without inline bytes"
    )


def begin_extreme_aspect_row() -> Token:
    """Start one fail-closed per-row repair context."""

    if _ACTIVE_EXTREME_ASPECT_REPAIRS.get() is not None:
        raise FineVisionTextMediaError("nested extreme-aspect row context")
    return _ACTIVE_EXTREME_ASPECT_REPAIRS.set([])


def record_extreme_aspect_image(
    image: Any, image_index: int, width: int, height: int
) -> None:
    """Record one image proven incompatible with Qwen smart_resize."""

    repairs = _ACTIVE_EXTREME_ASPECT_REPAIRS.get()
    if repairs is None:
        raise FineVisionTextMediaError(
            "extreme-aspect image observed outside audited row context"
        )
    if (
        type(image_index) is not int
        or image_index < 0
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise FineVisionTextMediaError("invalid extreme-aspect repair dimensions")
    numerator, denominator = max(width, height), min(width, height)
    if numerator / denominator <= EXTREME_ASPECT_MAX_RATIO:
        raise FineVisionTextMediaError(
            "extreme-aspect repair recorded for an accepted image"
        )
    repairs.append({
        "image_index": image_index,
        "original_path": _image_path(image),
        "width": width,
        "height": height,
    })


def end_extreme_aspect_row(token: Token) -> List[dict]:
    repairs = _ACTIVE_EXTREME_ASPECT_REPAIRS.get()
    if repairs is None:
        raise FineVisionTextMediaError("missing extreme-aspect row context")
    result = list(repairs)
    _ACTIVE_EXTREME_ASPECT_REPAIRS.reset(token)
    return result


def _rewrite_extreme_aspect_images(
    images: Sequence[Any], repairs: Sequence[Mapping[str, Any]]
) -> List[Any]:
    result = list(images)
    seen = set()
    for repair in repairs:
        index = repair.get("image_index")
        if type(index) is not int or not 0 <= index < len(result) or index in seen:
            raise FineVisionTextMediaError(
                f"invalid or duplicate extreme-aspect image index: {index!r}"
            )
        if _image_path(result[index]) != repair.get("original_path"):
            raise FineVisionTextMediaError(
                "extreme-aspect repair original path drifted"
            )
        result[index] = PLACEHOLDER_ABSOLUTE
        seen.add(index)
    return result


def serialize_extreme_aspect_repairs(
    repairs: Sequence[Mapping[str, Any]]
) -> str:
    if not repairs:
        return ""
    return json.dumps(
        {
            "schema_version": MEDIA_REPAIR_SCHEMA_VERSION,
            "policy": EXTREME_ASPECT_POLICY_NAME,
            "repairs": list(repairs),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def is_finevision_text_locator(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_FINEVISION_TEXT_PREFIX)
        and _FINEVISION_TEXT_LOCATOR.fullmatch(value) is not None
    )


def validate_placeholder_asset() -> str:
    path = PLACEHOLDER_PATH.resolve(strict=True)
    if not path.is_file():
        raise FineVisionTextMediaError(f"placeholder is not a regular file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PLACEHOLDER_SHA256:
        raise FineVisionTextMediaError(
            "FineVision text placeholder SHA256 drifted: "
            f"expected={PLACEHOLDER_SHA256} actual={digest}"
        )
    return str(path)


def canonicalize_source_paths(
    source_id: str, image_paths: Sequence[str]
) -> Tuple[List[str], int]:
    """Return identity paths after the source-scoped audited repair."""

    if source_id != FINEVISION_SOURCE_ID:
        return image_paths if isinstance(image_paths, list) else list(image_paths), 0
    result: List[str] | None = None
    rewritten = 0
    for index, value in enumerate(image_paths):
        if is_finevision_text_locator(value):
            if result is None:
                result = list(image_paths[:index])
            result.append(PLACEHOLDER_ABSOLUTE)
            rewritten += 1
        elif result is not None:
            result.append(value)
    if result is None:
        return image_paths if isinstance(image_paths, list) else list(image_paths), 0
    return result, rewritten


def rewrite_export_images(images: Sequence[Any]) -> Tuple[List[Any], int]:
    """Rewrite Swift input images without touching messages or row count."""

    result: List[Any] | None = None
    rewritten = 0
    for index, value in enumerate(images):
        if is_finevision_text_locator(value):
            if result is None:
                result = list(images[:index])
            result.append(PLACEHOLDER_ABSOLUTE)
            rewritten += 1
            continue
        if isinstance(value, Mapping) and is_finevision_text_locator(value.get("path")):
            payload = value.get("bytes")
            if payload not in (None, b""):
                raise FineVisionTextMediaError(
                    "synthetic FineVision text locator unexpectedly carries image bytes"
                )
            if result is None:
                result = list(images[:index])
            mapped = dict(value)
            mapped["path"] = PLACEHOLDER_ABSOLUTE
            mapped["bytes"] = None
            result.append(mapped)
            rewritten += 1
            continue
        if result is not None:
            result.append(value)
    if result is None:
        return images if isinstance(images, list) else list(images), 0
    return result, rewritten


def policy_report(rewritten_rows: int, rewritten_images: int) -> dict:
    return {
        "policy": POLICY_NAME,
        "source_id": FINEVISION_SOURCE_ID,
        "placeholder_path": PLACEHOLDER_ABSOLUTE,
        "placeholder_sha256": PLACEHOLDER_SHA256,
        "rewritten_rows": rewritten_rows,
        "rewritten_images": rewritten_images,
        "message_mutations": 0,
        "row_drops": 0,
    }


def install_finevision_text_media_patch() -> None:
    """Patch the pinned Swift row preprocessor so cache paths are usable."""

    validate_placeholder_asset()
    from swift.dataset.utils import AddLengthPreprocessor

    current = AddLengthPreprocessor.preprocess
    if getattr(current, "_gam_finevision_text_media_patch", False):
        return
    actual = hashlib.sha256(inspect.getsource(current).encode("utf-8")).hexdigest()
    if actual != EXPECTED_ADD_LENGTH_PREPROCESS_SHA256:
        raise FineVisionTextMediaError(
            "Docker Swift AddLengthPreprocessor.preprocess drifted: "
            f"expected={EXPECTED_ADD_LENGTH_PREPROCESS_SHA256} actual={actual}"
        )

    def audited_preprocess(self, row):
        images = row.get("images")
        if images:
            rewritten, count = rewrite_export_images(images)
            if count:
                row = dict(row)
                row["images"] = rewritten
        token = begin_extreme_aspect_row()
        try:
            result = current(self, row)
        except BaseException:
            end_extreme_aspect_row(token)
            raise
        repairs = end_extreme_aspect_row(token)
        if result is None:
            return None
        if repairs:
            result = dict(result)
            result["images"] = _rewrite_extreme_aspect_images(
                result.get("images") or [], repairs
            )
        # Always emit a string so every Arrow batch has one stable schema.
        # This column is mandatory in unpublished Swift output and is removed
        # by normalize_exported_cache before publication.
        result[MEDIA_REPAIR_COLUMN] = serialize_extreme_aspect_repairs(repairs)
        return result

    audited_preprocess._gam_finevision_text_media_patch = True
    audited_preprocess._gam_original_sha256 = actual
    AddLengthPreprocessor.preprocess = audited_preprocess
