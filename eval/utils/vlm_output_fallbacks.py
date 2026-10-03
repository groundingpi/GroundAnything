"""Conservative fallbacks for malformed-but-unambiguous VLM coordinates.

Every helper in this module is intended to run only after the canonical JSON
parser has failed.  It recovers numbers the model actually emitted; it never
guesses a missing coordinate or changes the evaluation prompt.
"""

from __future__ import annotations

import re
from typing import Iterable


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_LABEL_RE = re.compile(
    r"[\"'](?:label|category|name|text|text_content|transcription|word)[\"']"
    r"\s*:\s*[\"']([^\"']*)[\"']",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(
    r"[\"'](?P<field>bbox_2d|bbox|box_2d|box|point_2d|point|coordinate|coordinates)"
    r"[\"']\s*:\s*",
    re.IGNORECASE,
)
_OBJECT_RE = re.compile(r"<object>(.*?)</object>", re.IGNORECASE | re.DOTALL)
_AREA_RE = re.compile(r"<area>(.*?)</area>", re.IGNORECASE | re.DOTALL)
_COORD_PAIR_RE = re.compile(
    rf"[\[(]\s*({_NUMBER_RE.pattern})\s*,\s*({_NUMBER_RE.pattern})\s*[\])]"
)
_FLAT_BOX_RE = re.compile(
    rf"\[\s*({_NUMBER_RE.pattern})\s*,\s*({_NUMBER_RE.pattern})\s*,\s*"
    rf"({_NUMBER_RE.pattern})\s*,\s*({_NUMBER_RE.pattern})\s*\]"
)


def numbers(text: object) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(str(text or ""))]


def _field_segments(text: str) -> Iterable[tuple[str, str, str]]:
    matches = list(_FIELD_RE.finditer(text))
    for index, match in enumerate(matches):
        # Stop at the next coordinate field or object boundary.  Labels remain
        # inside the surrounding context and cannot contribute numeric values.
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        boundary = text.find("}", match.end(), end)
        if boundary >= 0:
            end = boundary
        context_start = max(text.rfind("{", 0, match.start()), 0)
        context_end = text.find("}", match.end())
        if context_end < 0:
            context_end = min(len(text), end + 512)
        yield match.group("field").lower(), text[match.end() : end], text[context_start:context_end]


def fallback_boxes(text: object) -> list[tuple[str, list[float]]]:
    """Recover ``<object>`` and malformed bbox fields in stable order."""

    payload = str(text or "")
    output: list[tuple[str, list[float]]] = []
    for match in _OBJECT_RE.finditer(payload):
        values = numbers(match.group(1))
        if len(values) >= 4:
            output.append(("object", values[:4]))
    if output:
        return output

    for field, segment, context in _field_segments(payload):
        if field not in {"bbox_2d", "bbox", "box_2d", "box"}:
            continue
        values = numbers(segment)
        if len(values) < 4:
            continue
        label_match = _LABEL_RE.search(context)
        label = label_match.group(1).strip() if label_match else "object"
        output.append((label or "object", values[:4]))
    return output


def fallback_points(
    text: object, *, allow_box_center: bool = False
) -> list[tuple[str, list[float]]]:
    """Recover ``<area>`` or malformed point fields.

    Formal point tasks call this with ``allow_box_center=False``: only an
    explicitly emitted two-value point is valid.  The optional conversion is
    retained solely for non-scoring diagnostics and must never be enabled by
    a formal point-task adapter.
    """

    payload = str(text or "")
    output: list[tuple[str, list[float]]] = []
    for match in _AREA_RE.finditer(payload):
        values = numbers(match.group(1))
        if len(values) >= 2:
            output.append(("", values[:2]))
    if output:
        return output

    for field, segment, context in _field_segments(payload):
        values = numbers(segment)
        is_box = field in {"bbox_2d", "bbox", "box_2d", "box"}
        if is_box and not allow_box_center:
            continue
        if len(values) >= 4:
            if not allow_box_center:
                continue
            point = [(values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0]
        elif len(values) >= 2 and not is_box:
            point = values[:2]
        else:
            continue
        label_match = _LABEL_RE.search(context)
        output.append((label_match.group(1).strip() if label_match else "", point))
    return output


def fallback_ocr_items(text: object) -> list[tuple[str, list[float]]]:
    """Recover text-labelled malformed bbox fields for OCR spotting."""

    output: list[tuple[str, list[float]]] = []
    for field, segment, context in _field_segments(str(text or "")):
        if field not in {"bbox_2d", "bbox", "box_2d", "box"}:
            continue
        values = numbers(segment)
        label_match = _LABEL_RE.search(context)
        if len(values) >= 4 and label_match and label_match.group(1).strip():
            output.append((label_match.group(1).strip(), values[:4]))
    return output


def fallback_unlabelled_boxes(text: object) -> list[list[float]]:
    """Recover coordinate-only boxes for box-only scoring.

    OCR transcription scoring must never invent missing text.  Boxonly tasks,
    however, legitimately ignore transcription, so two explicitly emitted
    coordinate pairs can be recovered as one xyxy box.  This helper is not
    used by text-aware OCR tasks.
    """

    payload = str(text or "")
    output: list[list[float]] = []
    pairs = [
        [float(match.group(1)), float(match.group(2))]
        for match in _COORD_PAIR_RE.finditer(payload)
    ]
    for index in range(0, len(pairs) - 1, 2):
        output.append([*pairs[index], *pairs[index + 1]])
    if output:
        return output
    for match in _FLAT_BOX_RE.finditer(payload):
        output.append([float(match.group(index)) for index in range(1, 5)])
    return output
