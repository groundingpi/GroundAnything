"""GAM evaluation prompt-mode bridge.

``VLM`` is the compatibility mode: every task keeps its historical prompt,
visual input, and parser.  ``GAM``, ``DLM`` and the checkpoint-provenance-only
``RLV2`` route share the special-token wire protocol:
prompts are built
by the single canonical registry in ``train/data/special_token_format.py`` and
outputs must use Qwen native object-ref/box wrappers with ``<0>`` ... ``<999>``
coordinate tokens.

This module intentionally contains only the small amount of evaluation-side
adaptation (mode selection, native-output extraction, and pixel/grid mapping).
It does not duplicate any prompt text.
"""

from __future__ import annotations

import os
from pathlib import Path
import importlib.util
import re
import sys
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple


VALID_EVAL_MODES = (
    "VLM",
    "GAM",
    "DLM",
    "RLV2",
    "REXOMNI",
    "LOCATEANYTHING",
    "GROUNDINGDINO",
)
EVAL_MODE_ENV = "GAM_EVAL_MODE"


def normalize_eval_mode(mode: str | None) -> str:
    """Return an uppercase validated mode; ``None`` means compatibility VLM."""

    raw = "VLM" if mode is None else str(mode).strip().upper()
    aliases = {
        "REX-OMNI": "REXOMNI",
        "REX_OMNI": "REXOMNI",
        "LOCATE-ANYTHING": "LOCATEANYTHING",
        "LOCATE_ANYTHING": "LOCATEANYTHING",
        "GROUNDING-DINO": "GROUNDINGDINO",
        "GROUNDING_DINO": "GROUNDINGDINO",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in VALID_EVAL_MODES:
        raise ValueError(
            f"invalid eval mode {mode!r}; expected one of {VALID_EVAL_MODES}"
        )
    return normalized


def get_eval_mode() -> str:
    """Resolve the mode at call time so tests and workers can set the env safely."""

    return normalize_eval_mode(os.environ.get(EVAL_MODE_ENV, "VLM"))


def is_gam_mode() -> bool:
    """GAM wire protocol, shared by AR-GAM and converted DLM decoding."""

    return get_eval_mode() in ("GAM", "DLM", "RLV2")


def is_rexomni_mode() -> bool:
    return get_eval_mode() == "REXOMNI"


def is_locateanything_mode() -> bool:
    return get_eval_mode() == "LOCATEANYTHING"


def is_groundingdino_mode() -> bool:
    """GroundingDINO adapter using VLM prompts and JSON spatial outputs."""

    return get_eval_mode() == "GROUNDINGDINO"


def is_native_spatial_mode() -> bool:
    """Whether the active mode returns a special-token spatial protocol."""

    # GroundingDINO is a model-family adapter, not a special-token wire
    # protocol.  It deliberately reuses the validated VLM prompts/parsers and
    # returns ordinary JSON coordinates through a local OpenAI-compatible
    # bridge.  Keep it out of every native-token branch.
    return get_eval_mode() in ("GAM", "DLM", "RLV2", "REXOMNI", "LOCATEANYTHING")


# Import the canonical contract from train without depending on the
# caller's cwd or on code outside GAM.
_GAM_ROOT = Path(__file__).resolve().parents[2]
_FORMAT_DIR = _GAM_ROOT / "train" / "data"
_FORMAT_PATH = _FORMAT_DIR / "special_token_format.py"
_FORMAT_MODULE_NAME = "_gam_eval_special_token_format"
if _FORMAT_MODULE_NAME in sys.modules:
    _format = sys.modules[_FORMAT_MODULE_NAME]
else:
    _format_spec = importlib.util.spec_from_file_location(
        _FORMAT_MODULE_NAME, _FORMAT_PATH
    )
    if _format_spec is None or _format_spec.loader is None:
        raise ImportError(f"cannot load canonical GAM format from {_FORMAT_PATH}")
    _format = importlib.util.module_from_spec(_format_spec)
    # dataclasses and Enum introspection require the module to be registered
    # while its body executes.
    sys.modules[_FORMAT_MODULE_NAME] = _format
    _format_spec.loader.exec_module(_format)

BOX_END = _format.BOX_END
GRID_MAX = _format.GRID_MAX
OBJECT_REF_START = _format.OBJECT_REF_START
TaskType = _format.TaskType
build_dense_bbox_prompt = _format.build_dense_bbox_prompt
build_dense_point_prompt = _format.build_dense_point_prompt
build_gui_prompt = _format.build_gui_prompt
build_layout_prompt = _format.build_layout_prompt
build_ocr_prompt = _format.build_ocr_prompt
build_point_prompt = _format.build_point_prompt
build_refer_bbox_prompt = _format.build_refer_bbox_prompt
build_refer_point_prompt = _format.build_refer_point_prompt
build_visual_prompt = _format.build_visual_prompt
canonicalize_phrase = _format.canonicalize_phrase
parse_answer = _format.parse_answer


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
NativePrediction = Tuple[str, Tuple[int, ...]]

_REX_REF_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    r"\s*<\|box_start\|>(.*?)<\|box_end\|>",
    re.DOTALL,
)
_LOCATE_REF_RE = re.compile(
    r"<ref>([^<]+)</ref>\s*((?:<box>.*?</box>\s*)+)",
    re.DOTALL,
)
_LOCATE_BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
_ANGLE_COORD_RE = re.compile(r"<([0-9]{1,4})>")
_REX_COORD_ID_START = 150643
_REX_COORD_ID_END = 151642


def _native_answer_slice(text: object) -> str:
    """Remove only sanctioned wrappers around an otherwise exact answer.

    After an optional ``<think>`` block or Markdown fence, the whole response
    must be the canonical native answer.  In particular, no misplaced box/ref
    wrapper can be hidden in ignored prefix/suffix prose.
    """

    if not isinstance(text, str) or not text:
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned.startswith(OBJECT_REF_START) or not cleaned.endswith(BOX_END):
        return ""
    return cleaned


def parse_native_predictions(text: object, task: TaskType | str) -> List[NativePrediction]:
    """Parse canonical native GAM output into one prediction per coordinate tuple.

    Natural-language labels may contain commas. Parsing is delegated to the
    training grammar instead of a regex. Therefore each label must occur once,
    each object-ref must be immediately followed by exactly one box wrapper,
    and that wrapper contains one or more same-arity tuples separated by a
    comma with no whitespace. Non-canonical model output is scored as no result.
    """

    candidate = _native_answer_slice(text)
    if not candidate:
        return []
    try:
        entries = parse_answer(candidate, task, require_canonical=True)
    except Exception:
        return []

    predictions: List[NativePrediction] = []
    for entry in entries:
        if entry.coordinates is None:
            continue
        predictions.extend((entry.phrase, coordinate) for coordinate in entry.coordinates)
    return predictions


_GAM_SCORING_ENTRY_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.DOTALL,
)
_GAM_SCORING_COORD_RE = re.compile(r"<([0-9]{1,3})>")


def _deduplicate_predictions(
    predictions: Iterable[NativePrediction],
) -> List[NativePrediction]:
    """Keep the first occurrence of an exact label/coordinate prediction."""

    output: List[NativePrediction] = []
    seen: set[NativePrediction] = set()
    for label, coordinate in predictions:
        item = (str(label), tuple(int(value) for value in coordinate))
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _parse_gam_predictions_deduplicated(
    text: object, task: TaskType | str
) -> List[NativePrediction]:
    """Relax only duplicate labels/tuples while retaining canonical GAM syntax.

    The training parser intentionally rejects duplicate labels and duplicate
    coordinate tuples.  Evaluation merges those repetitions for scoring, but
    still rejects malformed separators, unsafe phrases, invalid geometry,
    incomplete wrappers, and non-canonical coordinate payloads.
    """

    task_type = task if isinstance(task, TaskType) else TaskType(task)
    candidate = _native_answer_slice(text)
    if not candidate:
        return []
    matches = list(_GAM_SCORING_ENTRY_RE.finditer(candidate))
    if not matches:
        return []
    reconstructed = ", ".join(match.group(0) for match in matches)
    if reconstructed != candidate:
        return []

    output: List[NativePrediction] = []
    for match in matches:
        raw_label = match.group(1)
        try:
            label = canonicalize_phrase(raw_label)
        except Exception:
            return []
        if label != raw_label:
            return []
        payload = match.group(2)
        components = payload.split(",")
        if not components or any(not component for component in components):
            return []
        for component in components:
            values_text = _GAM_SCORING_COORD_RE.findall(component)
            if (
                not values_text
                or "".join(f"<{value}>" for value in values_text) != component
            ):
                return []
            coordinate = tuple(int(value) for value in values_text)
            if not _validate_foreign_tuple(coordinate, task_type, GRID_MAX):
                return []
            output.append((label, coordinate))
    return _deduplicate_predictions(output)


def parse_mode_predictions_for_scoring(
    text: object, task: TaskType | str
) -> List[NativePrediction]:
    """Parse mode output and deduplicate exact predictions for metric scoring."""

    predictions = parse_mode_predictions(text, task)
    if predictions:
        return _deduplicate_predictions(predictions)
    if get_eval_mode() in ("GAM", "DLM", "RLV2"):
        return _parse_gam_predictions_deduplicated(text, task)
    return []


@lru_cache(maxsize=2)
def _rex_tokenizer(path: str):
    """Load the FA3-safe Rex tokenizer used to reverse decoded coordinate ids."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def _rex_tokenizer_path() -> str:
    configured = os.environ.get("GAM_REXOMNI_TOKENIZER_PATH")
    if configured:
        return configured
    return str(_GAM_ROOT / "temp" / "model_overlays" / "rexomni-vllm-fa3")


@lru_cache(maxsize=2)
def _rex_coordinate_byte_prefixes(path: str):
    """Index Rex coordinate-token raw bytes, bypassing Unicode normalization."""

    tokenizer = _rex_tokenizer(path)
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    if byte_decoder is None:
        # Transformers 5.2 under Python 3.12 no longer exposes this legacy
        # Qwen2Tokenizer attribute.  Qwen2 uses the standard GPT-2 reversible
        # byte alphabet.  Reconstruct it locally because Transformers 5.2 also
        # removed the public ``bytes_to_unicode`` helper.
        byte_values = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        unicode_values = byte_values.copy()
        extra_index = 0
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value)
                unicode_values.append(256 + extra_index)
                extra_index += 1
        byte_decoder = {
            chr(character): value
            for value, character in zip(byte_values, unicode_values)
        }
    prefixes: dict[int, list[tuple[bytes, int]]] = {}
    for token_id in range(_REX_COORD_ID_START, _REX_COORD_ID_END + 1):
        token = tokenizer.convert_ids_to_tokens(token_id)
        try:
            raw = bytes(byte_decoder[character] for character in token)
        except (KeyError, TypeError):
            continue
        prefixes.setdefault(raw[0], []).append(
            (raw, token_id - _REX_COORD_ID_START)
        )
    for values in prefixes.values():
        values.sort(key=lambda item: len(item[0]), reverse=True)
    return prefixes


def _rex_payload_bytes_to_values(payload: str) -> Tuple[int, ...] | None:
    raw_payload = payload.encode("utf-8")
    prefixes = _rex_coordinate_byte_prefixes(_rex_tokenizer_path())
    # Dynamic programming is deliberate: BPE byte strings are not guaranteed
    # to be prefix-free, and compatibility ideographs are normalized by encode().
    paths: dict[int, Tuple[int, ...]] = {0: ()}
    for position in range(len(raw_payload)):
        path = paths.get(position)
        if path is None:
            continue
        for raw_token, value in prefixes.get(raw_payload[position], ()):
            if raw_payload.startswith(raw_token, position):
                paths.setdefault(position + len(raw_token), path + (value,))
    return paths.get(len(raw_payload))


def _rex_payload_values(payload: str) -> Tuple[int, ...] | None:
    """Recover Rex coordinates from either visible ``<n>`` or FA3 Unicode text."""

    direct = _ANGLE_COORD_RE.findall(payload)
    if direct and "".join(f"<{value}>" for value in direct) == payload:
        values = tuple(int(value) for value in direct)
        return values if all(0 <= value <= GRID_MAX for value in values) else None
    try:
        token_ids = _rex_tokenizer(_rex_tokenizer_path()).encode(
            payload, add_special_tokens=False
        )
    except Exception:
        return None
    if token_ids and all(
        _REX_COORD_ID_START <= token_id <= _REX_COORD_ID_END
        for token_id in token_ids
    ):
        return tuple(token_id - _REX_COORD_ID_START for token_id in token_ids)
    return _rex_payload_bytes_to_values(payload)


def _validate_foreign_tuple(values: Sequence[int], task: TaskType, maximum: int) -> bool:
    if len(values) != task.arity or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum
        for value in values
    ):
        return False
    return task is not TaskType.BBOX or (values[0] < values[2] and values[1] < values[3])


def parse_rexomni_predictions(text: object, task: TaskType | str) -> List[NativePrediction]:
    """Parse Rex-Omni wrappers without weakening the canonical GAM parser."""

    task_type = task if isinstance(task, TaskType) else TaskType(task)
    cleaned = _THINK_RE.sub("", str(text or "")).strip()
    matches = list(_REX_REF_RE.finditer(cleaned))
    if not matches:
        return []
    output: List[NativePrediction] = []
    for match in matches:
        label = match.group(1).strip()
        if not label:
            return []
        for component in match.group(2).split(","):
            values = _rex_payload_values(component.strip())
            if values is None or len(values) % task_type.arity:
                return []
            for offset in range(0, len(values), task_type.arity):
                coordinate = values[offset : offset + task_type.arity]
                if not _validate_foreign_tuple(coordinate, task_type, GRID_MAX):
                    return []
                output.append((label, coordinate))
    return output


def parse_locateanything_predictions(
    text: object, task: TaskType | str
) -> List[NativePrediction]:
    """Parse LocateAnything's native ``<ref>/<box>`` 0..1000 protocol."""

    task_type = task if isinstance(task, TaskType) else TaskType(task)
    cleaned = _THINK_RE.sub("", str(text or "")).strip()
    matches = list(_LOCATE_REF_RE.finditer(cleaned))
    if not matches:
        return []
    output: List[NativePrediction] = []
    for match in matches:
        label = match.group(1).strip()
        if not label:
            return []
        payloads = _LOCATE_BOX_RE.findall(match.group(2))
        if not payloads:
            return []
        for payload in payloads:
            compact = re.sub(r"\s+", "", payload)
            if compact == "None":
                continue
            values_text = _ANGLE_COORD_RE.findall(compact)
            if (
                not values_text
                or "".join(f"<{value}>" for value in values_text) != compact
            ):
                return []
            values = tuple(int(value) for value in values_text)
            if len(values) % task_type.arity:
                return []
            for offset in range(0, len(values), task_type.arity):
                coordinate = values[offset : offset + task_type.arity]
                if not _validate_foreign_tuple(coordinate, task_type, 1000):
                    return []
                output.append((label, coordinate))
    return output


def parse_mode_predictions(text: object, task: TaskType | str) -> List[NativePrediction]:
    mode = get_eval_mode()
    if mode in ("GAM", "DLM", "RLV2"):
        return parse_native_predictions(text, task)
    if mode == "REXOMNI":
        return parse_rexomni_predictions(text, task)
    if mode == "LOCATEANYTHING":
        return parse_locateanything_predictions(text, task)
    return []


def mode_grid_max() -> int:
    return 1000 if is_locateanything_mode() else GRID_MAX


def mode_grid_to_abs(coordinate: Sequence[int], width: int, height: int) -> List[float]:
    denominator = float(mode_grid_max())
    if len(coordinate) != 4:
        raise ValueError(f"bbox needs four coordinates, got {coordinate!r}")
    x1, y1, x2, y2 = (float(value) for value in coordinate)
    return [x1 / denominator * width, y1 / denominator * height,
            x2 / denominator * width, y2 / denominator * height]


def mode_grid_point_to_pixel(
    coordinate: Sequence[int], width: int, height: int
) -> Tuple[int, int]:
    denominator = float(mode_grid_max())
    if len(coordinate) != 2:
        raise ValueError(f"point needs two coordinates, got {coordinate!r}")
    return (
        round(float(coordinate[0]) / denominator * max(width - 1, 0)),
        round(float(coordinate[1]) / denominator * max(height - 1, 0)),
    )


def mode_grid_point_to_norm(coordinate: Sequence[int]) -> List[float]:
    denominator = float(mode_grid_max())
    if len(coordinate) != 2:
        raise ValueError(f"point needs two coordinates, got {coordinate!r}")
    return [float(coordinate[0]) / denominator, float(coordinate[1]) / denominator]


def _labels(values: Iterable[str]) -> Tuple[str, ...]:
    labels = tuple(str(value).strip() for value in values)
    if not labels or any(not value for value in labels):
        raise ValueError("prompt requires at least one non-empty label")
    return labels


def build_mode_dense_bbox_prompt(phrases: Iterable[str]) -> str:
    labels = _labels(phrases)
    if is_gam_mode():
        return build_dense_bbox_prompt(labels)
    if is_rexomni_mode():
        return ("Detect " + ", ".join(labels)
                + ". Output the bounding box coordinates in [x0, y0, x1, y1] format.")
    return ("Locate all the instances that matches the following description: "
            + "</c>".join(labels) + ".")


def build_mode_refer_bbox_prompt(phrase: str, *, multiple: bool = True) -> str:
    label = _labels((phrase,))[0]
    if is_gam_mode():
        return build_refer_bbox_prompt(label)
    if is_rexomni_mode():
        return (f"Detect {label}. Output the bounding box coordinates in "
                "[x0, y0, x1, y1] format.")
    quantifier = "all the instances" if multiple else "a single instance"
    return f"Locate {quantifier} that matches the following description: {label}."


def build_mode_dense_point_prompt(phrases: Iterable[str]) -> str:
    labels = _labels(phrases)
    if is_gam_mode():
        return build_dense_point_prompt(labels)
    if is_rexomni_mode():
        return "Point to " + ", ".join(labels) + "."
    return "Point to: " + "</c>".join(labels) + "."


def build_mode_refer_point_prompt(phrase: str) -> str:
    label = _labels((phrase,))[0]
    if is_gam_mode():
        return build_refer_point_prompt(label)
    if is_rexomni_mode():
        return f"Point to {label}."
    return f"Point to: {label}."


def _rex_coordinate_text(value: int) -> str:
    tokenizer = _rex_tokenizer(_rex_tokenizer_path())
    token_id = _REX_COORD_ID_START + int(value)
    # The Rex server may render output ids as safe visible coordinates.  A
    # visual-prompt reference must instead retain the underlying BPE piece so
    # the stripped vLLM tokenizer encodes it back to the intended model id.
    raw_token = tokenizer._convert_id_to_token(token_id)
    return tokenizer.convert_tokens_to_string([raw_token])


def build_mode_visual_prompt(reference_boxes: Sequence[Sequence[int]]) -> str:
    if is_gam_mode():
        return build_visual_prompt(reference_boxes)
    if is_rexomni_mode():
        chunks = ["".join(_rex_coordinate_text(value) for value in box)
                  for box in reference_boxes]
        visual = '{"object": "' + ", ".join(chunks) + '"}'
        return ("Given reference boxes " + visual
                + " indicating one or more objects, find all objects with the same category "
                  "in the image and output their bounding boxes in [x0, y0, x1, y1] format.")
    # LocateAnything's official worker crops every reference box and appends
    # those crops after the question.  The category-set placeholder remains
    # exactly <image-2> even when several reference crops are supplied.
    if not reference_boxes:
        raise ValueError("LocateAnything visual prompt requires a reference box")
    return (
        "Detect all the objects in the image that belong to the category set: "
        "<image-2>."
    )


def build_mode_layout_prompt(categories: Iterable[str]) -> str:
    labels = _labels(categories)
    if is_gam_mode():
        return build_layout_prompt(labels)
    return build_mode_dense_bbox_prompt(labels)


def canonicalize_gam_gui_instruction(instruction: str) -> str:
    """Normalize the GUI label shared by the GAM prompt and strict scorer."""

    label = _labels((instruction,))[0]
    # Remove a trailing period from raw benchmark instructions because
    # build_gui_prompt adds the template's final period.
    if label.endswith("."):
        label = canonicalize_phrase(label[:-1].strip())
    return label


def build_mode_gui_prompt(instruction: str) -> str:
    label = _labels((instruction,))[0]
    if is_gam_mode():
        label = canonicalize_gam_gui_instruction(label)
        return build_gui_prompt(label)
    if is_rexomni_mode():
        return f'Point to element "{label}".'
    return f"Point to: {label}."


def gam_grid_to_abs(coordinate: Sequence[int], width: int, height: int) -> List[float]:
    """Map a GAM 0..999 bbox to absolute pixels."""

    if len(coordinate) != 4:
        raise ValueError(f"bbox needs four coordinates, got {coordinate!r}")
    x1, y1, x2, y2 = (float(value) for value in coordinate)
    return [
        x1 / GRID_MAX * width,
        y1 / GRID_MAX * height,
        x2 / GRID_MAX * width,
        y2 / GRID_MAX * height,
    ]


def gam_grid_point_to_pixel(
    coordinate: Sequence[int], width: int, height: int
) -> Tuple[int, int]:
    """Map a GAM 0..999 point to integer pixel indices."""

    if len(coordinate) != 2:
        raise ValueError(f"point needs two coordinates, got {coordinate!r}")
    x, y = coordinate
    return (
        round(float(x) / GRID_MAX * max(width - 1, 0)),
        round(float(y) / GRID_MAX * max(height - 1, 0)),
    )


def gam_grid_point_to_norm(coordinate: Sequence[int]) -> List[float]:
    """Map a GAM 0..999 point to normalized coordinates for point-in-box/mask."""

    if len(coordinate) != 2:
        raise ValueError(f"point needs two coordinates, got {coordinate!r}")
    return [float(coordinate[0]) / GRID_MAX, float(coordinate[1]) / GRID_MAX]


def _round_half_up(numerator: float, denominator: float) -> int:
    if denominator <= 0:
        raise ValueError("image dimension must be positive")
    # Pixel boxes in evaluation annotations are numeric and non-negative after
    # clipping, so floor(x + 0.5) is exact round-half-up for this domain.
    return int(numerator / denominator + 0.5)


def pixel_boxes_to_gam_grid(
    boxes: Iterable[Sequence[float]], width: int, height: int
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Convert absolute-pixel xyxy exemplars to valid canonical 0..999 boxes."""

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    converted = []
    seen = set()
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(box[i]) for i in range(4))
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        if x1 >= x2 or y1 >= y2:
            continue
        mapped = [
            min(GRID_MAX, _round_half_up(x1 * GRID_MAX, width)),
            min(GRID_MAX, _round_half_up(y1 * GRID_MAX, height)),
            min(GRID_MAX, _round_half_up(x2 * GRID_MAX, width)),
            min(GRID_MAX, _round_half_up(y2 * GRID_MAX, height)),
        ]
        # Preserve a valid non-degenerate source exemplar when grid rounding
        # collapses a very narrow box.
        if mapped[0] >= mapped[2]:
            if mapped[2] < GRID_MAX:
                mapped[2] += 1
            elif mapped[0] > 0:
                mapped[0] -= 1
        if mapped[1] >= mapped[3]:
            if mapped[3] < GRID_MAX:
                mapped[3] += 1
            elif mapped[1] > 0:
                mapped[1] -= 1
        item = tuple(mapped)
        if item not in seen:
            converted.append(item)
            seen.add(item)
    if not converted:
        raise ValueError("visual prompt contains no valid reference bbox")
    return tuple(converted)


def pixel_boxes_to_mode_grid(
    boxes: Iterable[Sequence[float]], width: int, height: int
) -> Tuple[Tuple[int, int, int, int], ...]:
    converted = pixel_boxes_to_gam_grid(boxes, width, height)
    if not is_locateanything_mode():
        return converted
    return tuple(
        tuple((value * 1000 + GRID_MAX // 2) // GRID_MAX for value in box)
        for box in converted
    )


__all__ = [
    "EVAL_MODE_ENV",
    "VALID_EVAL_MODES",
    "TaskType",
    "build_dense_bbox_prompt",
    "build_dense_point_prompt",
    "build_gui_prompt",
    "build_layout_prompt",
    "build_ocr_prompt",
    "build_point_prompt",
    "build_refer_bbox_prompt",
    "build_refer_point_prompt",
    "build_visual_prompt",
    "build_mode_dense_bbox_prompt",
    "build_mode_dense_point_prompt",
    "build_mode_gui_prompt",
    "build_mode_layout_prompt",
    "build_mode_refer_bbox_prompt",
    "build_mode_refer_point_prompt",
    "build_mode_visual_prompt",
    "gam_grid_point_to_norm",
    "gam_grid_point_to_pixel",
    "gam_grid_to_abs",
    "get_eval_mode",
    "is_gam_mode",
    "is_groundingdino_mode",
    "is_locateanything_mode",
    "is_native_spatial_mode",
    "is_rexomni_mode",
    "mode_grid_point_to_norm",
    "mode_grid_point_to_pixel",
    "parse_mode_predictions_for_scoring",
    "mode_grid_to_abs",
    "normalize_eval_mode",
    "parse_locateanything_predictions",
    "parse_mode_predictions",
    "parse_native_predictions",
    "parse_rexomni_predictions",
    "pixel_boxes_to_gam_grid",
    "pixel_boxes_to_mode_grid",
]
