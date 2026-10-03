"""Exact Grounding/OCR GRPO mixture and GroundAnything prompt encoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle, islice
import json
from pathlib import Path
import random
from typing import Any

import torch

from train.dlm.data import NON_THINKING_PREFIX, _load_image, _render_groundinganything_chatml, _typed_messages


GROUNDING_DATA = Path("data/rl/grounding/train.jsonl")
OCR_DATA = Path("data/rl/ocr/train.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"{path.stem}-{line_number:07d}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty GRPO dataset: {path}")
    return rows


class GroundingOCRMixture:
    """Deterministic 50/50 all-exhausted mixture from the released recipe."""

    def __init__(self, seed: int, grounding: Path = GROUNDING_DATA, ocr: Path = OCR_DATA):
        grounding_rows = _read_jsonl(grounding)
        ocr_rows = _read_jsonl(ocr)
        rng = random.Random(int(seed))
        rng.shuffle(grounding_rows)
        rng.shuffle(ocr_rows)
        target = max(len(grounding_rows), len(ocr_rows))
        grounding_epoch = list(islice(cycle(grounding_rows), target))
        ocr_epoch = list(islice(cycle(ocr_rows), target))
        self.rows: list[dict[str, Any]] = []
        for grounding_row, ocr_row in zip(grounding_epoch, ocr_epoch, strict=True):
            self.rows.extend((grounding_row, ocr_row))
        self.audit = {
            "grounding_source_rows": len(grounding_rows),
            "ocr_source_rows": len(ocr_rows),
            "effective_rows": len(self.rows),
            "grounding_effective_rows": target,
            "ocr_effective_rows": target,
            "seed": int(seed),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def row_for_group(self, optimizer_step: int, group_index: int, groups_per_step: int) -> dict[str, Any]:
        index = (int(optimizer_step) * int(groups_per_step) + int(group_index)) % len(self.rows)
        return self.rows[index]


@dataclass
class EncodedPrompt:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    patch_positions: torch.Tensor | None
    mm_token_type_ids: torch.Tensor

    def to(self, device: torch.device) -> "EncodedPrompt":
        return EncodedPrompt(
            **{
                name: value.to(device, non_blocking=True) if value is not None else None
                for name, value in self.__dict__.items()
            }
        )


class GroundAnythingPromptEncoder:
    def __init__(self, processor: Any, *, max_prompt_tokens: int = 12288):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)

    def __call__(self, row: dict[str, Any]) -> EncodedPrompt:
        raw_images = row.get("images") or []
        images = [
            _load_image(value if isinstance(value, dict) else {"bytes": None, "path": value})
            for value in raw_images
        ]
        typed = _typed_messages(row["messages"], len(images))
        text = _render_groundinganything_chatml(typed)
        text += "<|im_start|>assistant\n" + NON_THINKING_PREFIX

        image_inputs: dict[str, torch.Tensor] = {}
        if images:
            image_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
            grid = image_inputs["image_grid_thw"]
            merge_size = int(self.processor.image_processor.merge_size)
            token_counts = (grid.prod(dim=-1) // (merge_size**2)).tolist()
            placeholder = "<|image_pad|>"
            if text.count(placeholder) != len(token_counts):
                raise ValueError("GroundAnything RL image placeholder count drift")
            pieces = text.split(placeholder)
            text = pieces[0] + "".join(
                placeholder * int(count) + suffix
                for count, suffix in zip(token_counts, pieces[1:], strict=True)
            )
        text_inputs = self.tokenizer([text], return_tensors="pt", padding=False)
        input_ids = text_inputs["input_ids"]
        if int(input_ids.shape[1]) > self.max_prompt_tokens:
            raise ValueError(
                f"GRPO prompt exceeds {self.max_prompt_tokens} tokens: id={row.get('id')} "
                f"length={input_ids.shape[1]}"
            )
        mm_types = torch.zeros_like(input_ids, dtype=torch.int32)
        image_pad = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        video_pad = int(self.tokenizer.convert_tokens_to_ids("<|video_pad|>"))
        mm_types[input_ids.eq(image_pad)] = 1
        mm_types[input_ids.eq(video_pad)] = 2
        return EncodedPrompt(
            input_ids=input_ids,
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs.get("pixel_values"),
            image_grid_thw=image_inputs.get("image_grid_thw"),
            patch_positions=image_inputs.get("patch_positions"),
            mm_token_type_ids=mm_types,
        )
