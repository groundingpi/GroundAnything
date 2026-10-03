"""Exact GroundAnything/K3 multimodal dataset adapter for standard DARE/verl PPO."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from train.dlm.data import NON_THINKING_PREFIX, _load_image, _render_groundinganything_chatml, _typed_messages


class GAMCausalRLDataset(Dataset):
    """Read the immutable RLV3 JSONL and emit exact causal GroundAnything prompts."""

    def __init__(self, data_files, tokenizer, config, processor=None):
        if isinstance(data_files, (str, Path)):
            data_files = [data_files]
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        if processor is None:
            raise RuntimeError("GAM RLV3 requires the GroundAnything multimodal processor")
        self.max_prompt_length = int(config.get("max_prompt_length", 8192))
        self.pad_to_batch_size = int(config.get("gam_pad_to_batch_size", 0))
        self.rows: list[dict[str, Any]] = []
        for path in data_files:
            with Path(str(path)).resolve(strict=True).open(encoding="utf-8") as stream:
                self.rows.extend(json.loads(line) for line in stream if line.strip())
        if not self.rows:
            raise RuntimeError("empty GAM RLV3 dataset")
        ids = [str(row["id"]) for row in self.rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("GAM RLV3 dataset contains duplicate immutable IDs")
        self.original_rows = len(self.rows)
        self.effective_rows = (
            math.ceil(self.original_rows / self.pad_to_batch_size) * self.pad_to_batch_size
            if self.pad_to_batch_size > 0
            else self.original_rows
        )
        print(
            json.dumps(
                {
                    "gam_rlv3_dataset": {
                        "status": "PASS",
                        "unique_rows": self.original_rows,
                        "effective_rows": self.effective_rows,
                        "deterministic_tail_repeats": self.effective_rows - self.original_rows,
                        "max_prompt_length": self.max_prompt_length,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def __len__(self) -> int:
        return self.effective_rows

    def resume_dataset_state(self) -> None:
        return None

    def _encode(
        self, row: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], list[Any], str, dict[str, torch.Tensor]]:
        images = [
            _load_image(value if isinstance(value, dict) else {"bytes": None, "path": value})
            for value in row.get("images") or []
        ]
        typed = _typed_messages(row["messages"], len(images))
        text = _render_groundinganything_chatml(typed) + "<|im_start|>assistant\n" + NON_THINKING_PREFIX
        image_inputs: dict[str, torch.Tensor] = {}
        if images:
            image_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
            grid = image_inputs["image_grid_thw"]
            merge_size = int(self.processor.image_processor.merge_size)
            counts = (grid.prod(dim=-1) // (merge_size**2)).tolist()
            placeholder = "<|image_pad|>"
            if text.count(placeholder) != len(counts):
                raise ValueError(f"image placeholder drift for {row['id']}")
            parts = text.split(placeholder)
            text = parts[0] + "".join(
                placeholder * int(count) + suffix
                for count, suffix in zip(counts, parts[1:], strict=True)
            )
        encoded = self.tokenizer([text], return_tensors="pt", padding=False, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if int(input_ids.shape[1]) > self.max_prompt_length:
            raise RuntimeError(
                f"GAM RLV3 prompt overrun id={row['id']} length={input_ids.shape[1]} "
                f"limit={self.max_prompt_length}"
            )
        return image_inputs, images, text, encoded

    def __getitem__(self, item: int) -> dict[str, Any]:
        from verl.utils.model import compute_position_id_with_mask
        import verl.utils.torch_functional as verl_F

        source_index = int(item) % self.original_rows
        row = copy.deepcopy(self.rows[source_index])
        image_inputs, images, text, encoded = self._encode(row)
        raw_prompt_ids = encoded["input_ids"][0].tolist()
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        output: dict[str, Any] = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": compute_position_id_with_mask(attention_mask)[0],
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": {"image": images},
            "multi_modal_inputs": image_inputs,
            "data_source": str(row["rlv3_route"]),
            "reward_model": {"ground_truth": row["solution"]},
            "extra_info": {
                "index": source_index,
                "immutable_id": str(row["id"]),
                "deterministic_tail_repeat": int(item) >= self.original_rows,
                "rlv3_row": row,
            },
            "index": source_index,
            "tools_kwargs": {},
        }
        return output

    def __getstate__(self):
        return self.__dict__.copy()
