"""Index-only multi-source dataset and online VLM collator for DLM mode."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from train.runtime.tar_uri import parse_tar_uri, read_tar_uri


IMAGE_PLACEHOLDER = "<image>"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedCacheDataset(Dataset):
    """Expose selected rows without copying or mutating any Arrow cache."""

    def __init__(self, manifest_path: str | Path) -> None:
        from datasets import load_from_disk

        self.manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sources: list[dict[str, Any]] = []
        self.stops: list[int] = []
        total = 0
        for source in manifest["sources"]:
            indices_path = self.manifest_path.parent / source["local_indices"]
            indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
            if len(indices) != int(source["sampled_rows"]):
                raise RuntimeError(f"sample index count drift: {indices_path}")
            dataset = load_from_disk(source["cache_path"], keep_in_memory=False)
            if len(dataset) != int(source["rows"]):
                raise RuntimeError(f"cache row count drift: {source['cache_path']}")
            self.sources.append({"metadata": source, "indices": indices, "dataset": dataset})
            total += len(indices)
            self.stops.append(total)
        if total != int(manifest["sampled_rows"]):
            raise RuntimeError("sample manifest row conservation failed")
        self.rows = total
        self._dlm_packing_workloads: np.ndarray | None = None
        self._dlm_packing_workload_path: Path | None = None

    def attach_packing_workloads(self, path: str | Path) -> None:
        """Attach the immutable per-row DLM workload sidecar.

        The sidecar is used only for pack scheduling and the Qwen3 selective
        activation-offload gate.  It never changes tokens, labels, or loss.
        """

        workload_path = Path(path).resolve(strict=True)
        workloads = np.load(workload_path, mmap_mode="r", allow_pickle=False)
        if workloads.ndim != 1 or len(workloads) != self.rows:
            raise RuntimeError("DLM packing-workload sidecar shape drift")
        if not np.issubdtype(workloads.dtype, np.integer):
            raise RuntimeError("DLM packing-workload sidecar must be integral")
        if int(workloads.min()) <= 0:
            raise RuntimeError("DLM packing-workload sidecar contains non-positive rows")
        self._dlm_packing_workloads = workloads
        self._dlm_packing_workload_path = workload_path

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.rows
        if not 0 <= index < self.rows:
            raise IndexError(index)
        source_index = bisect_right(self.stops, index)
        start = 0 if source_index == 0 else self.stops[source_index - 1]
        source = self.sources[source_index]
        local_index = int(source["indices"][index - start])
        row = dict(source["dataset"][local_index])
        row["_dlm_source_id"] = source["metadata"]["source_id"]
        row["_dlm_local_index"] = local_index
        if self._dlm_packing_workloads is not None:
            row["_dlm_packing_workload"] = int(self._dlm_packing_workloads[index])
        return row

    def extreme_cached_length_index(self, limit: int, longest: bool) -> int:
        """Find an extreme selected row without materializing image payloads."""

        remaining = min(int(limit), self.rows)
        global_start = 0
        best_global_index = -1
        best_length: int | None = None
        for source in self.sources:
            count = min(remaining, len(source["indices"]))
            if count <= 0:
                break
            selected = np.asarray(source["indices"][:count], dtype=np.int64)
            table = source["dataset"].data
            lengths = np.asarray(table.column("length").to_numpy(zero_copy_only=False))
            selected_lengths = lengths[selected]
            local_position = int(selected_lengths.argmax() if longest else selected_lengths.argmin())
            value = int(selected_lengths[local_position])
            if best_length is None or (value > best_length if longest else value < best_length):
                best_length = value
                best_global_index = global_start + local_position
            global_start += count
            remaining -= count
        if best_global_index < 0:
            raise ValueError("cannot select from an empty dataset")
        return best_global_index

    def cached_lengths(self) -> np.ndarray:
        """Return selected cache lengths in dataset order without loading payloads."""

        cached = getattr(self, "_cached_selected_lengths", None)
        if cached is not None:
            return cached
        values: list[np.ndarray] = []
        for source in self.sources:
            selected = np.asarray(source["indices"], dtype=np.int64)
            table = source["dataset"].data
            # Take only selected rows.  Materializing every source's full
            # length column is wasteful for the independent 18M DLM pool.
            import pyarrow as pa

            selected_lengths = table.column("length").take(pa.array(selected))
            values.append(
                np.asarray(selected_lengths.to_numpy(zero_copy_only=False)).astype(
                    np.int32,
                    copy=False,
                )
            )
        cached = np.concatenate(values) if values else np.empty(0, dtype=np.int32)
        if len(cached) != self.rows:
            raise RuntimeError("cached length vector drift")
        self._cached_selected_lengths = cached
        return cached


class ComplementaryLengthSampler(Sampler[int]):
    """Pair long and short examples so each padding-free BS2 has stable tokens."""

    def __init__(self, lengths: np.ndarray, seed: int) -> None:
        if len(lengths) == 0 or len(lengths) % 2:
            raise ValueError("complementary BS2 sampling requires a positive even dataset")
        order = np.argsort(np.asarray(lengths), kind="stable")
        half = len(order) // 2
        self.pairs = np.stack((order[:half], order[: half - 1 : -1]), axis=1)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.pairs.size)

    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64DXSM(self.seed + self.epoch))
        self.epoch += 1
        pair_order = rng.permutation(len(self.pairs))
        pairs = self.pairs[pair_order].copy()
        swaps = rng.integers(0, 2, size=len(pairs), dtype=np.int8).astype(bool)
        pairs[swaps] = pairs[swaps, ::-1]
        return iter(pairs.reshape(-1).tolist())


class PrecomputedPackingSampler(Sampler[int]):
    """Memory-map deterministic, independently built DLM packing orders."""

    def __init__(
        self,
        manifest_path: str | Path,
        rows: int,
        sampling_manifest_path: str | Path,
        world_size: int | None = None,
        pack_size: int = 2,
        ordering_lengths_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.algorithm = manifest.get("algorithm")
        if self.algorithm not in {
            "dlm-bs2-long-short-pcg64dxsm-v1",
            "dlm-bs2-long-short-global-rank-bucket-pcg64dxsm-v2",
            "dlm-fixed-pack-quantile-zigzag-global-rank-bucket-pcg64dxsm-v3",
            "dlm-fixed-pack-quantile-balanced-global-rank-bucket-pcg64dxsm-v4",
        }:
            raise RuntimeError(f"unsupported DLM packing manifest: {self.manifest_path}")
        if self.algorithm.endswith(("-v2", "-v3", "-v4")) and int(manifest.get("world_size", -1)) != int(world_size):
            raise RuntimeError("DLM rank-bucket packing world-size drift")
        manifest_pack_size = int(manifest.get("pack_size", 2))
        if manifest_pack_size != int(pack_size):
            raise RuntimeError("DLM packing microbatch-size drift")
        if int(manifest.get("rows", -1)) != int(rows) or rows <= 0:
            raise RuntimeError("DLM packing manifest row count drift")
        if self.algorithm not in {
            "dlm-fixed-pack-quantile-zigzag-global-rank-bucket-pcg64dxsm-v3",
            "dlm-fixed-pack-quantile-balanced-global-rank-bucket-pcg64dxsm-v4",
        } and rows % 2:
            # The fixed BS2 rank-bucket builder may carry one explicit tail
            # row when a sampling pool is odd (for example General=499,939).
            # Trainer's distributed drop_last contract consumes only complete
            # global microbatches; accepting the declared tail here prevents
            # an otherwise valid, auditable manifest from being rejected.
            bucket_audit = manifest.get("global_rank_bucketing", {})
            if not (
                self.algorithm == "dlm-bs2-long-short-global-rank-bucket-pcg64dxsm-v2"
                and int(bucket_audit.get("tail_rows", 0)) == 1
            ):
                raise RuntimeError("legacy DLM BS2 packing row count drift")
        if Path(manifest.get("sampling_manifest", "")).resolve() != Path(
            sampling_manifest_path
        ).resolve(strict=True):
            raise RuntimeError("DLM packing source sampling manifest drift")
        manifest_sampling_sha = manifest.get("sampling_manifest_sha256")
        if manifest_sampling_sha and manifest_sampling_sha != _sha256_file(
            Path(sampling_manifest_path).resolve(strict=True)
        ):
            raise RuntimeError("DLM packing source sampling SHA drift")
        manifest_ordering = manifest.get("ordering_lengths")
        if manifest_ordering is not None:
            if ordering_lengths_path is None:
                raise RuntimeError("DLM packing ordering-length sidecar was not attached")
            attached_ordering = Path(ordering_lengths_path).resolve(strict=True)
            if Path(manifest_ordering).resolve() != attached_ordering:
                raise RuntimeError("DLM packing ordering-length path drift")
            if manifest.get("ordering_lengths_sha256") != _sha256_file(attached_ordering):
                raise RuntimeError("DLM packing ordering-length SHA drift")
        epoch_files = manifest.get("epoch_orders")
        if not isinstance(epoch_files, list) or not epoch_files:
            raise RuntimeError("DLM packing manifest has no epoch orders")
        self.orders = [
            np.load(self.manifest_path.parent / entry["path"], mmap_mode="r", allow_pickle=False)
            for entry in epoch_files
        ]
        if any(order.ndim != 1 or len(order) != rows for order in self.orders):
            raise RuntimeError("DLM packing epoch order shape drift")
        self.rows = int(rows)
        self.epoch = 0
        self.audit = manifest.get("pack_sum_audit", manifest.get("pair_sum_audit"))
        if not isinstance(self.audit, dict):
            raise RuntimeError("DLM packing manifest has no pack-sum audit")

    def __len__(self) -> int:
        return self.rows

    def __iter__(self):
        order = self.orders[self.epoch % len(self.orders)]
        self.epoch += 1
        return (int(index) for index in order)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _load_image(value: Any):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if not isinstance(value, dict) or set(value) != {"bytes", "path"}:
        raise TypeError(f"invalid cached image record: {type(value)!r}")
    payload = value["bytes"]
    path = value["path"]
    if payload is not None:
        source: Any = BytesIO(payload)
    elif parse_tar_uri(path) is not None:
        source = BytesIO(read_tar_uri(path))
    else:
        source = path
    with Image.open(source) as image:
        image.load()
        return image.convert("RGB")


def _typed_messages(messages: list[dict[str, Any]], image_count: int) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("empty messages")
    normalized = [dict(message) for message in messages]
    consumed = 0
    rendered: list[dict[str, Any]] = []
    for message in normalized:
        role = message["role"]
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("message content must be a string")
        # Byte-match the Swift 4.2 Qwen3.5 training template used to build the
        # immutable GAM caches: no default system prompt, per-role trimming,
        # canonical thinking blocks, and an empty-thinking prefix for ordinary
        # assistant answers.
        if role in ("user", "system", "tool"):
            content = content.strip()
        elif role == "assistant":
            stripped = content.strip()
            if "</think>" in stripped and "<think>" in stripped:
                before, _, after = stripped.partition("</think>")
                reasoning = before.rstrip("\n").rsplit("<think>", 1)[-1].lstrip("\n").strip()
                rest = after.lstrip("\n")
                content = f"<think>\n{reasoning}\n</think>\n\n{rest}"
            else:
                content = stripped
            if not content.startswith(("<think>", NON_THINKING_PREFIX)):
                content = NON_THINKING_PREFIX + content
        pieces = content.split(IMAGE_PLACEHOLDER)
        if len(pieces) == 1:
            typed_content: Any = content
        else:
            typed_content = []
            for index, piece in enumerate(pieces):
                if piece:
                    typed_content.append({"type": "text", "text": piece})
                if index + 1 < len(pieces):
                    typed_content.append({"type": "image"})
                    consumed += 1
        rendered.append({"role": role, "content": typed_content})
    if consumed != image_count:
        raise ValueError(f"image placeholder mismatch: placeholders={consumed} images={image_count}")
    return rendered


def _render_groundinganything_chatml(messages: list[dict[str, Any]]) -> str:
    """Render the GroundAnything Swift template without tokenizer-injected system text.

    The checkpoint tokenizer's standalone Jinja template inserts ``You are a
    helpful assistant.`` when the first role is not ``system``.  The VLM
    training route registers ``default_system=None`` and uses Swift's ChatML
    contexts, so Direct Conversion must render that same byte stream.
    """

    chunks: list[str] = []
    for message in messages:
        role = str(message["role"])
        content = message["content"]
        if isinstance(content, str):
            rendered = content
        else:
            pieces: list[str] = []
            for part in content:
                if part.get("type") == "text":
                    pieces.append(str(part["text"]))
                elif part.get("type") == "image":
                    pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif part.get("type") == "video":
                    pieces.append("<|vision_start|><|video_pad|><|vision_end|>")
                else:
                    raise ValueError(f"unsupported GroundAnything message part: {part}")
            rendered = "".join(pieces)
        chunks.append(f"<|im_start|>{role}\n{rendered}<|im_end|>\n")
    return "".join(chunks)


def assistant_labels(input_ids: torch.Tensor, tokenizer: Any, messages: list[dict[str, Any]]) -> torch.Tensor:
    """Supervise assistant bodies and im_end, preserving AR token shift."""

    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ignored_prefix = tokenizer.encode(NON_THINKING_PREFIX, add_special_tokens=False)
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    labels = torch.full_like(input_ids, -100)
    assistant_loss = [
        message.get("loss") != 0
        for message in messages
        if message.get("role") == "assistant"
    ]
    cursor = 0
    occurrence = 0
    ids = input_ids.tolist()
    while cursor <= len(ids) - len(header):
        if ids[cursor : cursor + len(header)] != header:
            cursor += 1
            continue
        start = cursor + len(header)
        try:
            stop = ids.index(im_end, start)
        except ValueError as exc:
            raise ValueError("assistant response lacks <|im_end|>") from exc
        if occurrence >= len(assistant_loss):
            raise ValueError("rendered assistant/header count drift")
        if assistant_loss[occurrence]:
            labels[start : stop + 1] = input_ids[start : stop + 1]
            # Match GAM's `loss_scale: ignore_empty_think`: the injected empty
            # reasoning wrapper is context, not a supervised target.
            if ids[start : start + len(ignored_prefix)] == ignored_prefix:
                labels[start : start + len(ignored_prefix)] = -100
        occurrence += 1
        cursor = stop + 1
    if occurrence != len(assistant_loss):
        raise ValueError(
            f"assistant/header count drift: messages={len(assistant_loss)} rendered={occurrence}"
        )
    if not labels.ne(-100).any():
        raise ValueError("sample contains no supervised assistant tokens")
    return labels


class DLMDataCollator:
    def __init__(
        self,
        processor: Any,
        max_length: int = 4096,
        max_joint_length: int = 8192,
        model_family: str = "qwen3_5",
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_joint_length = max_joint_length
        if model_family not in {"qwen3_5", "groundinganything_qwen3"}:
            raise ValueError(f"unsupported DLM collator model family: {model_family}")
        self.model_family = model_family

    def _encode(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = []
        for image_index, value in enumerate(row["images"]):
            try:
                images.append(_load_image(value))
            except Exception as error:
                path = value.get("path") if isinstance(value, dict) else None
                raise OSError(
                    "DLM image decode failed: "
                    f"id={row.get('id')!r} source={row.get('_dlm_source_id')!r} "
                    f"local_index={row.get('_dlm_local_index')!r} "
                    f"image_index={image_index} path={path!r}: {error}"
                ) from error
        messages = _typed_messages(row["messages"], len(images))
        if self.model_family == "groundinganything_qwen3":
            text = _render_groundinganything_chatml(messages)
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        if self.model_family == "groundinganything_qwen3" and images:
            # GroundAnything's lightweight remote processor intentionally leaves one
            # image placeholder per image.  The VLM Swift template expands it
            # after Kimi-K3 preprocessing to exactly prod(grid)/merge_size^2
            # language tokens.  Reproduce that route before tokenization.
            image_inputs = self.processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            grid = image_inputs["image_grid_thw"]
            merge_size = int(self.processor.image_processor.merge_size)
            token_counts = (grid.prod(dim=-1) // (merge_size**2)).tolist()
            placeholder = "<|image_pad|>"
            if text.count(placeholder) != len(token_counts):
                raise ValueError(
                    "GroundAnything image placeholder count drift: "
                    f"placeholders={text.count(placeholder)} images={len(token_counts)}"
                )
            pieces = text.split(placeholder)
            text = pieces[0] + "".join(
                placeholder * int(token_count) + suffix
                for token_count, suffix in zip(token_counts, pieces[1:], strict=True)
            )
            text_inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=False,
            )
            encoded = {**dict(text_inputs), **dict(image_inputs)}
        else:
            encoded = self.processor(
                text=[text],
                images=images or None,
                padding=False,
                return_tensors="pt",
            )
        result = dict(encoded)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in result and result[key].ndim == 2 and result[key].shape[0] == 1:
                result[key] = result[key].squeeze(0)
        input_ids = result["input_ids"]
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"tokenized row exceeds max_length: id={row.get('id')} length={input_ids.numel()}"
            )
        vision_ids = {
            int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>")),
            int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>")),
            int(self.tokenizer.convert_tokens_to_ids("<|video_pad|>")),
        }
        noisy_length = sum(int(token) not in vision_ids for token in input_ids.tolist())
        if noisy_length + input_ids.numel() > self.max_joint_length:
            raise ValueError(
                f"DLM joint row exceeds max_joint_length: id={row.get('id')} "
                f"joint={noisy_length + input_ids.numel()}"
            )
        result["labels"] = assistant_labels(input_ids, self.tokenizer, row["messages"])
        if "mm_token_type_ids" not in result:
            mm_types = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_types[input_ids.eq(int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>")))] = 1
            mm_types[input_ids.eq(int(self.tokenizer.convert_tokens_to_ids("<|video_pad|>")))] = 2
            result["mm_token_type_ids"] = mm_types
        return result

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in rows]
        pad_id = int(self.tokenizer.pad_token_id)
        batch: dict[str, torch.Tensor] = {
            "input_ids": nn_pad([item["input_ids"] for item in encoded], pad_id),
            "attention_mask": nn_pad([item["attention_mask"] for item in encoded], 0),
            "labels": nn_pad([item["labels"] for item in encoded], -100),
            "mm_token_type_ids": nn_pad([item["mm_token_type_ids"] for item in encoded], 0),
        }
        packing_workloads = [row.get("_dlm_packing_workload") for row in rows]
        if any(value is not None for value in packing_workloads):
            if not all(value is not None for value in packing_workloads):
                raise RuntimeError("partial DLM packing-workload metadata in one microbatch")
            # Keep this as a Python scalar.  Trainer leaves scalars on host, so
            # the model can make the gate decision without a CUDA .item() sync.
            batch["packing_workload_tokens"] = sum(int(value) for value in packing_workloads)
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "patch_positions",
        ):
            values = [item[key] for item in encoded if key in item]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


def nn_pad(values: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding_value)
