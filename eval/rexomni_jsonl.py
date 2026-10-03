"""HuggingFace dataset loader for the official Rex-Omni JSONL files.

The official annotations intentionally use dynamic category names as JSON object
keys.  Arrow's generic JSON reader treats those keys as a struct schema and can
therefore fail when later records introduce new categories.  This loader keeps
the official files as the sole source of truth and serializes only the dynamic
objects into JSON strings before Arrow sees them.
"""

from __future__ import annotations

import json
from pathlib import Path

import datasets


_JSON_FIELDS = ("gt", "gt_mask", "visual_prompt", "predict")


class RexOmniJsonl(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image_path": datasets.Value("string"),
                    "image_id": datasets.Value("string"),
                    "categories": datasets.Sequence(datasets.Value("string")),
                    "gt": datasets.Value("string"),
                    "gt_mask": datasets.Value("string"),
                    "visual_prompt": datasets.Value("string"),
                    "predict": datasets.Value("string"),
                    "task_name": datasets.Value("string"),
                    "dataset_name": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "sub_domain": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError("RexOmniJsonl requires data_files")
        resolved = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(resolved, dict):
            return [
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={"files": self._as_file_list(files)},
                )
                for split_name, files in resolved.items()
            ]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": self._as_file_list(resolved)},
            )
        ]

    @staticmethod
    def _as_file_list(files):
        if isinstance(files, (str, Path)):
            return [str(files)]
        return [str(path) for path in files]

    def _generate_examples(self, files):
        index = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        source = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid Rex-Omni JSONL at {path}:{line_number}") from exc

                    categories = source.get("categories") or []
                    if isinstance(categories, str):
                        categories = [categories]
                    row = {
                        "image_path": str(source.get("image_path", "")),
                        "image_id": str(source.get("image_id", "")),
                        "categories": [str(value) for value in categories],
                        "task_name": str(source.get("task_name", "")),
                        "dataset_name": str(source.get("dataset_name", "")),
                        "domain": str(source.get("domain", "")),
                        "sub_domain": str(source.get("sub_domain", "")),
                    }
                    for field in _JSON_FIELDS:
                        value = source.get(field)
                        row[field] = (
                            "" if value is None else json.dumps(value, ensure_ascii=False)
                        )
                    yield index, row
                    index += 1
