"""Auditable Grounding/OCR sampling used by corrected TraceRL and Causal-JustGRPO.

The released recipe cycles the two sources to a 50/50 mixture.  The corrected
contract keeps every Grounding row, selects exactly 80% of the 5,500 unique OCR
rows without replacement, and interleaves both sources with bounded prefix
error.  No source row is repeated inside one logical epoch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from train.rl.data import GROUNDING_DATA, OCR_DATA, _read_jsonl


EXPECTED_GROUNDING_ROWS = 6_557
EXPECTED_OCR_ROWS = 5_500
OCR_KEEP_NUMERATOR = 4
OCR_KEEP_DENOMINATOR = 5


def _stable_ids(rows: list[dict[str, Any]]) -> list[str]:
    values = [str(row["id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError("corrected RL source contains duplicate row ids")
    return values


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _proportional_schedule(grounding_count: int, ocr_count: int) -> list[str]:
    """Return an exact-count, low-discrepancy two-source schedule."""

    total = grounding_count + ocr_count
    schedule: list[str] = []
    previous_ocr = 0
    for index in range(total):
        expected_ocr = ((index + 1) * ocr_count) // total
        if expected_ocr > previous_ocr:
            schedule.append("ocr")
            previous_ocr = expected_ocr
        else:
            schedule.append("grounding")
    if schedule.count("ocr") != ocr_count or schedule.count("grounding") != grounding_count:
        raise RuntimeError("corrected RL proportional interleave lost source rows")
    return schedule


class CorrectedGroundingOCRMixture:
    """6,557 Grounding + deterministic 4,400/5,500 unique OCR rows."""

    def __init__(
        self,
        seed: int,
        grounding: Path = GROUNDING_DATA,
        ocr: Path = OCR_DATA,
    ) -> None:
        grounding_rows = _read_jsonl(grounding)
        ocr_rows = _read_jsonl(ocr)
        if len(grounding_rows) != EXPECTED_GROUNDING_ROWS:
            raise ValueError(
                f"Grounding source count drift: {len(grounding_rows)} != {EXPECTED_GROUNDING_ROWS}"
            )
        if len(ocr_rows) != EXPECTED_OCR_ROWS:
            raise ValueError(f"OCR source count drift: {len(ocr_rows)} != {EXPECTED_OCR_ROWS}")
        _stable_ids(grounding_rows)
        _stable_ids(ocr_rows)

        grounding_rng = random.Random(int(seed) ^ 0x47524F554E44)
        ocr_rng = random.Random(int(seed) ^ 0x4F4352)
        grounding_rng.shuffle(grounding_rows)
        ocr_rng.shuffle(ocr_rows)
        ocr_keep = len(ocr_rows) * OCR_KEEP_NUMERATOR // OCR_KEEP_DENOMINATOR
        if ocr_keep != 4_400:
            raise RuntimeError(f"OCR 80% contract drift: {ocr_keep}")
        selected_ocr = ocr_rows[:ocr_keep]

        schedule = _proportional_schedule(len(grounding_rows), len(selected_ocr))
        grounding_iter = iter(grounding_rows)
        ocr_iter = iter(selected_ocr)
        self.rows = [
            next(ocr_iter) if source == "ocr" else next(grounding_iter)
            for source in schedule
        ]
        row_ids = _stable_ids(self.rows)
        selected_ocr_ids = _stable_ids(selected_ocr)
        self.audit = {
            "schema_version": 1,
            "policy": "grounding_all_plus_ocr_unique_without_replacement_80pct_low_discrepancy",
            "seed": int(seed),
            "grounding_source": str(grounding.resolve(strict=True)),
            "ocr_source": str(ocr.resolve(strict=True)),
            "grounding_source_rows": len(grounding_rows),
            "ocr_source_unique_rows": len(ocr_rows),
            "ocr_keep_numerator": OCR_KEEP_NUMERATOR,
            "ocr_keep_denominator": OCR_KEEP_DENOMINATOR,
            "grounding_effective_rows": len(grounding_rows),
            "ocr_effective_rows": len(selected_ocr),
            "effective_rows": len(self.rows),
            "grounding_effective_share": len(grounding_rows) / len(self.rows),
            "ocr_effective_share": len(selected_ocr) / len(self.rows),
            "selected_ocr_ids_sha256": _sha256_lines(selected_ocr_ids),
            "ordered_effective_ids_sha256": _sha256_lines(row_ids),
            "selected_ocr_ids": selected_ocr_ids,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def row_for_group(
        self,
        optimizer_step: int,
        group_index: int,
        groups_per_step: int,
    ) -> dict[str, Any]:
        index = int(optimizer_step) * int(groups_per_step) + int(group_index)
        return self.rows[index % len(self.rows)]

    def write_audit(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
