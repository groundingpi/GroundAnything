"""Immutable 17.7K multi-route epoch adapter for the RLV2 control run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/rl/multiroute/train.jsonl"
EXPECTED_ROWS = 17_700
EXPECTED_ROUTES = frozenset(
    {
        "grounding",
        "ocr",
        "dense",
        "referring",
        "referring_point",
        "gui",
        "robo_point",
        "visual_prompt",
        "grounding_point",
        "layout",
        "dense_point",
    }
)


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class MultirouteRLV2Mixture:
    """One deterministic, without-replacement epoch over the approved data."""

    def __init__(self, seed: int, path: Path = DATA) -> None:
        with path.resolve(strict=True).open(encoding="utf-8") as stream:
            rows: list[dict[str, Any]] = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"RLV2 multi-route row drift: {len(rows)} != {EXPECTED_ROWS}")
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("RLV2 multi-route epoch contains duplicate immutable IDs")
        routes = {str(row["rlv3_route"]) for row in rows}
        if routes != EXPECTED_ROUTES:
            raise ValueError(f"RLV2 multi-route route drift: {sorted(routes)}")
        rng = random.Random(int(seed) ^ 0x524C56324D554C54)
        rng.shuffle(rows)
        self.rows = rows
        ordered_ids = [str(row["id"]) for row in rows]
        route_rows = Counter(str(row["rlv3_route"]) for row in rows)
        route_tokens = Counter()
        tiers = Counter()
        buckets = Counter()
        for row in rows:
            audit = row["rlv3_audit"]
            route_tokens[str(row["rlv3_route"])] += int(audit["reference_tokens"])
            tiers[str(audit["source_tier"])] += 1
            buckets[f"{row['rlv3_route']}:{audit['length_bucket']}"] += 1
        self.audit = {
            "schema_version": 1,
            "policy": "rlv2_same_17p7k_multiroute_epoch_without_replacement",
            "seed": int(seed),
            "source": str(path.resolve()),
            "effective_rows": len(rows),
            "route_rows": dict(sorted(route_rows.items())),
            "route_reference_tokens": dict(sorted(route_tokens.items())),
            "provenance_rows": dict(sorted(tiers.items())),
            "route_length_bucket_rows": dict(sorted(buckets.items())),
            "ordered_effective_ids_sha256": _sha256_lines(ordered_ids),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def row_for_group(
        self, optimizer_step: int, group_index: int, groups_per_step: int
    ) -> dict[str, Any]:
        index = int(optimizer_step) * int(groups_per_step) + int(group_index)
        return self.rows[index % len(self.rows)]

    def write_audit(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
