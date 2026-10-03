"""Deterministic loader and audit for the isolated RLV2 rebalanced epoch."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/rl/rebalanced/train.jsonl"
AUDIT = DATA.with_name("audit.json")
EXPECTED_ROWS = 17_700
EXPECTED_ROUTES = frozenset(
    {
        "grounding", "ocr", "dense", "referring", "referring_point", "gui",
        "robo_point", "visual_prompt", "grounding_point", "layout", "dense_point",
    }
)


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class RebalancedRLV2Mixture:
    """One deterministic shuffled epoch, with explicit repeat provenance."""

    def __init__(self, seed: int, path: Path = DATA) -> None:
        path = path.resolve(strict=True)
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"rebalanced RLV2 row drift: {len(rows)} != {EXPECTED_ROWS}")
        ids = [str(row.get("id", "")) for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("rebalanced RLV2 IDs are not unique")
        routes = {str(row.get("rlv3_route", "")) for row in rows}
        if routes != EXPECTED_ROUTES:
            raise ValueError(f"rebalanced RLV2 route drift: {sorted(routes)}")
        audit_path = path.with_name("audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS" or int(audit.get("effective_rows", -1)) != EXPECTED_ROWS:
            raise ValueError("rebalanced RLV2 audit is not PASS")
        if audit.get("data_sha256"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != audit["data_sha256"]:
                raise ValueError("rebalanced RLV2 data SHA drift")
        rng = random.Random(int(seed) ^ 0x524C563252454241)
        rng.shuffle(rows)
        self.rows = rows
        route_rows = Counter(str(row["rlv3_route"]) for row in rows)
        route_tokens = Counter()
        buckets = Counter()
        tiers = Counter()
        repeats = 0
        for row in rows:
            route = str(row["rlv3_route"])
            item = row["rlv3_audit"]
            route_tokens[route] += int(item["reference_tokens"])
            buckets[f"{route}:{item['length_bucket']}"] += 1
            tiers[str(item.get("source_tier", "unknown"))] += 1
            repeats += int("rlv2_rebalance" in row)
        self.audit = {
            "schema_version": 2,
            "policy": "modest_route_rebalance_fixed_total_17700_stratified_length",
            "seed": int(seed),
            "source": str(path),
            "effective_rows": len(rows),
            "route_rows": dict(sorted(route_rows.items())),
            "route_reference_tokens": dict(sorted(route_tokens.items())),
            "route_token_share": {
                route: route_tokens[route] / sum(route_tokens.values())
                for route in sorted(route_tokens)
            },
            "route_length_bucket_rows": dict(sorted(buckets.items())),
            "provenance_rows": dict(sorted(tiers.items())),
            "explicit_repeat_rows": repeats,
            "ordered_effective_ids_sha256": _sha256_lines([str(row["id"]) for row in rows]),
            "builder_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def row_for_group(self, optimizer_step: int, group_index: int, groups_per_step: int) -> dict[str, Any]:
        index = int(optimizer_step) * int(groups_per_step) + int(group_index)
        return self.rows[index % len(self.rows)]

    def write_audit(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(self.audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

