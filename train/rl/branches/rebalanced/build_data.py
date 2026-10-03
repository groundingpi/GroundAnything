#!/usr/bin/env python3
"""Build the immutable, modestly rebalanced RLV2 epoch.

The previous RLV2 epoch is kept byte-for-byte at its original location.  This
builder creates a new manifest from that epoch only.  Route quotas change by
at most 150 rows, the total stays 17,700 (therefore the optimizer budget and
learning-rate exposure stay comparable), and every route's short/medium/long
mix is preserved by deterministic stratified sampling.  Repeats are explicit,
have new IDs, and are counted in the audit rather than silently changing the
source data.
"""

from __future__ import annotations


from collections import Counter, defaultdict
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[4]
GAM = ROOT
DEFAULT_SOURCE = GAM / "data/rl/multiroute/train.jsonl"
DEFAULT_OUTPUT = GAM / "data/rl/rebalanced"

ROUTES = (
    "grounding",
    "ocr",
    "dense",
    "dense_point",
    "grounding_point",
    "gui",
    "layout",
    "referring",
    "referring_point",
    "robo_point",
    "visual_prompt",
)

# Fixed total = 17,700.  The previous run over-invested in OCR and did not
# give enough independent dense/point/referring/GUI trajectories.  Changes
# are deliberately modest so this remains a controlled RLV2 follow-up.
TARGET_QUOTAS = {
    "grounding": 2300,
    "ocr": 2650,
    "dense": 1350,
    "dense_point": 800,
    "grounding_point": 1350,
    "gui": 1800,
    "layout": 800,
    "referring": 2050,
    "referring_point": 1900,
    "robo_point": 1000,
    "visual_prompt": 1700,
}
EXPECTED_ROWS = 17_700
EXPECTED_BUCKETS = ("short", "medium", "long")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def allocate_buckets(counts: dict[str, int], quota: int) -> dict[str, int]:
    """Largest-remainder allocation with a floor for every available bucket."""

    available = [bucket for bucket in EXPECTED_BUCKETS if counts.get(bucket, 0)]
    if not available or quota <= 0:
        return {bucket: 0 for bucket in EXPECTED_BUCKETS}
    if quota < len(available):
        # This is not reached by the approved quotas, but fail closed instead
        # of silently removing an entire length bucket in a future edit.
        raise ValueError(f"quota {quota} is smaller than buckets {available}")
    total = sum(counts[bucket] for bucket in available)
    raw = {bucket: quota * counts[bucket] / total for bucket in available}
    allocation = {bucket: min(1, counts[bucket]) for bucket in available}
    remaining = quota - sum(allocation.values())
    # Allocate the remainder by fractional part; deterministic ties by name.
    order = sorted(
        available,
        key=lambda bucket: (-(raw[bucket] - int(raw[bucket])), bucket),
    )
    # First bring each bucket toward its proportional floor, then distribute
    # any last units by the same deterministic ordering.
    for bucket in order:
        if remaining <= 0:
            break
        floor_target = min(counts[bucket], max(allocation[bucket], int(raw[bucket])))
        add = floor_target - allocation[bucket]
        if add > 0:
            add = min(add, remaining)
            allocation[bucket] += add
            remaining -= add
    cursor = 0
    while remaining:
        bucket = order[cursor % len(order)]
        if allocation[bucket] < counts[bucket] or quota > total:
            allocation[bucket] += 1
            remaining -= 1
        cursor += 1
        if cursor > quota * 10 + 100:
            raise RuntimeError("bucket allocation could not converge")
    return {bucket: allocation.get(bucket, 0) for bucket in EXPECTED_BUCKETS}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.resolve(strict=True).open(encoding="utf-8") if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"source row drift: {len(rows)} != {EXPECTED_ROWS}")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("source IDs must be non-empty and unique")
    routes = {str(row.get("rlv3_route", "")) for row in rows}
    if routes != set(ROUTES):
        raise ValueError(f"source route drift: {sorted(routes)}")
    for row in rows:
        audit = row.get("rlv3_audit")
        if not isinstance(audit, dict) or audit.get("length_bucket") not in EXPECTED_BUCKETS:
            raise ValueError(f"invalid length audit for {row['id']}")
        if int(audit.get("reference_tokens", 0)) <= 0:
            raise ValueError(f"invalid reference token count for {row['id']}")
    return rows


def build(rows: list[dict], *, seed: int) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["rlv3_route"]), str(row["rlv3_audit"]["length_bucket"]))].append(row)

    selected: list[dict] = []
    bucket_plan: dict[str, dict[str, int]] = {}
    repeat_count = 0
    dropped_ids: list[str] = []
    source_id_by_new_id: dict[str, str] = {}
    for route in ROUTES:
        counts = {bucket: len(grouped[(route, bucket)]) for bucket in EXPECTED_BUCKETS}
        bucket_plan[route] = allocate_buckets(counts, TARGET_QUOTAS[route])
        for bucket in EXPECTED_BUCKETS:
            pool = list(grouped[(route, bucket)])
            local_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{route}:{bucket}".encode()).digest()[:8], "big"
            )
            random.Random(local_seed).shuffle(pool)
            quota = bucket_plan[route][bucket]
            if quota <= len(pool):
                chosen = pool[:quota]
                dropped_ids.extend(str(row["id"]) for row in pool[quota:])
            else:
                chosen = list(pool)
                for index in range(quota - len(pool)):
                    original = pool[index % len(pool)]
                    repeated = copy.deepcopy(original)
                    repeat_count += 1
                    new_id = f"{original['id']}::rlv2r2rep{repeat_count:04d}"
                    repeated["id"] = new_id
                    repeated["rlv2_rebalance"] = {
                        "source_id": str(original["id"]),
                        "repeat_index": repeat_count,
                        "route": route,
                        "length_bucket": bucket,
                    }
                    chosen.append(repeated)
                    source_id_by_new_id[new_id] = str(original["id"])
            selected.extend(chosen)

    if len(selected) != EXPECTED_ROWS:
        raise RuntimeError(f"rebalance row conservation failed: {len(selected)}")
    random.Random(seed ^ 0x52564B32).shuffle(selected)
    route_rows = Counter(str(row["rlv3_route"]) for row in selected)
    route_tokens = Counter()
    bucket_rows = Counter()
    source_tiers = Counter()
    source_names = Counter()
    for row in selected:
        route = str(row["rlv3_route"])
        audit = row["rlv3_audit"]
        route_tokens[route] += int(audit["reference_tokens"])
        bucket_rows[f"{route}:{audit['length_bucket']}"] += 1
        source_tiers[str(audit.get("source_tier", "unknown"))] += 1
        source_names[str(audit.get("source_name", "unknown"))] += 1
    total_tokens = sum(route_tokens.values())
    audit = {
        "status": "PASS",
        "schema_version": 2,
        "policy": "modest_route_rebalance_fixed_total_17700_stratified_length",
        "seed": seed,
        "source_rows": len(rows),
        "effective_rows": len(selected),
        "source_sha256": sha256_lines(rows),
        "ordered_effective_ids_sha256": sha256_lines(selected),
        "route_target_rows": dict(TARGET_QUOTAS),
        "route_rows": dict(sorted(route_rows.items())),
        "route_reference_tokens": dict(sorted(route_tokens.items())),
        "route_token_share": {
            route: route_tokens[route] / total_tokens for route in sorted(route_tokens)
        },
        "route_length_bucket_rows": dict(sorted(bucket_rows.items())),
        "bucket_plan": bucket_plan,
        "provenance_rows": dict(sorted(source_tiers.items())),
        "source_names": dict(sorted(source_names.items())),
        "repeated_rows": repeat_count,
        "dropped_source_rows": len(dropped_ids),
        "dropped_source_id_sha256": hashlib.sha256("\n".join(sorted(dropped_ids)).encode()).hexdigest(),
        "repeat_ids_are_explicit": True,
        "optimizer_contract": {
            "rows": EXPECTED_ROWS,
            "groups_per_step": 7,
            "optimizer_steps": 2529,
            "save_steps": 281,
        },
        "intended_effect": (
            "reduce OCR/visual-prompt saturation modestly; increase independent "
            "dense, point, referring and GUI coverage; preserve within-route length mix"
        ),
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    rows = load_rows(args.source)
    selected, audit = build(rows, seed=args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "train.jsonl"
    audit_path = output / "audit.json"
    tmp_data = output / f".train.jsonl.tmp-{__import__('os').getpid()}"
    with tmp_data.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_data.replace(data_path)
    audit["data_sha256"] = sha256_file(data_path)
    tmp_audit = output / f".audit.json.tmp-{__import__('os').getpid()}"
    tmp_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_audit.replace(audit_path)
    print(json.dumps({"status": "PASS", "data": str(data_path), **audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
