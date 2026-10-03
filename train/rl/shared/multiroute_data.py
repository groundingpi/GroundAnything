"""Build the approved 17.7K, eleven-route GAM RL V3 dataset.

This is intentionally independent from ``balanced_data.py``.  That file is
the superseded two-route experiment; changing it would make the V1/V2 data
lineage ambiguous.  Selection here is without replacement and follows the
strict provenance order requested for V3: released RL data, then SFT stage 2,
then SFT stage 1 only for the remaining Robo Point deficit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import yaml

from train.rl.data import GROUNDING_DATA, OCR_DATA, _read_jsonl
from train.rl.shared.balanced_data import (
    MAX_REFERENCE_TOKENS,
    SEED,
    SFT1_YAML,
    SFT2_YAML,
    TokenCounter,
    _bucket,
    _grounding_from_sft,
    _grounding_reference,
    _ocr_from_sft,
    _ocr_reference,
    _sft_images,
    _sft_messages,
)
from train.rl.shared.route_schema import COARSE_ROUTE, ROUTES, SOURCE_TO_ROUTE


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "data/rl/multiroute"

BBOX_ROUTES = frozenset({"grounding", "referring", "dense", "visual_prompt", "layout"})
POINT_ROUTES = frozenset(
    {"grounding_point", "referring_point", "dense_point", "robo_point", "gui"}
)
OCR_ROUTES = frozenset({"ocr"})

SFT1_ROBO_SOURCES = frozenset(
    {
        "d_roboafford_affordance_point",
        "d_robointer_droid_contact_point",
        "d_robointer_rh20t_contact_point",
        "d_robointer_droid_traj_with_init_point",
        "d_robointer_rh20t_traj_with_init_point",
        "d_robointer_droid_traj_without_init_point",
        "d_robointer_rh20t_traj_without_init_point",
    }
)

GROUP_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.S,
)
COORD_RE = re.compile(r"<\s*(\d{1,4})\s*>")


@dataclass(frozen=True)
class Candidate:
    route: str
    source_tier: str
    source_name: str
    row_id: str
    reference_tokens: int
    bucket: str
    row: dict[str, Any]


def _quota() -> dict[tuple[str, str, str, str], int]:
    """Return the user-approved exact source/length allocation."""

    result: dict[tuple[str, str, str, str], int] = {}

    def add(route: str, tier: str, source: str, **buckets: int) -> None:
        for bucket, count in buckets.items():
            if count:
                result[(route, tier, source, bucket)] = int(count)

    # Grounding: 2,150 native RL rows plus 300 short SFT2 rows.
    add("grounding", "rl", "lvis_grpo_gt20", medium=1_400, long=750)
    add("grounding", "sft2", "gam_lvis", short=150)
    add("grounding", "sft2", "gam_coco", short=150)

    # Referring: exactly 50% of the approved first draft, preserving every source.
    for source, count in (
        ("gam_refcoco", 425),
        ("gam_refcocog", 325),
        ("gam_refcocoplus", 325),
        ("gam_refcocog_test", 300),
        ("gam_refcocog_val", 250),
    ):
        add("referring", "sft2", source, short=count)
    add("referring", "sft2", "gam_humanref", short=261, medium=64)

    # Dense is deliberately raised above 1K without replacement.  Consume all
    # admissible Dense200 rows, then use a controlled VisDrone length mix rather
    # than exhausting its long bucket.
    add("dense", "sft2", "gam_dense200", short=1, medium=67, long=121)
    add("dense", "sft2", "gam_visdrone", short=113, medium=583, long=315)

    add(
        "grounding_point",
        "sft2",
        "gam_rex_point_lvis",
        short=350,
        medium=350,
        long=100,
    )
    add("grounding_point", "sft2", "gam_rex_point_coco", short=500)

    add("referring_point", "sft2", "gam_rex_point_refcocog_test", short=700)
    add("referring_point", "sft2", "gam_rex_point_refcocog_val", short=500)
    add(
        "referring_point",
        "sft2",
        "gam_rex_point_humanref",
        short=737,
        medium=13,
    )

    add(
        "dense_point",
        "sft2",
        "gam_rex_point_dense200",
        short=8,
        medium=72,
        long=70,
    )
    add(
        "dense_point",
        "sft2",
        "gam_rex_point_visdrone",
        short=222,
        medium=228,
        long=50,
    )

    # Keep every unique SFT2 Robo row; fill only its deficit from SFT1.
    for source, short, medium in (
        ("gam_robospatial_context", 41, 81),
        ("gam_refspatial_location", 99, 0),
        ("gam_refspatial_placement", 100, 0),
        ("gam_refspatial_unseen", 76, 0),
    ):
        add("robo_point", "sft2", source, short=short, medium=medium)
    for source, short, medium, long in (
        ("d_roboafford_affordance_point", 150, 39, 11),
        ("d_robointer_droid_contact_point", 75, 0, 0),
        ("d_robointer_rh20t_contact_point", 60, 40, 0),
        ("d_robointer_droid_traj_with_init_point", 60, 15, 0),
        ("d_robointer_rh20t_traj_with_init_point", 46, 30, 0),
        ("d_robointer_droid_traj_without_init_point", 48, 2, 0),
        ("d_robointer_rh20t_traj_without_init_point", 47, 30, 0),
    ):
        add("robo_point", "sft1", source, short=short, medium=medium, long=long)

    add(
        "visual_prompt",
        "sft2",
        "gam_visual_dense200",
        short=1,
        medium=54,
        long=75,
    )
    add("visual_prompt", "sft2", "gam_fsc147", short=50, medium=175, long=75)
    add("visual_prompt", "sft2", "gam_visual_lvis", short=540, medium=65, long=15)
    add("visual_prompt", "sft2", "gam_visual_coco", short=700, medium=50)

    add("gui", "sft2", "gam_screenspot_pro", short=782)
    add("gui", "sft2", "gam_screenspot_v2", short=636)
    add("gui", "sft2", "gam_osworld_g", short=282)

    add("layout", "sft2", "gam_doclaynet", short=200, medium=300)
    add("layout", "sft2", "gam_m6doc", medium=180, long=120)

    # OCR is exactly half of the previous 5,700-row route.  Every source and
    # each aggregate length bucket are retained at a deterministic 50% split.
    add(
        "ocr",
        "rl",
        "ocr_mixed4583",
        short=1_541,
        medium=396,
        long=355,
    )
    add(
        "ocr",
        "rl",
        "ocr_totaltext_recovery917",
        short=131,
        medium=317,
        long=10,
    )
    add("ocr", "sft2", "gam_icdar2015", short=25)
    add("ocr", "sft2", "gam_totaltext", short=20, medium=5)
    add("ocr", "sft2", "gam_hiertext", medium=15, long=10)
    add("ocr", "sft2", "gam_sroie", long=25)
    return result


QUOTA = _quota()
EXPECTED_ROWS = sum(QUOTA.values())
assert EXPECTED_ROWS == 17_700


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_groups(text: str, arity: int) -> list[tuple[str, list[list[int]]]]:
    matches = list(GROUP_RE.finditer(str(text)))
    if not matches:
        raise ValueError("assistant completion has no GAM groups")
    output: list[tuple[str, list[list[int]]]] = []
    for match in matches:
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        values = [int(item.group(1)) for item in COORD_RE.finditer(match.group(2))]
        residual = COORD_RE.sub("", match.group(2))
        residual = re.sub(r"[\s,]", "", residual)
        if not label or residual or not values or len(values) % arity:
            raise ValueError("invalid GAM group structure")
        records = [values[index : index + arity] for index in range(0, len(values), arity)]
        if any(any(value < 0 or value > 999 for value in record) for record in records):
            raise ValueError("coordinate outside GAM 0..999 range")
        if arity == 4 and any(
            not (record[0] < record[2] and record[1] < record[3])
            for record in records
        ):
            raise ValueError("invalid xyxy box")
        output.append((label, sorted(records, key=tuple)))
    outside = GROUP_RE.sub("", str(text))
    if re.sub(r"[\s,]", "", outside):
        raise ValueError("content outside GAM groups")
    return output


def _point_from_sft(
    row: dict[str, Any], source_name: str, tier: str
) -> tuple[dict[str, Any], str]:
    messages, completion = _sft_messages(row)
    groups = _strict_groups(completion, 2)
    items: list[dict[str, Any]] = []
    labels: list[str] = []
    for label, points in groups:
        labels.append(label)
        items.extend({"point_2d": point, "label": label} for point in points)
    identifier = f"{tier}:{source_name}:{row.get('id')}"
    return (
        {
            "id": identifier,
            "messages": messages,
            "images": _sft_images(row),
            "task_type": "point_gam",
            "target_labels": labels,
            "source_dataset": source_name,
            "source_index": str(row.get("id")),
            "solution": json.dumps(
                {
                    "format": "gam_point_v1",
                    "coord": "gam_norm999_xy_tokens",
                    "items": items,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "source": {"id": identifier, "provenance_tier": tier},
        },
        completion,
    )


def reference_completion(row: dict[str, Any]) -> str:
    """Render the canonical answer used only for token/audit calculations."""

    task = str(row["task_type"])
    if task == "grounding_bbox_gam":
        return _grounding_reference(row)
    if task == "ocr_bbox_text_gam":
        return _ocr_reference(row)
    if task != "point_gam":
        raise ValueError(f"unsupported task type: {task}")
    solution = json.loads(row["solution"])
    grouped: dict[str, list[list[int]]] = defaultdict(list)
    order: list[str] = []
    for item in solution["items"]:
        label = str(item["label"])
        if label not in grouped:
            order.append(label)
        grouped[label].append([int(value) for value in item["point_2d"]])
    pieces: list[str] = []
    for label in order:
        points = sorted(grouped[label], key=tuple)
        coordinates = ",".join("".join(f"<{value}>" for value in point) for point in points)
        pieces.append(
            f"<|object_ref_start|>{label}<|object_ref_end|>"
            f"<|box_start|>{coordinates}<|box_end|>"
        )
    return ", ".join(pieces)


def _canonical_grounding_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize only bbox order; preserve the exact labels and coordinates."""

    row = dict(raw)
    solution = json.loads(row["solution"])
    order: list[str] = []
    grouped: dict[str, list[list[int]]] = defaultdict(list)
    for item in solution.get("items") or []:
        label = str(item["label"])
        if label not in grouped:
            order.append(label)
        grouped[label].append([int(value) for value in item["bbox_2d"]])
    if not grouped:
        raise ValueError("empty native grounding solution")
    items: list[dict[str, Any]] = []
    for label in order:
        for box in sorted(grouped[label], key=tuple):
            items.append({"bbox_2d": box, "label": label})
    normalized = dict(solution)
    normalized["items"] = items
    normalized["bbox_norm1000"] = {
        label: sorted(grouped[label], key=tuple) for label in order
    }
    normalized["bbox_count"] = {label: len(grouped[label]) for label in order}
    normalized["target_labels"] = order
    normalized["total_boxes"] = len(items)
    row["target_labels"] = order
    row["solution"] = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    )
    return row


def _candidate(
    row: dict[str, Any],
    reference: str,
    route: str,
    tier: str,
    source: str,
    counter: TokenCounter,
) -> Candidate:
    output = dict(row)
    output["rlv3_route"] = route
    output["id"] = f"{tier}:{source}:{row['id']}" if not str(row["id"]).startswith(f"{tier}:{source}:") else str(row["id"])
    tokens = counter(reference)
    return Candidate(
        route=route,
        source_tier=tier,
        source_name=source,
        row_id=str(output["id"]),
        reference_tokens=tokens,
        bucket=_bucket(tokens),
        row=output,
    )


def _load_rl(counter: TokenCounter) -> list[Candidate]:
    output: list[Candidate] = []
    for raw in _read_jsonl(GROUNDING_DATA):
        row = _canonical_grounding_row(raw)
        output.append(
            _candidate(
                row,
                _grounding_reference(row),
                "grounding",
                "rl",
                "lvis_grpo_gt20",
                counter,
            )
        )
    for raw in _read_jsonl(OCR_DATA):
        row = dict(raw)
        path = Path(str((row.get("images") or [""])[0]))
        source = (
            "ocr_mixed4583"
            if path.parts[-3] == "final5k_reference_groups_v2_ckpt2400_materialized"
            else "ocr_totaltext_recovery917"
        )
        output.append(
            _candidate(row, _ocr_reference(row), "ocr", "rl", source, counter)
        )
    return output


def _load_sft(counter: TokenCounter, yaml_path: Path, tier: str) -> tuple[list[Candidate], dict[str, Any]]:
    from datasets import load_from_disk

    needed = {key[2] for key in QUOTA if key[1] == tier}
    payload = yaml.safe_load(yaml_path.resolve(strict=True).read_text(encoding="utf-8"))
    output: list[Candidate] = []
    report: dict[str, Any] = {"yaml": str(yaml_path.resolve()), "sources": {}}
    found: set[str] = set()
    for entry in payload["datasets"]:
        source = str(entry.get("source_id") or entry["name"])
        if source not in needed:
            continue
        found.add(source)
        route = SOURCE_TO_ROUTE[source] if tier == "sft2" else "robo_point"
        if tier == "sft1" and source not in SFT1_ROBO_SOURCES:
            raise ValueError(f"unapproved SFT1 fallback source: {source}")
        dataset = load_from_disk(str(Path(entry["path"]).resolve(strict=True)))
        scan_rows = len(dataset)
        if tier == "sft1":
            scan_rows = min(scan_rows, int(entry.get("sample_count") or scan_rows))
        invalid = 0
        accepted = 0
        for index in range(scan_rows):
            raw = dataset[index]
            try:
                if route in BBOX_ROUTES:
                    converted, reference = _grounding_from_sft(raw, source, tier)
                elif route in POINT_ROUTES:
                    converted, reference = _point_from_sft(raw, source, tier)
                elif route in OCR_ROUTES:
                    converted, reference = _ocr_from_sft(raw, source, tier)
                else:
                    raise ValueError(f"unsupported semantic route: {route}")
                # Admission and length balancing must use the exact canonical
                # answer implied by the final reward solution, not incidental
                # whitespace/grouping in the source SFT assistant string.
                reference = reference_completion(converted)
                output.append(
                    _candidate(converted, reference, route, tier, source, counter)
                )
                accepted += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
        report["sources"][source] = {
            "path": str(Path(entry["path"]).resolve()),
            "dataset_rows": len(dataset),
            "scanned_rows": scan_rows,
            "accepted_rows": accepted,
            "invalid_rows": invalid,
        }
    missing = sorted(needed - found)
    if missing:
        raise RuntimeError(f"configured sources absent from {yaml_path}: {missing}")
    return output, report


def _stable_sample(values: list[Candidate], count: int, seed: int, salt: str) -> list[Candidate]:
    if len(values) < count:
        raise RuntimeError(f"candidate deficit {salt}: {len(values)} < {count}")
    rng_seed = int(seed) ^ int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    return random.Random(rng_seed).sample(values, count)


def _interleave(rows: list[Candidate], seed: int) -> list[Candidate]:
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for row in rows:
        groups[(row.route, row.bucket)].append(row)
    for key, values in groups.items():
        rng_seed = int(seed) ^ int(hashlib.sha256(":".join(key).encode()).hexdigest()[:16], 16)
        random.Random(rng_seed).shuffle(values)
    total = len(rows)
    targets = {key: len(values) / total for key, values in groups.items()}
    emitted: Counter[tuple[str, str]] = Counter()
    output: list[Candidate] = []
    for index in range(total):
        available = [key for key in sorted(groups) if groups[key]]
        key = max(
            available,
            key=lambda candidate: (index + 1) * targets[candidate] - emitted[candidate],
        )
        output.append(groups[key].pop())
        emitted[key] += 1
    return output


def build(output_dir: Path = OUTPUT_DIR, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter()
    rl = _load_rl(counter)
    sft2, sft2_report = _load_sft(counter, SFT2_YAML, "sft2")
    sft1, sft1_report = _load_sft(counter, SFT1_YAML, "sft1")
    pools: dict[tuple[str, str, str, str], list[Candidate]] = defaultdict(list)
    inadmissible = Counter()
    for item in rl + sft2 + sft1:
        if item.bucket == "inadmissible":
            inadmissible[(item.route, item.source_tier, item.source_name)] += 1
        else:
            pools[(item.route, item.source_tier, item.source_name, item.bucket)].append(item)

    selected: list[Candidate] = []
    capacity: dict[str, int] = {}
    for key, count in sorted(QUOTA.items()):
        salt = ":".join(key)
        capacity[salt] = len(pools[key])
        selected.extend(_stable_sample(pools[key], count, seed, salt))
    if len(selected) != EXPECTED_ROWS:
        raise RuntimeError("selected row count drift")
    if len({item.row_id for item in selected}) != len(selected):
        raise RuntimeError("duplicate selected row IDs")

    ordered = _interleave(selected, seed)
    train_path = output_dir / "train.jsonl"
    temporary = train_path.with_name(f".{train_path.name}.tmp")
    file_digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as stream:
        for item in ordered:
            row = dict(item.row)
            row["rlv3_audit"] = {
                "source_tier": item.source_tier,
                "source_name": item.source_name,
                "semantic_route": item.route,
                "coarse_route": COARSE_ROUTE[item.route],
                "reference_tokens": item.reference_tokens,
                "length_bucket": item.bucket,
            }
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            stream.write(line + "\n")
            file_digest.update(line.encode("utf-8"))
            file_digest.update(b"\n")
    temporary.replace(train_path)

    route_rows = Counter(item.route for item in selected)
    route_tokens = Counter()
    route_buckets = Counter()
    tier_rows = Counter()
    tier_tokens = Counter()
    source_rows = Counter()
    for item in selected:
        route_tokens[item.route] += item.reference_tokens
        route_buckets[(item.route, item.bucket)] += 1
        tier_rows[item.source_tier] += 1
        tier_tokens[item.source_tier] += item.reference_tokens
        source_rows[(item.route, item.source_tier, item.source_name)] += 1
    total_tokens = sum(route_tokens.values())
    audit: dict[str, Any] = {
        "schema_version": 2,
        "status": "BUILT_PENDING_INDEPENDENT_AUDIT",
        "policy": "RL-first/SFT2-second/SFT1-last; 11 semantic routes; no replacement",
        "seed": int(seed),
        "rows": len(selected),
        "reference_tokens": total_tokens,
        "train_jsonl": str(train_path.resolve()),
        "train_jsonl_sha256": file_digest.hexdigest(),
        "ordered_ids_sha256": _sha256_lines(item.row_id for item in ordered),
        "tokenizer_json": str(counter.tokenizer_path),
        "tokenizer_json_sha256": hashlib.sha256(counter.tokenizer_path.read_bytes()).hexdigest(),
        "max_reference_tokens": MAX_REFERENCE_TOKENS,
        "route_rows": dict(route_rows),
        "route_row_share": {route: route_rows[route] / len(selected) for route in ROUTES},
        "route_reference_tokens": dict(route_tokens),
        "route_reference_token_share": {
            route: route_tokens[route] / total_tokens for route in ROUTES
        },
        "route_bucket_rows": {
            f"{route}/{bucket}": route_buckets[(route, bucket)]
            for route in ROUTES
            for bucket in ("short", "medium", "long")
        },
        "tier_rows": dict(tier_rows),
        "tier_row_share": {tier: tier_rows[tier] / len(selected) for tier in tier_rows},
        "tier_reference_tokens": dict(tier_tokens),
        "tier_reference_token_share": {
            tier: tier_tokens[tier] / total_tokens for tier in tier_tokens
        },
        "route_tier_source_rows": {
            "/".join(key): value for key, value in sorted(source_rows.items())
        },
        "selection_quota": {"/".join(key): value for key, value in sorted(QUOTA.items())},
        "selection_capacity": capacity,
        "inadmissible_over_1024": {
            "/".join(key): value for key, value in sorted(inadmissible.items())
        },
        "sft2": sft2_report,
        "sft1": sft1_report,
        "reward_contract": {
            "bbox_routes": "released 0.7*RexOmni_Eq4 + 0.3*strict_IoU",
            "ocr": "released GAM_reference_groups_v2_mixed_reward",
            "point_routes": "new isolated V3 point-set reward; audited separately",
            "v1_v2_modified": False,
        },
    }
    audit["audit_sha256"] = _json_sha256(audit)
    audit_path = output_dir / "build.json"
    temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
