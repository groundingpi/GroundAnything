"""Build the auditable, token-balanced Grounding/OCR RL V3 manifest.

The released GRPO data are complementary but heavily length-skewed:
Grounding contains no short targets, while OCR is dominated by short targets.
This builder balances three independent dimensions without replacement:

* rows per reward route (Grounding/OCR: 1:1);
* completion-reference token mass per route;
* short/medium/long rows inside each route (40/30/30).

Original RL rows always have first priority.  Missing buckets are filled from
the second-stage SFT source pool and only then from the first-stage SFT pool.
The output remains compatible with the existing GAM reward implementations;
the reward formula itself is not changed.
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


ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_JSON = ROOT / "weights/base_model/tokenizer.json"
SFT2_YAML = ROOT / "configs/train/sft2.yaml"
SFT1_YAML = ROOT / "configs/train/sft1.yaml"

ROUTES = ("grounding_bbox_gam", "ocr_bbox_text_gam")
BUCKET_TARGETS = {"short": 2_400, "medium": 1_800, "long": 1_800}
TARGET_PER_ROUTE = sum(BUCKET_TARGETS.values())
MAX_REFERENCE_TOKENS = 1_024
SEED = 20_260_831

GROUP_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.S,
)
BOX_RE = re.compile(
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
    r"<\s*(\d{1,4})\s*><\s*(\d{1,4})\s*>"
)


@dataclass(frozen=True)
class Candidate:
    route: str
    source_tier: str
    source_name: str
    row_id: str
    reference_tokens: int
    bucket: str
    row: dict[str, Any]


class TokenCounter:
    def __init__(self, tokenizer_json: Path = TOKENIZER_JSON) -> None:
        from tokenizers import Tokenizer

        self.tokenizer_path = tokenizer_json.resolve(strict=True)
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def _bucket(tokens: int) -> str:
    if tokens <= 64:
        return "short"
    if tokens <= 256:
        return "medium"
    if tokens <= MAX_REFERENCE_TOKENS:
        return "long"
    return "inadmissible"


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_box(box: list[int]) -> bool:
    return (
        len(box) == 4
        and 0 <= box[0] < box[2] <= 999
        and 0 <= box[1] < box[3] <= 999
    )


def _parse_completion(text: str) -> list[tuple[str, list[list[int]]]]:
    groups: list[tuple[str, list[list[int]]]] = []
    matches = list(GROUP_RE.finditer(str(text)))
    if not matches:
        raise ValueError("assistant completion has no GAM object/box group")
    for match in matches:
        label = match.group(1).strip()
        boxes = [
            [int(item.group(index)) for index in range(1, 5)]
            for item in BOX_RE.finditer(match.group(2))
        ]
        residual = BOX_RE.sub("", match.group(2))
        residual = re.sub(r"[\s,]", "", residual)
        if not label or not boxes or residual or any(not _valid_box(box) for box in boxes):
            raise ValueError("assistant completion contains an invalid GAM group")
        groups.append((label, boxes))
    outside = GROUP_RE.sub("", str(text))
    outside = re.sub(r"[\s,]", "", outside)
    if outside:
        raise ValueError("assistant completion contains content outside GAM groups")
    return groups


def _grounding_reference(row: dict[str, Any]) -> str:
    solution = json.loads(row["solution"])
    order: list[str] = []
    grouped: dict[str, list[list[int]]] = {}
    for item in solution["items"]:
        label = str(item["label"])
        if label not in grouped:
            order.append(label)
            grouped[label] = []
        box = [int(value) for value in item["bbox_2d"]]
        if not _valid_box(box):
            raise ValueError(f"invalid grounding box in row {row.get('id')}")
        grouped[label].append(box)
    pieces: list[str] = []
    for label in order:
        # Preserve the deterministic GAM bbox contract inside each label.
        boxes = sorted(grouped[label], key=lambda value: tuple(value))
        coordinates = ",".join("".join(f"<{value}>" for value in box) for box in boxes)
        pieces.append(
            f"<|object_ref_start|>{label}<|object_ref_end|>"
            f"<|box_start|>{coordinates}<|box_end|>"
        )
    return ", ".join(pieces)


def _ocr_reference(row: dict[str, Any]) -> str:
    solution = json.loads(row["solution"])
    references = solution.get("references") or []
    if not references:
        raise ValueError(f"OCR row has no references: {row.get('id')}")
    instances = references[0].get("instances") or []
    pieces: list[str] = []
    for item in instances:
        box = [int(value) for value in (item.get("bbox_norm1000") or item.get("bbox_2d"))]
        if not _valid_box(box):
            raise ValueError(f"invalid OCR box in row {row.get('id')}")
        coordinates = "".join(f"<{value}>" for value in box)
        pieces.append(
            f"<|object_ref_start|>{str(item.get('text', ''))}<|object_ref_end|>"
            f"<|box_start|>{coordinates}<|box_end|>"
        )
    if not pieces:
        raise ValueError(f"OCR row has an empty primary reference: {row.get('id')}")
    return ", ".join(pieces)


def _sft_images(row: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for value in row.get("images") or []:
        path = value.get("path") if isinstance(value, dict) else value
        if not path:
            raise ValueError("SFT row has an image without a path")
        output.append(str(path))
    return output


def _sft_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    users = [str(item["content"]) for item in row["messages"] if item["role"] == "user"]
    assistants = [str(item["content"]) for item in row["messages"] if item["role"] == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError("SFT fill row must have exactly one user and one assistant turn")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": users[0]},
    ], assistants[0]


def _grounding_from_sft(row: dict[str, Any], source_name: str, tier: str) -> tuple[dict[str, Any], str]:
    messages, completion = _sft_messages(row)
    groups = _parse_completion(completion)
    items: list[dict[str, Any]] = []
    target_labels: list[str] = []
    for label, boxes in groups:
        target_labels.append(label)
        for box in sorted(boxes, key=lambda value: tuple(value)):
            items.append({"bbox_2d": box, "label": label})
    identifier = f"{tier}:{source_name}:{row.get('id')}"
    output = {
        "id": identifier,
        "messages": messages,
        "images": _sft_images(row),
        "task_type": "grounding_bbox_gam",
        "target_labels": target_labels,
        "source_dataset": source_name,
        "source_index": str(row.get("id")),
        "solution": json.dumps(
            {
                "format": "gam_bbox_v1",
                "coord": "gam_norm999_xyxy_tokens",
                "items": items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "source": {"id": identifier, "provenance_tier": tier},
    }
    return output, completion


def _ocr_from_sft(row: dict[str, Any], source_name: str, tier: str) -> tuple[dict[str, Any], str]:
    messages, completion = _sft_messages(row)
    groups = _parse_completion(completion)
    instances: list[dict[str, Any]] = []
    for text, boxes in groups:
        for box in boxes:
            instances.append({"bbox_norm1000": box, "text": text})
    references = [
        {"name": name, "instances": instances}
        for name in ("ppocr", "rex")
    ]
    reference_groups = []
    for index, item in enumerate(instances):
        member = {"index": index, **item}
        reference_groups.append(
            {
                "type": "strong_one_to_one",
                "ppocr": [member],
                "rex": [member],
                "text_alternatives": [item["text"]],
                "bbox_alternatives_norm1000": [item["bbox_norm1000"]],
                "source_iou": 1.0,
                "text_similarity": 1.0,
            }
        )
    identifier = f"{tier}:{source_name}:{row.get('id')}"
    output = {
        "id": identifier,
        "messages": messages,
        "images": _sft_images(row),
        "task_type": "ocr_bbox_text_gam",
        "source_dataset": source_name,
        "source_index": str(row.get("id")),
        "solution": json.dumps(
            {
                "format": "gam_ocr_bbox_text_v1",
                "references": references,
                "reference_groups": reference_groups,
                "unmatched": {"ppocr": [], "rex": []},
                "coord": "gam_norm999_xyxy_tokens",
                "task_type": "ocr_bbox_text",
                "reward_mode": "reference_groups_v2",
                "reference_mode": "reference_groups_v2",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "source": {"id": identifier, "provenance_tier": tier},
    }
    return output, completion


def _candidate(
    row: dict[str, Any],
    reference: str,
    route: str,
    source_tier: str,
    source_name: str,
    counter: TokenCounter,
) -> Candidate:
    tokens = counter(reference)
    return Candidate(
        route=route,
        source_tier=source_tier,
        source_name=source_name,
        row_id=str(row["id"]),
        reference_tokens=tokens,
        bucket=_bucket(tokens),
        row=row,
    )


def _load_rl(counter: TokenCounter) -> list[Candidate]:
    output: list[Candidate] = []
    for row in _read_jsonl(GROUNDING_DATA):
        output.append(
            _candidate(
                row,
                _grounding_reference(row),
                "grounding_bbox_gam",
                "rl",
                "lvis_grpo_gt20",
                counter,
            )
        )
    for row in _read_jsonl(OCR_DATA):
        output.append(
            _candidate(
                row,
                _ocr_reference(row),
                "ocr_bbox_text_gam",
                "rl",
                "ocr_grpo_mixed_reference",
                counter,
            )
        )
    return output


def _load_sft_yaml(path: Path, tier: str, counter: TokenCounter) -> tuple[list[Candidate], dict[str, Any]]:
    from datasets import load_from_disk

    payload = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    output: list[Candidate] = []
    audit: dict[str, Any] = {"yaml": str(path.resolve()), "sources": {}, "invalid_rows": 0}
    grounding_routes = {"bbox_grounding"} if tier == "sft2" else {"grounding", "dense", "referring"}
    for entry in payload["datasets"]:
        raw_route = str(entry.get("route", ""))
        if raw_route not in grounding_routes | {"ocr"}:
            continue
        source_name = str(entry.get("source_id") or entry["name"])
        source_path = Path(entry["path"])
        if not source_path.exists():
            audit["sources"][source_name] = {"status": "MISSING", "path": str(source_path)}
            continue
        dataset = load_from_disk(str(source_path))
        accepted = 0
        invalid = 0
        for row in dataset:
            try:
                if raw_route == "ocr":
                    converted, reference = _ocr_from_sft(row, source_name, tier)
                    route = "ocr_bbox_text_gam"
                else:
                    converted, reference = _grounding_from_sft(row, source_name, tier)
                    route = "grounding_bbox_gam"
                output.append(
                    _candidate(converted, reference, route, tier, source_name, counter)
                )
                accepted += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
        audit["invalid_rows"] += invalid
        audit["sources"][source_name] = {
            "status": "LOADED",
            "path": str(source_path.resolve()),
            "rows": len(dataset),
            "accepted": accepted,
            "invalid": invalid,
        }
    return output, audit


def _stable_shuffle(values: list[Candidate], seed: int, salt: str) -> list[Candidate]:
    output = list(values)
    rng = random.Random(int(seed) ^ int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16))
    rng.shuffle(output)
    return output


def _choose(
    pools: dict[tuple[str, str, str], list[Candidate]],
    route: str,
    bucket: str,
    count: int,
    seed: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for tier in ("rl", "sft2", "sft1"):
        available = _stable_shuffle(pools[(route, bucket, tier)], seed, f"{route}:{bucket}:{tier}")
        take = min(count - len(selected), len(available))
        selected.extend(available[:take])
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"insufficient candidates route={route} bucket={bucket}: {len(selected)} < {count}"
    )


def _interleave(rows: list[Candidate], seed: int) -> list[Candidate]:
    # Shuffle inside every route/bucket, then use a deficit scheduler so no
    # long prefix is monopolized by one route or one length bucket.
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for row in rows:
        groups[(row.route, row.bucket)].append(row)
    for key, values in groups.items():
        groups[key] = _stable_shuffle(values, seed, ":".join(key))
    order = sorted(groups)
    total = len(rows)
    target = {key: len(values) / total for key, values in groups.items()}
    emitted = Counter()
    output: list[Candidate] = []
    for index in range(total):
        choices = [key for key in order if groups[key]]
        key = max(
            choices,
            key=lambda item: ((index + 1) * target[item] - emitted[item], tuple(item)),
        )
        output.append(groups[key].pop())
        emitted[key] += 1
    return output


def build(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter()
    rl = _load_rl(counter)
    sft2, sft2_audit = _load_sft_yaml(SFT2_YAML, "sft2", counter)
    candidates = rl + sft2
    pools: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    inadmissible = Counter()
    for item in candidates:
        if item.bucket == "inadmissible":
            inadmissible[(item.route, item.source_tier)] += 1
            continue
        pools[(item.route, item.bucket, item.source_tier)].append(item)

    # The current targets are fully covered by RL + SFT2.  SFT1 is loaded only
    # if a future source drift creates a real deficit, avoiding a multi-million
    # row scan on every reproducibility check.
    need_sft1 = any(
        sum(len(pools[(route, bucket, tier)]) for tier in ("rl", "sft2")) < count
        for route in ROUTES
        for bucket, count in BUCKET_TARGETS.items()
    )
    sft1_audit: dict[str, Any] = {"loaded": False, "reason": "not_needed"}
    if need_sft1:
        sft1, raw_audit = _load_sft_yaml(SFT1_YAML, "sft1", counter)
        for item in sft1:
            if item.bucket == "inadmissible":
                inadmissible[(item.route, item.source_tier)] += 1
            else:
                pools[(item.route, item.bucket, item.source_tier)].append(item)
        sft1_audit = {"loaded": True, **raw_audit}

    selected: list[Candidate] = []
    for route in ROUTES:
        for bucket, count in BUCKET_TARGETS.items():
            selected.extend(_choose(pools, route, bucket, count, seed))

    if len(selected) != 2 * TARGET_PER_ROUTE:
        raise RuntimeError("balanced RL row count drift")
    ids = [item.row_id for item in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("balanced RL output contains duplicate row ids")

    ordered = _interleave(selected, seed)
    train_path = output_dir / "train.jsonl"
    temporary = train_path.with_name(f".{train_path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as stream:
        for item in ordered:
            row = dict(item.row)
            row["rlv3_audit"] = {
                "source_tier": item.source_tier,
                "source_name": item.source_name,
                "reference_tokens": item.reference_tokens,
                "length_bucket": item.bucket,
            }
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            stream.write(line + "\n")
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    temporary.replace(train_path)

    route_rows = Counter(item.route for item in selected)
    route_tokens = Counter()
    route_bucket_rows = Counter()
    route_bucket_tokens = Counter()
    tier_rows = Counter()
    source_rows = Counter()
    for item in selected:
        route_tokens[item.route] += item.reference_tokens
        route_bucket_rows[(item.route, item.bucket)] += 1
        route_bucket_tokens[(item.route, item.bucket)] += item.reference_tokens
        tier_rows[(item.route, item.source_tier)] += 1
        source_rows[(item.route, item.source_tier, item.source_name)] += 1
    total_tokens = sum(route_tokens.values())
    audit = {
        "schema_version": 1,
        "policy": "route_rows_1to1_reference_token_balanced_per_route_length_40_30_30",
        "seed": int(seed),
        "train_jsonl": str(train_path.resolve()),
        "train_jsonl_sha256": digest.hexdigest(),
        "ordered_ids_sha256": _sha256_lines(item.row_id for item in ordered),
        "tokenizer_json": str(counter.tokenizer_path),
        "tokenizer_json_sha256": hashlib.sha256(counter.tokenizer_path.read_bytes()).hexdigest(),
        "max_reference_tokens": MAX_REFERENCE_TOKENS,
        "rows": len(selected),
        "reference_tokens": total_tokens,
        "route_rows": dict(route_rows),
        "route_row_share": {key: value / len(selected) for key, value in route_rows.items()},
        "route_reference_tokens": dict(route_tokens),
        "route_reference_token_share": {
            key: value / total_tokens for key, value in route_tokens.items()
        },
        "route_bucket_rows": {
            f"{route}/{bucket}": route_bucket_rows[(route, bucket)]
            for route in ROUTES
            for bucket in BUCKET_TARGETS
        },
        "route_bucket_reference_tokens": {
            f"{route}/{bucket}": route_bucket_tokens[(route, bucket)]
            for route in ROUTES
            for bucket in BUCKET_TARGETS
        },
        "route_tier_rows": {
            f"{route}/{tier}": tier_rows[(route, tier)]
            for route in ROUTES
            for tier in ("rl", "sft2", "sft1")
        },
        "route_tier_source_rows": {
            "/".join(key): value for key, value in sorted(source_rows.items())
        },
        "inadmissible_over_1024": {
            "/".join(key): value for key, value in sorted(inadmissible.items())
        },
        "candidate_pool_rows": {
            "/".join(key): len(value) for key, value in sorted(pools.items())
        },
        "sft2": sft2_audit,
        "sft1": sft1_audit,
        "reward_contract": {
            "grounding": "0.7*RexOmni_Eq4 + 0.3*strict_IoU",
            "ocr": "GAM_reference_groups_v2_mixed_reward",
            "formula_changed": False,
            "sft_fill_reference_note": (
                "SFT fills are converted into the existing reward schemas; OCR uses two "
                "identical teacher views derived from the SFT ground truth."
            ),
        },
    }
    audit["audit_sha256"] = _json_sha256(audit)
    audit_path = output_dir / "audit.json"
    temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/rl/balanced",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
