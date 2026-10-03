#!/usr/bin/env python3
"""Fail-closed source-inventory contract shared by GAM pipeline stages.

The inventory, rather than a stage-local constant, owns the data-pool identity,
source order and collection cardinalities.  V1 inventories without explicit
cardinality declarations remain supported for reproducibility; every newer
inventory must declare and satisfy both cardinality fields.
"""

from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Tuple

import yaml


MANIFEST_TYPE = "gam_special_token_cleaning_sources"
POOL_RE = re.compile(
    r"^(?:special_token_data_V[1-9][0-9]*|general_data_V1)$"
)
SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COLLECTION_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
REJECTION_REASON_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
LEGACY_V1_COLLECTION_COUNTS = {"A": 11, "B": 25, "C": 9}
GAM_ROUTES = frozenset({
    "referring",
    "grounding",
    "dense",
    "dense_point",
    "refer_point",
    "gui",
    "ocr",
    "layout",
    "visual_prompt",
    "general_support",
})
DATA_BASE = Path(__file__).resolve().parents[2] / "data/train"


class ManifestContractError(RuntimeError):
    """Raised when an inventory does not form an immutable pipeline contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestContractError(message)


def _strict_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_honey(value: object) -> bool:
    if isinstance(value, str):
        return "honey" in value.casefold()
    if isinstance(value, Mapping):
        return any(_contains_honey(key) or _contains_honey(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_honey(item) for item in value)
    return False


def load_manifest_source_document(path: Path) -> Tuple[Path, Dict[str, Any]]:
    """Load a manifest and expand an integrity-pinned base inventory.

    A delta inventory keeps the already released V4 source definitions
    immutable and auditable while appending a new collection.  The child must
    pin the exact SHA256 of its base; source order is base first, child second.
    """

    resolved = path.resolve(strict=True)
    _require(
        resolved.is_file() and not path.is_symlink(),
        f"source manifest 必须为非符号链接文件: {resolved}",
    )

    def load_one(current: Path, stack: Tuple[Path, ...]) -> Dict[str, Any]:
        _require(current not in stack, f"source manifest base_manifest 循环: {current}")
        try:
            value = yaml.safe_load(current.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestContractError(
                f"无法读取 source manifest {current}: {exc}"
            ) from exc
        _require(isinstance(value, dict), "source manifest 顶层必须为 mapping")
        base_value = value.get("base_manifest")
        if base_value is None:
            return dict(value)
        _require(
            isinstance(base_value, str) and base_value,
            "base_manifest 必须为非空路径",
        )
        base_path = Path(base_value)
        if not base_path.is_absolute():
            base_path = current.parent / base_path
        base_path = base_path.resolve(strict=True)
        _require(base_path.is_file(), f"base_manifest 不是普通文件: {base_path}")
        expected_sha = value.get("base_manifest_sha256")
        _require(
            isinstance(expected_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None,
            "base_manifest_sha256 必须为 64 位小写 SHA256",
        )
        actual_sha = _sha256_file(base_path)
        _require(
            actual_sha == expected_sha,
            "base_manifest SHA256 漂移: "
            f"expected={expected_sha} actual={actual_sha}",
        )
        base = load_one(base_path, stack + (current,))
        for field in ("manifest_type", "pool_name", "coordinate_only"):
            if field in value:
                _require(
                    value[field] == base.get(field),
                    f"delta manifest 不得改变 base {field}",
                )
        child_sources = value.get("sources")
        _require(
            isinstance(child_sources, list) and child_sources,
            "delta manifest.sources 必须为非空 list",
        )
        merged = dict(base)
        merged.update(value)
        merged_defaults = dict(base.get("defaults") or {})
        merged_defaults.update(value.get("defaults") or {})
        merged["defaults"] = merged_defaults
        merged["sources"] = list(base.get("sources") or []) + list(child_sources)
        merged["_base_manifest_binding"] = {
            "path": str(base_path),
            "sha256": actual_sha,
        }
        return merged

    return resolved, load_one(resolved, ())


@dataclass(frozen=True)
class ManifestContract:
    path: Path
    raw: Dict[str, Any]
    pool_name: str
    inventory_role: str
    source_ids: Tuple[str, ...]
    collection_counts: Dict[str, int]
    route_counts: Dict[str, int]
    expected_profile_rejections: Dict[str, Dict[str, Any]]
    resolved_output_root: Path
    file_sha256: str
    content_sha256: str

    @property
    def source_count(self) -> int:
        return len(self.source_ids)

    @property
    def output_root(self) -> Path:
        return self.resolved_output_root

    def plan_binding(self) -> Dict[str, Any]:
        return {
            "data_pool": self.pool_name,
            "manifest": str(self.path),
            "manifest_sha256": self.file_sha256,
            "manifest_content_sha256": self.content_sha256,
            "source_count": self.source_count,
            "source_ids": list(self.source_ids),
            "collection_counts": dict(self.collection_counts),
        }


def normalize_expected_profile_rejections(
    value: object, *, source_name: str
) -> Dict[str, Any]:
    """Normalize the exact, manifest-declared full-profile reject contract.

    ``input_error_reason_counts`` covers iterator-level rejects.  The separate
    ``no_coordinate_payload`` counter covers rows that are valid source rows
    but intentionally excluded by a coordinate-only route.  Undeclared values
    default to zero; unknown fields and non-positive reason counts fail closed.
    """

    if value is None:
        value = {}
    _require(
        isinstance(value, Mapping),
        f"expected_profile_rejections 必须为 mapping: {source_name}",
    )
    allowed = {"input_error_reason_counts", "no_coordinate_payload"}
    unknown = set(value) - allowed
    _require(
        not unknown,
        f"expected_profile_rejections 含未知字段: {source_name}: {sorted(unknown)}",
    )
    raw_reasons = value.get("input_error_reason_counts") or {}
    _require(
        isinstance(raw_reasons, Mapping),
        f"input_error_reason_counts 必须为 mapping: {source_name}",
    )
    reasons: Dict[str, int] = {}
    for reason, count in raw_reasons.items():
        _require(
            isinstance(reason, str)
            and REJECTION_REASON_RE.fullmatch(reason) is not None,
            f"profile rejection reason 非法: {source_name}: {reason!r}",
        )
        _require(
            _strict_positive_int(count),
            f"profile rejection count 必须为正整数: {source_name}: {reason}",
        )
        reasons[reason] = count
    no_coordinate = value.get("no_coordinate_payload", 0)
    _require(
        type(no_coordinate) is int and no_coordinate >= 0,
        f"no_coordinate_payload 必须为非负整数: {source_name}",
    )
    return {
        "input_error_reason_counts": dict(sorted(reasons.items())),
        "no_coordinate_payload": no_coordinate,
    }


def require_profile_rejection_contract(
    profile_source: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source_name: str,
    require_complete_histogram: bool = True,
) -> None:
    """Require one full-profile source to match its exact manifest contract."""

    counts = profile_source.get("counts")
    _require(isinstance(counts, Mapping), f"profile counts 缺失: {source_name}")
    expected_normalized = normalize_expected_profile_rejections(
        expected, source_name=source_name
    )
    expected_reasons = expected_normalized["input_error_reason_counts"]

    has_complete_histogram = "input_error_reason_counts" in profile_source
    raw_actual_reasons = profile_source.get("input_error_reason_counts")
    if raw_actual_reasons is None:
        # Only callers that explicitly identify a frozen V1 artifact may use
        # the legacy schema.  V2+ must never downgrade by deleting fields.
        _require(
            not require_complete_histogram,
            f"V2+ profile 缺少完整 input_error_reason_counts: {source_name}",
        )
        _require(
            not expected_reasons
            and expected_normalized["no_coordinate_payload"] == 0,
            f"profile 缺少完整 input_error_reason_counts: {source_name}",
        )
        for label in (
            "input_errors",
            "parse_errors",
            "ambiguous_payload_task",
            "ref_mode_multi_label_violations",
        ):
            count = counts.get(label, 0)
            _require(
                type(count) is int and count == 0,
                f"legacy profile 存在未声明 {label}: {source_name}",
            )
        examples = profile_source.get("error_examples")
        _require(
            isinstance(examples, list) and not examples,
            f"legacy profile error_examples 非空/非法: {source_name}",
        )
        return
    _require(
        isinstance(raw_actual_reasons, Mapping),
        f"profile input_error_reason_counts 非法: {source_name}",
    )
    actual_reasons: Dict[str, int] = {}
    for reason, count in raw_actual_reasons.items():
        _require(
            isinstance(reason, str)
            and REJECTION_REASON_RE.fullmatch(reason) is not None
            and _strict_positive_int(count),
            f"profile input error histogram 非法: {source_name}: {reason!r}",
        )
        actual_reasons[reason] = count
    actual_reasons = dict(sorted(actual_reasons.items()))
    _require(
        actual_reasons == expected_reasons,
        "profile input error 与 manifest 精确合同不一致: "
        f"{source_name}: expected={expected_reasons} actual={actual_reasons}",
    )

    input_errors = counts.get("input_errors", 0)
    no_coordinate = counts.get("no_coordinate_payload", 0)
    parse_errors = counts.get("parse_errors", 0)
    ambiguous = counts.get("ambiguous_payload_task", 0)
    ref_violations = counts.get("ref_mode_multi_label_violations", 0)
    for label, count in (
        ("input_errors", input_errors),
        ("no_coordinate_payload", no_coordinate),
        ("parse_errors", parse_errors),
        ("ambiguous_payload_task", ambiguous),
        ("ref_mode_multi_label_violations", ref_violations),
    ):
        _require(
            type(count) is int and count >= 0,
            f"profile {label} 必须为非负整数: {source_name}",
        )
    _require(
        input_errors == sum(actual_reasons.values()),
        f"profile input_errors 与 reason histogram 不守恒: {source_name}",
    )
    _require(
        no_coordinate == expected_normalized["no_coordinate_payload"],
        "profile no_coordinate_payload 与 manifest 精确合同不一致: "
        f"{source_name}: expected={expected_normalized['no_coordinate_payload']} "
        f"actual={no_coordinate}",
    )
    _require(parse_errors == 0, f"profile 存在未声明 parse error: {source_name}")
    _require(ambiguous == 0, f"profile 存在 ambiguous payload: {source_name}")
    _require(ref_violations == 0, f"profile 存在 ref multi-label violation: {source_name}")

    examples = profile_source.get("error_examples")
    _require(isinstance(examples, list), f"profile error_examples 非法: {source_name}")
    _require(
        len(examples) == min(20, input_errors),
        f"profile error_examples 数量与 input_errors 不一致: {source_name}",
    )
    _require(
        all(
            isinstance(item, Mapping)
            and item.get("reason") in expected_reasons
            for item in examples
        ),
        f"profile error_examples 含未声明 reason: {source_name}",
    )

    input_records = counts.get("input_records")
    coordinate_records = counts.get("coordinate_payload_records")
    _require(has_complete_histogram, f"profile histogram 内部状态非法: {source_name}")
    _require(
        type(input_records) is int and input_records > 0,
        f"profile input_records 必须为正整数: {source_name}",
    )
    _require(
        type(coordinate_records) is int and coordinate_records >= 0,
        f"profile coordinate_payload_records 必须为非负整数: {source_name}",
    )
    _require(
        input_records == coordinate_records + input_errors + no_coordinate,
        f"profile input/coordinate/rejection 数量不守恒: {source_name}",
    )


def load_manifest_contract(path: Path) -> ManifestContract:
    resolved, raw = load_manifest_source_document(path)
    _require(type(raw.get("schema_version")) is int
             and raw["schema_version"] == 1,
             "source manifest schema_version 必须为整数 1")
    _require(raw.get("manifest_type") == MANIFEST_TYPE,
             "source manifest_type 不一致")
    pool = raw.get("pool_name")
    _require(isinstance(pool, str) and POOL_RE.fullmatch(pool) is not None,
             "source manifest pool_name 必须为 versioned GAM pool")
    inventory_role = raw.get("inventory_role", "full")
    _require(
        inventory_role in {"full", "cache_subset"},
        "source manifest inventory_role 必须为 full/cache_subset",
    )
    _require(
        inventory_role == "full" or pool == "special_token_data_V4",
        "cache_subset inventory 仅允许用于 special_token_data_V4",
    )
    output_root_value = raw.get("output_root")
    if output_root_value is None:
        resolved_output_root = (DATA_BASE / pool).resolve(strict=False)
    else:
        _require(
            pool == "general_data_V1",
            "仅 general_data_V1 inventory 允许显式 output_root revision",
        )
        _require(
            isinstance(output_root_value, str)
            and Path(output_root_value).is_absolute(),
            "general_data_V1 output_root 必须为绝对路径",
        )
        resolved_output_root = Path(output_root_value).resolve(strict=False)
        data_base = DATA_BASE.resolve(strict=False)
        _require(
            resolved_output_root.parent == data_base
            and re.fullmatch(
                r"general_data_V1_r[2-9][0-9]*",
                resolved_output_root.name,
            )
            is not None,
            "general_data_V1 output_root 必须是 DATA_BASE 下的 r2+ revision",
        )
        _require(
            not Path(output_root_value).is_symlink(),
            "general_data_V1 output_root 禁止为 symlink",
        )
    coordinate_only = raw.get("coordinate_only", False)
    _require(type(coordinate_only) is bool,
             "source manifest coordinate_only 必须为 boolean")
    if pool.startswith("special_token_data_") and pool != "special_token_data_V1":
        _require(coordinate_only is True,
                 "V2+ source manifest 必须启用 coordinate_only=true")
    defaults = raw.get("defaults") or {}
    _require(isinstance(defaults, dict), "source manifest.defaults 必须为 mapping")
    sources = raw.get("sources")
    _require(isinstance(sources, list) and sources,
             "source manifest.sources 必须为非空 list")
    names = []
    collections: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    expected_profile_rejections: Dict[str, Dict[str, Any]] = {}
    for index, source in enumerate(sources):
        _require(isinstance(source, dict), f"sources[{index}] 必须为 mapping")
        if _contains_honey(source) and pool != "general_data_V1":
            _require(
                pool == "special_token_data_V4"
                and source.get("route") == "general_support"
                and source.get("adapter") == "passthrough",
                "Honey source is forbidden: "
                f"仅允许作为 V4 general_support passthrough source: sources[{index}]",
            )
        name = source.get("name")
        collection = source.get("collection")
        _require(isinstance(name, str) and SOURCE_RE.fullmatch(name) is not None,
                 f"sources[{index}].name 非法")
        _require(name not in names, f"source name 重复: {name}")
        _require(isinstance(collection, str)
                 and COLLECTION_RE.fullmatch(collection) is not None,
                 f"source collection 非法: {name}")
        names.append(name)
        collections[collection] += 1
        route = source.get("route")
        if pool == "special_token_data_V4":
            _require(
                isinstance(route, str) and route in GAM_ROUTES,
                f"V4 source route 缺失或非法: {name}: {route!r}",
            )
            routes[route] += 1
        expected_profile_rejections[name] = normalize_expected_profile_rejections(
            source.get("expected_profile_rejections"), source_name=name
        )
        if coordinate_only:
            effective = dict(defaults)
            effective.update(source)
            if effective.get("coordinate_only") is False:
                _require(
                    pool == "special_token_data_V4"
                    and route == "general_support"
                    and effective.get("adapter") == "passthrough"
                    and effective.get("task") is None,
                    f"仅 V4 general_support passthrough source 可关闭 coordinate gate: {name}",
                )
            else:
                _require(effective.get("coordinate_only") is True,
                         f"coordinate-only source coordinate_only 必须为 boolean: {name}")
                _require(effective.get("task") in {"bbox", "point"},
                         f"coordinate-only source task 必须为 bbox/point: {name}")
                _require(effective.get("adapter") != "passthrough",
                         f"coordinate-only source 禁止 passthrough adapter: {name}")

    actual_counts = dict(sorted(collections.items()))
    actual_route_counts = dict(sorted(routes.items()))
    if pool == "special_token_data_V4" and inventory_role == "full":
        _require(
            set(actual_route_counts) == GAM_ROUTES,
            "V4 source manifest 必须覆盖九种空间 route 和 general_support: "
            f"missing={sorted(GAM_ROUTES - set(actual_route_counts))}",
        )
    declared_count = raw.get("expected_source_count")
    declared_collections = raw.get("expected_collection_counts")
    if pool == "special_token_data_V1" and declared_count is None \
            and declared_collections is None:
        declared_count = sum(LEGACY_V1_COLLECTION_COUNTS.values())
        declared_collections = LEGACY_V1_COLLECTION_COUNTS
    _require(_strict_positive_int(declared_count),
             "V2+ source manifest 必须声明 expected_source_count 正整数")
    _require(isinstance(declared_collections, dict) and declared_collections,
             "V2+ source manifest 必须声明 expected_collection_counts")
    normalized_declared: Dict[str, int] = {}
    for key, value in declared_collections.items():
        _require(isinstance(key, str) and COLLECTION_RE.fullmatch(key) is not None,
                 f"expected collection 名称非法: {key!r}")
        _require(_strict_positive_int(value),
                 f"expected collection 数量必须为正整数: {key}")
        normalized_declared[key] = value
    normalized_declared = dict(sorted(normalized_declared.items()))
    _require(
        len(names) == declared_count,
        "source manifest.sources 必须精确包含 "
        f"{declared_count} 个数据源: expected={declared_count} actual={len(names)}",
    )
    _require(actual_counts == normalized_declared,
             "source collection 数量与 expected_collection_counts 不一致: "
             f"expected={normalized_declared} actual={actual_counts}")

    return ManifestContract(
        path=resolved,
        raw=raw,
        pool_name=pool,
        inventory_role=inventory_role,
        source_ids=tuple(names),
        collection_counts=actual_counts,
        route_counts=actual_route_counts,
        expected_profile_rejections=expected_profile_rejections,
        resolved_output_root=resolved_output_root,
        file_sha256=_sha256_file(resolved),
        content_sha256=_canonical_sha256(raw),
    )


def require_contract_binding(
    plan: Mapping[str, Any], contract: ManifestContract, *, include_source_ids: bool = True
) -> None:
    """Reject any plan whose inventory identity/cardinalities have drifted."""

    expected = contract.plan_binding()
    keys = (
        "data_pool",
        "manifest",
        "manifest_sha256",
        "manifest_content_sha256",
        "source_count",
        "collection_counts",
    )
    if include_source_ids:
        keys += ("source_ids",)
    for key in keys:
        _require(plan.get(key) == expected[key],
                 f"plan manifest contract binding 漂移: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=("pool_name", "source_count", "output_root", "json"),
        default="json",
    )
    args = parser.parse_args()
    contract = load_manifest_contract(args.manifest)
    if args.field == "pool_name":
        print(contract.pool_name)
    elif args.field == "source_count":
        print(contract.source_count)
    elif args.field == "output_root":
        print(contract.output_root)
    else:
        print(json.dumps(contract.plan_binding(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
