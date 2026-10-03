#!/usr/bin/env python3
"""RLV2 with a modest route rebalance and route×length telemetry.

The optimizer, rollout, reward formula, topology, and config validator are
the released RLV2 implementation. This module only swaps the isolated data
mixture and adds read-only telemetry after reward gathering; it deliberately
does not alter rewards, advantages, or gradients.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("This RL branch is disabled. Use configs/release/rl_train.yaml.")

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from train.rl.loss import group_normalized_advantage
from train.rl.reward_adapter import RewardResult, gdpo_total_rewards
from train.rl.current import trainer as legacy
from train.rl.branches.rebalanced.data import DATA, EXPECTED_ROWS, RebalancedRLV2Mixture
from train.rl.distributed import multiroute_56 as base
from train.rl.shared.reward import MultiRouteGAMRewardAdapter


GROUP_SIZE = 8
GROUPS_PER_STEP = 7
SAVE_STEPS = 281
TOTAL_STEPS = 2529

_CURRENT_META: dict[str, Any] = {}
_LAST_TELEMETRY: dict[str, Any] = {}
_ORIGINAL_APPEND = legacy.append_jsonl


class TelemetryRewardAdapter:
    """Delegate scoring byte-for-byte while retaining row metadata locally."""

    def __init__(self) -> None:
        self.inner = MultiRouteGAMRewardAdapter()

    def score(self, response: str, row: dict[str, Any]) -> RewardResult:
        global _CURRENT_META
        result = self.inner.score(response, row)
        audit = row.get("rlv3_audit") or {}
        _CURRENT_META = {
            "id": str(row.get("id", "")),
            "route": str(row.get("rlv3_route", "unknown")),
            "length_bucket": str(audit.get("length_bucket", "unknown")),
            "reference_tokens": int(audit.get("reference_tokens", 0)),
            "source_tier": str(audit.get("source_tier", "unknown")),
        }
        return result


def _aggregate_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Each generation group contributes eight rank copies of one prompt.
    # Deduplicate by ID so counts represent rollout groups, not replicas.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("id", ""))
        if key and key not in unique:
            unique[key] = row
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "reward_sum": 0.0, "reference_tokens": 0.0}
    )
    routes: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "reward_sum": 0.0, "reference_tokens": 0.0}
    )
    for row in unique.values():
        route = str(row.get("route", "unknown"))
        bucket = str(row.get("length_bucket", "unknown"))
        reward = float(row.get("reward", 0.0))
        tokens = float(row.get("reference_tokens", 0))
        item = buckets[(route, bucket)]
        item["count"] += 1.0
        item["reward_sum"] += reward
        item["reference_tokens"] += tokens
        route_item = routes[route]
        route_item["count"] += 1.0
        route_item["reward_sum"] += reward
        route_item["reference_tokens"] += tokens
    total_tokens = sum(item["reference_tokens"] for item in routes.values())
    total_count = sum(item["count"] for item in routes.values())

    def finish(item: dict[str, float], denominator: float) -> dict[str, float]:
        count = item["count"]
        return {
            "count": int(count),
            "mean_reward": item["reward_sum"] / count if count else 0.0,
            "reference_tokens": int(item["reference_tokens"]),
            "token_share": item["reference_tokens"] / denominator if denominator else 0.0,
        }

    route_bucket = {
        f"{route}:{bucket}": finish(item, total_tokens)
        for (route, bucket), item in sorted(buckets.items())
    }
    route_summary = {
        route: finish(item, total_tokens) for route, item in sorted(routes.items())
    }
    return {
        "schema_version": 1,
        "deduplicated_generation_groups": int(total_count),
        "route_bucket": route_bucket,
        "route": route_summary,
        "token_share_definition": "reference_tokens / sum(reference_tokens) in this optimizer step",
    }


def telemetry_gather_reward_group(
    local: RewardResult,
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[float, float, float]:
    """Exact RLV2 reward/advantage path plus a synchronized read-only audit."""

    gathered: list[RewardResult | None] = [None] * GROUP_SIZE
    dist.all_gather_object(gathered, local, group=group)
    rewards = [item for item in gathered if item is not None]
    if len(rewards) != GROUP_SIZE:
        raise RuntimeError("incomplete rebalanced RLV2 reward group")
    weights = rewards[0].weights
    if any(item.weights != weights for item in rewards):
        raise RuntimeError("reward route drift within rebalanced RLV2 group")
    totals = gdpo_total_rewards([item.components for item in rewards], weights).to(device)
    advantages = group_normalized_advantage(totals)
    local_rank = dist.get_rank(group)
    local_total = float(totals[local_rank].item())

    # All ranks enter this default-group collective in the same order. This
    # telemetry never feeds a tensor back into the optimizer update.
    local_meta = dict(_CURRENT_META)
    local_meta["reward"] = local_total
    local_meta["format_valid"] = bool(local.format_valid)
    world_rows: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(world_rows, local_meta)
    global _LAST_TELEMETRY
    _LAST_TELEMETRY = _aggregate_telemetry([item for item in world_rows if item])
    return local_total, float(advantages[local_rank].item()), float(
        totals.std(unbiased=False).item()
    )


def telemetry_append_jsonl(path: Any, payload: dict[str, Any]) -> None:
    enriched = dict(payload)
    enriched["data_version"] = "rl-rebalanced"
    enriched["route_length_telemetry"] = _LAST_TELEMETRY
    _ORIGINAL_APPEND(path, enriched)


def main() -> None:
    # Patch only this process's imported legacy module. V1/V2/V3 files on
    # disk remain untouched and the base algorithm remains authoritative.
    base.EXPECTED_ROWS = EXPECTED_ROWS
    base.TOTAL_STEPS = TOTAL_STEPS
    base.SAVE_STEPS = SAVE_STEPS
    base.MultirouteRLV2Mixture = RebalancedRLV2Mixture
    base.MultiRouteGAMRewardAdapter = TelemetryRewardAdapter
    legacy.gather_reward_group = telemetry_gather_reward_group
    legacy.append_jsonl = telemetry_append_jsonl
    base.main()


if __name__ == "__main__":
    main()

