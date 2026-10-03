"""Shared image-read failure tracking for GAM evaluation tasks.

An unreadable image must not silently become a scored black placeholder.
Task visualizers return no visual, ``process_results`` emits the marker below,
and aggregators remove that sample from their denominator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from loguru import logger


IMAGE_READ_FAILED = "image_read_failed"
_FAILURES: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)


def record_image_failure(
    task_name: object,
    sample_id: object,
    image_path: object,
    error: object,
) -> dict[str, Any]:
    """Record one failure by stable task/id and return a serializable marker."""

    task = str(task_name or "unknown_task")
    identifier = str(sample_id or image_path or "unknown_id")
    details = {
        "task_name": task,
        "sample_id": identifier,
        "image_path": str(image_path or ""),
        "error": f"{type(error).__name__}: {error}",
    }
    first = identifier not in _FAILURES[task]
    _FAILURES[task][identifier] = details
    if first:
        logger.warning(
            "[gam-eval] 读图失败，跳过计分: task={} id={} path={} error={}",
            task,
            identifier,
            details["image_path"],
            details["error"],
        )
    return {IMAGE_READ_FAILED: True, **details}


def image_failure_marker(
    task_name: object,
    sample_id: object,
    image_path: object,
    error: object,
) -> dict[str, Any]:
    return record_image_failure(task_name, sample_id, image_path, error)


def recorded_image_failure(task_name: object, sample_id: object) -> dict[str, Any] | None:
    return _FAILURES.get(str(task_name or "unknown_task"), {}).get(str(sample_id))


def is_image_failure(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value.get(IMAGE_READ_FAILED))


def valid_results(results: Iterable[Any], *, context: str) -> list[Any]:
    """Filter read failures and log the exact skipped ids once per aggregate."""

    values = list(results)
    failed = [value for value in values if is_image_failure(value)]
    if failed:
        identifiers = sorted({str(value.get("sample_id", "")) for value in failed})
        logger.warning(
            "[gam-eval] {} 读图失败跳过 {}/{}，ids={}",
            context,
            len(failed),
            len(values),
            identifiers,
        )
    return [value for value in values if not is_image_failure(value)]


def failure_snapshot() -> dict[str, list[dict[str, str]]]:
    """Return deterministic diagnostics for tests/reporting."""

    return {
        task: [rows[key] for key in sorted(rows)]
        for task, rows in sorted(_FAILURES.items())
    }
