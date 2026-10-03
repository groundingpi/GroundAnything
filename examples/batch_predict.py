#!/usr/bin/env python3
"""Sequential, resumable GAM annotation through an OpenAI-compatible service."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grounding_anything.protocol import parse_answer

DEFAULT_BASE_URL = "http://127.0.0.1:8101/v1"
DEFAULT_MODEL = "groundinganything"
SCHEMA_VERSION = 1


def read_inputs(path: Path) -> list[dict]:
    """Validate the manifest before issuing any request."""
    rows, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                for key in ("id", "image", "prompt"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                if row.get("task") not in {"bbox", "point"}:
                    raise ValueError("task must be bbox or point")
                if row["id"] in seen:
                    raise ValueError(f"duplicate id: {row['id']!r}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("input JSONL contains no records")
    return rows


def successful_requests(path: Path) -> set[str]:
    """Never overwrite existing results or silently discard a damaged log."""
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("unexpected output schema")
                request_id = record.get("request_id")
                if record.get("status") == "success":
                    if not isinstance(request_id, str) or len(request_id) != 64:
                        raise ValueError("success record is missing a request fingerprint")
                    completed.add(request_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{path}:{number}: invalid result log ({exc}); preserve and repair "
                    "the damaged line, or choose a new output path"
                ) from exc
    return completed


def fingerprint(value: dict) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def parse_completion(response: dict, task: str) -> dict:
    result = {"raw_output": None, "predictions": [], "finish_reason": None,
              "usage": None, "error": None, "status": "invalid_response"}
    try:
        if not isinstance(response, dict):
            raise ValueError("service returned a non-object response")
        result["usage"] = response.get("usage")
        choice = response["choices"][0]
        raw = choice["message"]["content"]
        result["raw_output"] = raw
        result["finish_reason"] = choice.get("finish_reason")
        if not isinstance(raw, str):
            raise ValueError("service returned non-text content")
        if result["finish_reason"] != "stop":
            raise ValueError(f"incomplete generation: finish_reason={result['finish_reason']!r}")
        entries = parse_answer(raw, task=task, require_canonical=True)
        result["predictions"] = [{"label": e.phrase, "coordinates": e.coordinates} for e in entries]
        result["status"] = "success"
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def append_result(destination, result: dict) -> None:
    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
    destination.flush()
    os.fsync(destination.fileno())


def run(args: argparse.Namespace) -> dict:
    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("input and output must be different files")
    parsed_url = urlsplit(args.base_url)
    if (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc
            or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment):
        raise ValueError("base-url must be an HTTP(S) URL without credentials, query or fragment")
    if not args.model.strip():
        raise ValueError("model must not be empty")
    if args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("max-tokens and timeout must be positive; timeout must be finite")
    rows, completed = read_inputs(input_path), successful_requests(output_path)
    config = {"base_url": args.base_url.rstrip("/"), "model": args.model,
              "max_tokens": args.max_tokens, "temperature": 0, "run_tag": args.run_tag,
              "skip_special_tokens": False, "spaces_between_special_tokens": False}
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError(f"environment variable {args.api_key_env!r} is not set")
    stats = {"success": 0, "failed": 0, "skipped": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A complete last record without a newline is valid JSONL; insert the
    # separator before the next append. Partial JSON records fail above.
    needs_newline = False
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open("rb") as source:
            source.seek(-1, os.SEEK_END)
            needs_newline = source.read(1) != b"\n"
    with output_path.open("a", encoding="utf-8", newline="\n") as destination:
        if needs_newline:
            destination.write("\n")
        for row in rows:
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = input_path.parent / image_path
            result = {"schema_version": SCHEMA_VERSION, "id": row["id"],
                      "image": str(image_path.resolve()), "prompt": row["prompt"],
                      "task": row["task"], "config": config, "request_id": None,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "status": "request_error", "error": None, "raw_output": None,
                      "predictions": [], "finish_reason": None, "usage": None}
            try:
                mime = mimetypes.guess_type(image_path.name)[0]
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    raise ValueError("image extension must be JPEG, PNG or WebP")
                image_bytes = image_path.read_bytes()
                result["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                request_id = fingerprint({"schema_version": SCHEMA_VERSION,
                    "id": row["id"], "image_sha256": result["image_sha256"],
                    "prompt": row["prompt"], "task": row["task"], "config": config})
                result["request_id"] = request_id
                if request_id in completed:
                    stats["skipped"] += 1
                    continue
                image_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
                body = {key: config[key] for key in ("model", "temperature", "max_tokens",
                        "skip_special_tokens", "spaces_between_special_tokens")}
                body["messages"] = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": row["prompt"]}]}]
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key
                request = Request(config["base_url"] + "/chat/completions",
                                  data=json.dumps(body).encode("utf-8"), headers=headers)
                with urlopen(request, timeout=args.timeout) as response:
                    result.update(parse_completion(json.load(response), row["task"]))
            except HTTPError as exc:
                result["error"] = f"HTTP {exc.code}: {exc.reason}"
            except (OSError, URLError, ValueError) as exc:
                result["error"] = str(exc)
            append_result(destination, result)
            if result["status"] == "success":
                completed.add(result["request_id"])
                stats["success"] += 1
            else:
                stats["failed"] += 1
            print(json.dumps({"id": row["id"], "status": result["status"]}), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL; image paths are relative to this file")
    parser.add_argument("--output", required=True, help="Append-only result JSONL; successful requests are resumed")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--run-tag", default="", help="Change this when server weights or decoding settings change")
    parser.add_argument("--api-key-env", help="Name of an environment variable holding a service API key")
    args = parser.parse_args()
    try:
        stats = run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(stats))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
