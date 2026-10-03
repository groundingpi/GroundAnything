"""Canonical JSONL row validation, independent of historical data adapters."""
from typing import List, Mapping

class RecordReject(ValueError):
    """A recoverable per-input rejection that must be written to rejects.jsonl."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail

def normalize_images(value) -> List[str]:
    if not isinstance(value, list) or not value:
        raise RecordReject("invalid_images", "images must be a non-empty list")
    paths = []
    for image in value:
        if isinstance(image, str):
            path = image
        elif isinstance(image, dict):
            path = image.get("path")
            if not path and image.get("bytes"):
                raise RecordReject("embedded_image_bytes", "path is required for output JSONL")
        else:
            path = None
        if not isinstance(path, str) or not path:
            raise RecordReject("invalid_image_path", repr(image)[:256])
        paths.append(path)
    return paths

def normalize_messages(value) -> List[dict]:
    if not isinstance(value, list) or not value:
        raise RecordReject("invalid_messages", "messages must be a non-empty list")
    messages = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise RecordReject("invalid_message", f"message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise RecordReject("invalid_role", f"message {index}: {role!r}")
        if not isinstance(content, str):
            raise RecordReject("invalid_content", f"message {index} content is not str")
        messages.append({"role": role, "content": content, "loss": None})
    if not any(message["role"] == "user" for message in messages):
        raise RecordReject("missing_user")
    if not any(message["role"] == "assistant" for message in messages):
        raise RecordReject("missing_assistant")
    return messages

def validate_uniform_row(row: Mapping) -> None:
    if set(row) != {"id", "messages", "images", "source"}:
        raise RecordReject("output_schema_keys", repr(sorted(row)))
    if not isinstance(row["id"], str) or not row["id"]:
        raise RecordReject("output_invalid_id")
    if not isinstance(row["source"], str) or not row["source"]:
        raise RecordReject("output_invalid_source")
    normalize_images(row["images"])
    if not isinstance(row["messages"], list) or not row["messages"]:
        raise RecordReject("output_invalid_messages")
    for index, message in enumerate(row["messages"]):
        if not isinstance(message, dict):
            raise RecordReject("output_invalid_message", str(index))
        if set(message) != {"role", "content", "loss"}:
            raise RecordReject(
                "output_message_schema_keys", f"{index}:{sorted(message)}"
            )
        if message["loss"] is not None:
            raise RecordReject("output_loss_not_null", str(index))
    messages = normalize_messages(row["messages"])
    image_tokens = sum(message["content"].count("<image>") for message in messages)
    if image_tokens != len(row["images"]):
        raise RecordReject(
            "image_token_count_mismatch",
            f"tokens={image_tokens} images={len(row['images'])}",
        )
