"""Metadata-only admission check for the GroundAnything/plain-Qwen3 reference server."""
import json
from pathlib import Path

def validate_qwen3_base_config(model_dir):
    path=Path(model_dir)/"config.json"
    config=json.loads(path.read_text(encoding="utf-8"))
    text=config.get("text_config",{})
    if config.get("model_type") not in {"groundinganything", "groundinganything_vlm"} or text.get("model_type")!="qwen3":
        raise ValueError("dlm-serve requires a GroundAnything-VLM / plain-Qwen3 base model; Qwen3.5 is a separate backend")
    return config
