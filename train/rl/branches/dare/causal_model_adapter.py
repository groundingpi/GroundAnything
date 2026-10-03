"""Freeze K3 ViT and assert the standard causal GroundAnything policy contract."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _install_transformers5_compat():
    """Expose DARE's Transformers-4 auto-class name inside RLV3 workers."""

    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        replacement = getattr(transformers, "AutoModelForImageTextToText", None)
        if replacement is None:
            raise RuntimeError(
                "RLV3 requires AutoModelForImageTextToText as the "
                "AutoModelForVision2Seq compatibility target"
            )
        # At this point Transformers has finished installing its _LazyModule;
        # unlike sitecustomize bootstrap time, this attribute survives and is
        # visible to DARE's subsequent ``from transformers import ...``.
        transformers.AutoModelForVision2Seq = replacement
    return transformers.AutoModelForCausalLM


def _group(name: str) -> str:
    if ".visual.merger." in name or ".visual.projector." in name:
        return "projector"
    if ".visual." in name:
        return "vision_encoder"
    return "language"


def install() -> None:
    AutoModelForCausalLM = _install_transformers5_compat()

    if getattr(AutoModelForCausalLM, "_gam_rlv3_causal_patch", False):
        return
    original = AutoModelForCausalLM.from_pretrained

    def wrapped(*args, **kwargs):
        model = original(*args, **kwargs)
        if getattr(model.config, "model_type", None) != "groundinganything_vlm":
            return model
        counts = {
            "vision_encoder": {"total": 0, "trainable": 0},
            "projector": {"total": 0, "trainable": 0},
            "language": {"total": 0, "trainable": 0},
        }
        for name, parameter in model.named_parameters():
            group = _group(name)
            parameter.requires_grad_(group != "vision_encoder")
            counts[group]["total"] += parameter.numel()
            counts[group]["trainable"] += parameter.numel() if parameter.requires_grad else 0
        if any(value["total"] <= 0 for value in counts.values()):
            raise RuntimeError(f"RLV3 parameter taxonomy incomplete: {counts}")
        if counts["vision_encoder"]["trainable"] != 0:
            raise RuntimeError("RLV3 K3 ViT freeze contract failed")
        for group in ("projector", "language"):
            if counts[group]["trainable"] != counts[group]["total"]:
                raise RuntimeError(f"RLV3 {group} trainability contract failed")
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        audit = {
            "status": "PASS",
            "policy_mode": "causal",
            "freeze_vision_encoder": True,
            "freeze_projector": False,
            "groups": counts,
        }
        model._gam_rlv3_trainability_audit = audit
        print(json.dumps({"gam_rlv3_model_audit": audit}, sort_keys=True), flush=True)
        audit_root = os.environ.get("GAM_RLV3_AUDIT_DIR")
        rank_value = os.environ.get("RANK")
        if audit_root and rank_value is not None:
            target = Path(audit_root) / f"model-audit-rank-{int(rank_value):02d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            temporary.write_text(
                json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(target)
        return model

    AutoModelForCausalLM.from_pretrained = wrapped
    AutoModelForCausalLM._gam_rlv3_causal_patch = True


install()
