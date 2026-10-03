"""RoboSpatial context task prompts and normalized point scoring.

ROBOSPATIAL_CONTEXT_PROMPT=native selects the official tuple prompt;
the default uses the shared spatial prompt format."""

import io
import re
import ast
import json
import os
import sys
import base64
import numpy as np
from PIL import Image

_SHARED_UTILS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "utils",
)
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

from prompt_mode import (
    TaskType,
    build_mode_refer_point_prompt,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_point_to_norm,
    parse_mode_predictions_for_scoring,
)
from eval_io import image_failure_marker, record_image_failure, recorded_image_failure, valid_results
from coordinate_mode import vlm_point_to_norm


# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #
def _strip_think(text: str) -> str:
    """去掉 <think>...</think> 思考块（enable_thinking=False 下一般已无，防御性处理）。"""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _first_str(model_output):
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0] if model_output else ""
    return model_output if isinstance(model_output, str) else str(model_output)


def robospatial_doc_to_visual(doc):
    try:
        return [doc["img"].convert("RGB")]
    except Exception as exc:
        record_image_failure("gam_robospatial_context", doc.get("id", ""), "hf://img", exc)
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]


# ---- context prompt：默认对齐 RefSpatial（Qwen 原生 JSON point_2d, 0-1000）---- #
def _ctx_core(doc):
    """取 context 问句的空间描述部分，剥掉原 [0,1] 格式说明句。"""
    q = str(doc["question"]).strip()
    idx = q.find("Your answer should be formatted")
    return q[:idx].strip() if idx != -1 else q


# RefSpatial 官方多行 JSON 后缀（与 RefSpatialBench 系列 prompt 完全一致）
_CTX_JSON_SUFFIX = ''' Output the point coordinates in JSON format.
For example:
[
{"point_2d": [x, y], "label": "point_1"}
]'''


def robospatial_context_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """context 点定位默认 prompt。
    默认走 grounding 口径：RefSpatial 风格 JSON point_2d(0-1000)，对 Qwen3-VL 友好。
    设 ROBOSPATIAL_CONTEXT_PROMPT=native 时切回数据集原生 [0,1] 元组提示（忠实官方 benchmark 原文）。
    """
    if is_native_spatial_mode():
        return build_mode_refer_point_prompt(_ctx_core(doc))
    if os.environ.get("ROBOSPATIAL_CONTEXT_PROMPT", "refspatial").strip().lower() == "native":
        return str(doc["question"]).strip()
    return _ctx_core(doc) + _CTX_JSON_SUFFIX


# ---- context (point-in-mask) 打分 ---- #
_POINT_RE = re.compile(
    r"[\(\[]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[\)\]]"
)
_NUM_POINTS_TO_MATCH = 2


def _extract_points(text: str, limit: int = _NUM_POINTS_TO_MATCH):
    """对齐官方：先正则抓 (x,y)/[x,y]，不足再 ast.literal_eval 兜底，最多 limit 个。"""
    text = _strip_think(text)
    points = []
    if os.environ.get("GAM_EVAL_VLM_FAMILY", "").lower() in {
        "mimo", "bagel", "rynn"
    }:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            coord = item.get("point_2d")
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                continue
            points.append((float(coord[0]), float(coord[1])))
            if len(points) >= limit:
                return points
        if points:
            return points
    for m in _POINT_RE.finditer(text):
        try:
            points.append((float(m.group(1)), float(m.group(2))))
        except (ValueError, TypeError):
            continue
        if len(points) >= limit:
            return points
    if points:
        return points
    try:
        val = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        return []

    def _collect(v):
        if len(points) >= limit:
            return
        if (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(t, (int, float)) for t in v)):
            points.append((float(v[0]), float(v[1])))
            return
        if isinstance(v, (list, tuple)):
            for it in v:
                _collect(it)
                if len(points) >= limit:
                    return

    _collect(val)
    return points


def _to_norm(x, y, w, h):
    """按显式模型坐标合同换算为归一化 [0,1]。"""
    return vlm_point_to_norm((x, y), w, h)


def _point_in_mask(x, y, mask_arr, w, h):
    px = int(round(x * (w - 1)))
    py = int(round(y * (h - 1)))
    px = min(max(px, 0), w - 1)
    py = min(max(py, 0), h - 1)
    return int(mask_arr[py, px]) > 0


def robospatial_context_process_results(doc, results):
    # lmms-eval strips Image 列（dataset_no_image），process_results 拿不到 mask，
    # 故用预处理注入的 base64 字符串列 mask_b64（Value(string)，不会被剥离）。
    pred = _first_str(results)
    failure = recorded_image_failure("gam_robospatial_context", doc.get("id", ""))
    if failure:
        return {"score": dict(failure, image_read_failed=True)}
    b64 = doc.get("mask_b64")
    if not b64:
        return {
            "score": image_failure_marker(
                "gam_robospatial_context",
                doc.get("id", ""),
                "base64://mask_b64",
                ValueError("missing mask_b64"),
            )
        }
    try:
        mask = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    except Exception as exc:
        return {
            "score": image_failure_marker(
                "gam_robospatial_context",
                doc.get("id", ""),
                "base64://mask_b64",
                exc,
            )
        }
    w, h = mask.size
    mask_arr = np.array(mask, dtype=np.uint8)

    if is_native_spatial_mode():
        native = parse_mode_predictions_for_scoring(pred, TaskType.POINT)
        points = []
        seen = set()
        for _, point in native:
            normalized = tuple(mode_grid_point_to_norm(point))
            if normalized not in seen:
                seen.add(normalized)
                points.append(normalized)
    else:
        points = _extract_points(pred)
    if not points:
        return {"score": 0.0}
    correct = False
    for x, y in points:
        if is_native_spatial_mode():
            nx, ny = x, y
        else:
            nx, ny = _to_norm(x, y, w, h)
        if _point_in_mask(nx, ny, mask_arr, w, h):
            correct = True
            break
    return {"score": 1.0 if correct else 0.0}


def aggregate_score(results, args=None):
    rows = valid_results(results, context="gam_robospatial_context/score")
    return sum(float(value) for value in rows) / len(rows) if rows else 0.0
