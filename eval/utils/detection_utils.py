"""
检测/grounding 评测任务的共享实现（doc_to_visual / doc_to_text / process_results / 聚合函数）。

被 Referring/Grounding/Dense 三个类别的 utils.py 共用（各自的 utils.py 只是一个"薄垫片"，
`from detection_utils import *` 把这里的实现重新导出——lmms_eval 的 `!function utils.xxx`
要求 utils.py physically 和引用它的 yaml 同目录，所以每个类别目录必须留一个 utils.py，
但真正的实现只在这一份文件里维护，改一处、三个类别全部生效）。

设计原则：
  - 推理：复用 lmms-eval 的 async_openai 模型（走 vLLM，Qwen3-VL 系列）。
  - 指标：调用 eval/metrics/detection_metrics.py 中的 UniversalMetricsCalculator，
          逐样本复用其打分逻辑（recall / precision / F1 / MAE）。

坐标系约定（COORD_MODE，环境变量 GAM_COORD_MODE 可覆盖）：
  - 标注 gt 为原图“绝对像素”坐标。
  - VLM mode 下 Qwen3-VL 系列原生输出 [0,1000] 归一化坐标
    （见 refcoco bbox_rec：qwen3 预测直接对齐归一化到 1000 的 GT）。
  - GAM mode 使用 atomic <0>..<999> 坐标 token，固定按 999 分母还原，不读取 COORD_MODE。
  - VLM 默认 qwen3：把模型预测从 [0,1000] 换算回绝对像素再喂给检测指标计算引擎。
    其它可选：norm01（[0,1] 浮点）、abs（已是绝对像素）、auto（按数值量级自动判断）。
"""

import os
import re
import json
import ast
import importlib.util
from functools import lru_cache

from PIL import Image, ImageDraw
from loguru import logger

from prompt_mode import (
    TaskType,
    build_mode_dense_bbox_prompt,
    build_mode_refer_bbox_prompt,
    build_mode_visual_prompt,
    is_gam_mode,
    is_locateanything_mode,
    is_native_spatial_mode,
    mode_grid_to_abs,
    parse_mode_predictions_for_scoring,
    pixel_boxes_to_mode_grid,
)
from eval_io import image_failure_marker, record_image_failure, valid_results
from coordinate_mode import vlm_box_to_pixel
from eval_data_root import dataset_path
from vlm_output_fallbacks import fallback_boxes

# --------------------------------------------------------------------------- #
# 路径配置（可用环境变量覆盖）
# --------------------------------------------------------------------------- #
# 检测指标计算引擎独立成 eval/metrics/detection_metrics.py（本文件的父目录的
# 兄弟目录 metrics/），按文件路径动态加载。
OTHER_METRIC_PATH = os.environ.get(
    "GAM_METRIC_MODULE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics", "detection_metrics.py"),
)
# Rex-Omni 正式官方数据根。image_path 形如 "dense200/xxx.jpg"，故根目录下应有
# dense200/、coco/ 等子目录。环境变量只用于显式的镜像/离线部署覆盖。
DEFAULT_IMAGE_ROOT = str(dataset_path("images"))
IMAGE_ROOT = os.environ.get("GAM_IMAGE_ROOT", DEFAULT_IMAGE_ROOT)

# 坐标模式：qwen3 / norm01 / abs / auto
COORD_MODE = os.environ.get("GAM_COORD_MODE", "qwen3").lower()
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


# --------------------------------------------------------------------------- #
# 检测指标计算引擎（按文件路径动态加载）
# --------------------------------------------------------------------------- #
_METRIC_MODULE = None


def _metric():
    global _METRIC_MODULE
    if _METRIC_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "detection_metrics", OTHER_METRIC_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _METRIC_MODULE = mod
    return _METRIC_MODULE


# --------------------------------------------------------------------------- #
# 图像相关
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=200000)
def _image_size(path):
    """只读图像 header 拿宽高，速度快。"""
    with Image.open(path) as im:
        return im.size  # (w, h)


def _full_image_path(doc):
    return os.path.join(IMAGE_ROOT, doc["image_path"])


def _load_json_field(doc, *names):
    for n in names:
        if n in doc and doc[n] is not None:
            v = doc[n]
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return v
            return v
    return None


def _visual_prompt_boxes(visual_prompt):
    """Flatten supported FSC147 exemplar encodings into absolute xyxy boxes."""

    raw_boxes = []
    if isinstance(visual_prompt, dict):
        for value in visual_prompt.values():
            if isinstance(value, list):
                raw_boxes.extend(value)
    elif isinstance(visual_prompt, list):
        if len(visual_prompt) >= 4 and all(
            isinstance(value, (int, float)) for value in visual_prompt[:4]
        ):
            raw_boxes = [visual_prompt]
        else:
            raw_boxes = visual_prompt

    boxes = []
    for box in raw_boxes:
        try:
            if (
                isinstance(box, (list, tuple))
                and len(box) >= 4
                and all(isinstance(value, (int, float)) for value in box[:4])
            ):
                x1, y1, x2, y2 = box[:4]
            elif (
                isinstance(box, (list, tuple))
                and len(box) >= 3
                and isinstance(box[0], (list, tuple))
            ):
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue
            boxes.append((x1, y1, x2, y2))
        except (IndexError, TypeError, ValueError):
            continue
    return boxes


def _draw_visual_prompt(image, visual_prompt):
    """在图上画出示例框（红色），用于 visual_prompt_detection（如 FSC147）。

    兼容多种 exemplar 格式：
      - {"object1": [[x1,y1,x2,y2], ...]}
      - [[x1,y1,x2,y2], ...]
      - [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]（多边形/四角点）
    """
    if not visual_prompt:
        return image
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for b in _visual_prompt_boxes(visual_prompt):
        try:
            x1, y1, x2, y2 = b
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        except Exception:
            continue
    return image


# 单图最大像素预算（超过则等比降采样）。因为模型输出 [0,1000] 归一化坐标，降采样不影响
# 坐标正确性（process_results 用原图 W/H 还原）。避免大图导致输入 token 超过 max-model-len。
MAX_IMAGE_PIXELS = int(os.environ.get("GAM_MAX_IMAGE_PIXELS", str(2560 * 28 * 28)))


def _maybe_downscale(image):
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS and w > 0 and h > 0:
        scale = (MAX_IMAGE_PIXELS / float(w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return image


def doc_to_visual(doc):
    path = _full_image_path(doc)
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        record_image_failure(
            doc.get("dataset_name", doc.get("task_name", "detection")),
            doc.get("id", doc.get("sample_index", doc.get("image_path", ""))),
            path,
            e,
        )
        # Do not fabricate a black image.  The failure registry makes
        # process_results emit an excluded marker; an empty visual list keeps
        # the request serializable without pretending that inference saw data.
        if is_gam_mode():
            return []
        image = Image.new("RGB", (28, 28), (0, 0, 0))
    if (
        not is_native_spatial_mode()
        and doc.get("task_name", "") == "visual_prompt_detection"
    ):
        vp = _load_json_field(doc, "visual_prompt_json", "visual_prompt")
        image = _draw_visual_prompt(image, vp)
    if (
        is_locateanything_mode()
        and doc.get("task_name", "") == "visual_prompt_detection"
    ):
        crops = []
        width, height = image.size
        vp = _load_json_field(doc, "visual_prompt_json", "visual_prompt")
        for box in _visual_prompt_boxes(vp):
            x1, y1, x2, y2 = (float(value) for value in box)
            left = max(0, min(width - 1, round(x1)))
            top = max(0, min(height - 1, round(y1)))
            right = max(left + 1, min(width, round(x2)))
            bottom = max(top + 1, min(height, round(y2)))
            crops.append(image.crop((left, top, right, bottom)).convert("RGB"))
        if not crops:
            raise ValueError("LocateAnything visual prompt has no valid reference crop")
        return [_maybe_downscale(image), *crops]
    return [_maybe_downscale(image)]


# --------------------------------------------------------------------------- #
# Prompt 构造
#   - 使用 Qwen-VL 原生 grounding 的
#     JSON bbox_2d 格式，且“不指定坐标尺度”，让模型用其原生 [0,1000] 千分位坐标输出。
#   - 多类别检测沿用 Qwen2.5/3-VL cookbook 的检测 prompt（bbox_2d + label 列表）。
# --------------------------------------------------------------------------- #
def _categories(doc):
    cats = _load_json_field(doc, "categories_json", "categories")
    if cats is None:
        gt = _load_json_field(doc, "gt_json", "gt")
        cats = list(gt.keys()) if isinstance(gt, dict) else []
    return cats


BOX_INSTRUCTION = (
    "Detect all the objects in this image that belong to these categories: {cats}. "
    "Report their locations as a JSON list, where each element is "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<category name>"}}. '
    "Use the exact category name as the label, and output one element per object instance. "
    "If a category has no object, omit it. If nothing matches, output an empty list []."
)

REFER_INSTRUCTION = (
    'Locate every object that matches the description "{phrase}" in the image. '
    "Report bbox coordinates in JSON format, as a list where each element is "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "{phrase}"}}. '
    "If nothing matches, output an empty list []."
)

VISUAL_PROMPT_INSTRUCTION = (
    "Some example objects are marked with red bounding boxes in this image. "
    "Detect ALL objects in the image of the same category as the marked examples "
    "(including the marked ones). "
    "Report their locations as a JSON list, where each element is "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}.'
)


def _deepseek_vl2_native_ref_prompt(phrase: str) -> str:
    """Render DeepSeek-VL2's released visual-grounding wire protocol.

    Keep this behind an explicit family route: generic VLM models are trained
    on the JSON ``bbox_2d`` contract, while DeepSeek-VL2's official checkpoint
    expects the special-token ``ref -> det`` protocol and otherwise commonly
    answers with ``0`` or free-form prose.
    """

    style = os.environ.get("GAM_DEEPSEEK_NATIVE_PROMPT_STYLE", "ref").strip().lower()
    prefix = "<|grounding|>" if style == "grounding_ref" else ""
    if style not in {"ref", "grounding_ref", "native_all"}:
        raise ValueError(f"unsupported DeepSeek native prompt style: {style!r}")
    return f"{prefix}<|ref|>{phrase}<|/ref|>."


def _deepseek_vl2_native_category_prompt(categories) -> str:
    """Ask DeepSeek-VL2 to localize every requested category natively.

    The released DeepSeek-VL2 examples use ``Find all the
    <|ref|>category<|/ref|>`` for category localization.  Keep one sentence
    per category so the model can return one labelled ``<|det|>`` block per
    query.  This route is opt-in and never changes the generic VLM contract.
    """

    cats = [str(category).strip() for category in categories if str(category).strip()]
    if not cats:
        cats = ["object"]
    return " ".join(f"Find all the <|ref|>{category}<|/ref|>." for category in cats)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    task = doc.get("task_name", "")
    cats = _categories(doc)
    if is_native_spatial_mode():
        if task == "referring_object_detection":
            phrase = str(cats[0]) if cats else "object"
            dataset_name = str(doc.get("dataset_name", "")).lower()
            return build_mode_refer_bbox_prompt(
                phrase, multiple=dataset_name == "humanref"
            )
        if task == "visual_prompt_detection":
            vp = _load_json_field(doc, "visual_prompt_json", "visual_prompt")
            width, height = _image_size(_full_image_path(doc))
            reference_boxes = pixel_boxes_to_mode_grid(
                _visual_prompt_boxes(vp), width, height
            )
            return build_mode_visual_prompt(reference_boxes)
        return build_mode_dense_bbox_prompt(str(category) for category in cats)
    if task == "referring_object_detection":
        phrase = cats[0] if cats else ""
        if os.environ.get("GAM_EVAL_VLM_FAMILY", "").lower() == "deepseek_vl2_native":
            return _deepseek_vl2_native_ref_prompt(str(phrase))
        return REFER_INSTRUCTION.format(phrase=phrase)
    if task == "visual_prompt_detection":
        return VISUAL_PROMPT_INSTRUCTION
    if (
        os.environ.get("GAM_EVAL_VLM_FAMILY", "").lower() == "deepseek_vl2_native"
        and os.environ.get("GAM_DEEPSEEK_NATIVE_PROMPT_STYLE", "ref").strip().lower()
        == "native_all"
    ):
        return _deepseek_vl2_native_category_prompt(cats)
    # common_object_detection / dense_object_detection / 其它默认走多类别检测
    return BOX_INSTRUCTION.format(cats=", ".join(str(c) for c in cats))


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Preserve LocateAnything's official source/text/reference image order."""

    visuals = doc_to_visual(doc)
    text = doc_to_text(doc, lmms_eval_specific_kwargs)
    if is_locateanything_mode() and doc.get("task_name") == "visual_prompt_detection":
        content = [{"type": "image", "url": visuals[0]}]
        content.append({"type": "text", "text": text})
        content.extend({"type": "image", "url": image} for image in visuals[1:])
    else:
        content = [{"type": "image", "url": image} for image in visuals]
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# 解析模型输出
# --------------------------------------------------------------------------- #
def _normalize_kimi_vlm_response(text):
    """Repair observed Kimi-only VLM JSON wrappers without changing prompts.

    Kimi occasionally emits a list item as ``[{{...}}]``.  Python interprets
    the extra braces as a set containing a dict and raises ``TypeError`` during
    ``ast.literal_eval``.  Restrict this repair to the explicit Kimi VLM family
    branch so GAM mode and every other VLM family retain their byte-for-byte
    parsing contract.
    """
    if is_gam_mode() or os.environ.get("GAM_EVAL_VLM_FAMILY", "").lower() != "kimi":
        return text
    doubled_item = re.compile(r"(?P<prefix>\[|,)\s*\{\s*\{(?=\s*[\"'])")
    if not doubled_item.search(text):
        return text
    repaired = doubled_item.sub(lambda match: f"{match.group('prefix')}{{", text)
    # Only remove the matching surplus closing brace after a doubled opening
    # was positively identified above.  This avoids touching valid nested JSON.
    repaired = re.sub(r"(?<=\})\s*\}(?=\s*(?:,|\]))", "", repaired)
    return repaired


def _parse_predictions(text):
    """解析模型输出，返回 [(label, [x1,y1,x2,y2])]（坐标为模型原始尺度）。"""
    if not text:
        return []
    # 去掉思考链
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = _normalize_kimi_vlm_response(text)
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE)

    results = []

    # 0) Qwen-VL 老格式 grounding token（容错兜底，保证各种历史格式都能 match）：
    #    <ref>name</ref>? <box>(x1,y1),(x2,y2)</box>
    #    <|object_ref_start|>name<|object_ref_end|>? <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    if "<box>" in cleaned or "<|box_start|>" in cleaned:
        legacy_pair = re.compile(
            r"(?:<ref>(?P<ref1>[^<]*)</ref>|<\|object_ref_start\|>(?P<ref2>[^<]*)<\|object_ref_end\|>)?\s*"
            r"(?:<box>|<\|box_start\|>)(?P<box>[^<]*?)(?:</box>|<\|box_end\|>)",
            re.DOTALL,
        )
        pt_re = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
        for m in legacy_pair.finditer(cleaned):
            ref = (m.group("ref1") or m.group("ref2") or "object").strip() or "object"
            pts = pt_re.findall(m.group("box") or "")
            if len(pts) >= 2:
                try:
                    results.append(
                        (ref, [float(pts[0][0]), float(pts[0][1]), float(pts[1][0]), float(pts[1][1])])
                    )
                except (ValueError, TypeError):
                    continue
        if results:
            return results

    # 0b) DeepSeek-VL2 native grounding protocol.  Coordinates use the
    # model's 0..999 grid and are subsequently routed by GAM_COORD_MODE=auto.
    native_det = re.compile(
        r"(?:<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*)?"
        r"<\|det\|>(?P<boxes>.*?)<\|/det\|>",
        re.DOTALL,
    )
    number_pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
    for match in native_det.finditer(cleaned):
        label = str(match.group("label") or "object").strip() or "object"
        numbers = number_pattern.findall(match.group("boxes") or "")
        for offset in range(0, len(numbers) - 3, 4):
            try:
                results.append(
                    (label, [float(value) for value in numbers[offset : offset + 4]])
                )
            except (TypeError, ValueError):
                continue
    if results:
        return results

    # 1) Qwen 系列优先按 JSON/Python-literal 解析。兼容顶层 list、单
    # dict，以及 {"objects"|"predictions"|"results": [...]} 包装。
    candidates = [cleaned]
    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opening), cleaned.rfind(closing)
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])
    for cand in sorted(set(candidates), key=len, reverse=True):
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, RecursionError, TypeError):
            try:
                parsed = ast.literal_eval(cand)
            # Malformed model output such as ``[{{"bbox_2d": ...}}]`` can
            # construct a set containing a dict and make literal_eval raise
            # TypeError.  A single unparsable prediction must score as empty;
            # it must never abort postprocessing for the whole benchmark.
            except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
                continue
        if isinstance(parsed, dict):
            nested = next(
                (
                    parsed.get(key)
                    for key in ("objects", "predictions", "results", "detections")
                    if isinstance(parsed.get(key), list)
                ),
                None,
            )
            parsed = nested if nested is not None else [parsed]
        if isinstance(parsed, (list, tuple)):
            for item in parsed:
                if isinstance(item, dict):
                    box = (
                        item.get("bbox_2d")
                        or item.get("bbox")
                        or item.get("box_2d")
                        or item.get("box")
                    )
                    label = (
                        item.get("label")
                        or item.get("category")
                        or item.get("name")
                        or "object"
                    )
                    if (
                        isinstance(box, (list, tuple))
                        and len(box) == 1
                        and isinstance(box[0], (list, tuple))
                    ):
                        box = box[0]
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        try:
                            results.append(
                                (str(label), [float(box[i]) for i in range(4)])
                            )
                        except (ValueError, TypeError):
                            continue
            if results:
                return results

    # 2) 兜底：逐个匹配 {"bbox_2d": [...], "label": "..."}
    pattern = r'"(?:bbox_2d|bbox|box_2d|box)"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\][^{}\]]*?"(?:label|category|name)"\s*:\s*"([^"]*)"'
    for m in re.finditer(pattern, cleaned):
        try:
            box = [float(m.group(i)) for i in range(1, 5)]
            results.append((str(m.group(5) or "object"), box))
        except (ValueError, TypeError):
            continue
    if results:
        return results

    # 3) Qwen3.5 系模型有时会把原生两个点坐标嵌进 JSON 模板，形成
    #    {"bbox_2d": [x1,y1),(x2,y2], "label": "..."}。这不是合法
    #    JSON，但 label 之前最后四个数仍是完整且无歧义的 xyxy 坐标。
    #    仅在显式存在 label/category/name 时恢复，避免从普通文本中猜框。
    label_pattern = re.compile(
        r'"(?:label|category|name)"\s*:\s*"([^"]*)"', re.IGNORECASE
    )
    number_pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
    seen = set()
    for label_match in label_pattern.finditer(cleaned):
        entry_start = cleaned.rfind("{", 0, label_match.start())
        segment = cleaned[entry_start + 1 : label_match.start()]
        numbers = number_pattern.findall(segment)
        if len(numbers) < 4:
            continue
        try:
            box = [float(value) for value in numbers[-4:]]
        except ValueError:
            continue
        label = str(label_match.group(1) or "object")
        key = (label, *box)
        if key not in seen:
            seen.add(key)
            results.append((label, box))
    if results:
        return results

    # 4) 最后恢复模型已经明确输出、但括号族不配对的坐标，以及
    #    ``<object>(x1,y1),(x2,y2)</object>``。该路径位于所有标准解析
    #    之后，不改变合法 JSON 的任何行为。
    results.extend(fallback_boxes(cleaned))
    if results:
        return results

    # 5) 再兜底：纯 [x1,y1,x2,y2]（无 label）
    for m in re.finditer(
        r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", cleaned
    ):
        try:
            results.append(("object", [float(m.group(i)) for i in range(1, 5)]))
        except (ValueError, TypeError):
            continue
    return results


def _to_abs(box, w, h):
    """把模型坐标换算到绝对像素，按 COORD_MODE 处理。"""
    return vlm_box_to_pixel(box, w, h)


def _score_sample(sample, iou=0.5):
    """逐样本调用检测指标计算引擎的 UniversalMetricsCalculator，返回 (recall, precision, mae)。"""
    m = _metric()
    calc = m.UniversalMetricsCalculator()
    try:
        calc.calculate_metrics_for_sample(dict(sample), iou)
    except Exception as e:
        logger.warning(f"[gam] 单样本算分失败: {e}")
        return 0.0, 0.0, 0.0
    key = f"{sample['task_name']}_{sample['dataset_name']}"
    recalls = calc.results.get(key, {}).get("recalls", [])
    precisions = calc.results.get(key, {}).get("precisions", [])
    recall = float(recalls[0]) if recalls else 0.0
    precision = float(precisions[0]) if precisions else 0.0
    maes = calc.visual_prompt_metrics.get(key, {}).get("maes", [])
    mae = float(maes[0]) if maes else 0.0
    return recall, precision, mae


def _normalized_box_mapping(mapping, task_name):
    """Normalize category names without losing boxes from colliding aliases."""

    metric = _metric()
    output = {}
    for raw_label, boxes in (mapping or {}).items():
        label = metric.normalize_category_name(str(raw_label).lower(), task_name)
        output.setdefault(label, []).extend(
            box for box in (boxes or []) if box is not None and box != "None"
        )
    return _deduplicate_box_mapping(output)


def _score_sample_category_aware(sample, iou=0.5):
    """GAM category detection scorer with label-constrained one-to-one matching.

    The shared historical metric flattens predictions from all queried GT
    categories before matching.  That lets, for example, a ``cat`` box match a
    ``dog`` GT box.  GAM category routes require labels, so matching is done
    independently inside each normalized category.  Predictions under unknown
    labels remain false positives in the precision denominator.
    """

    metric = _metric()
    task_name = sample["task_name"]
    gt = _normalized_box_mapping(sample.get("gt", {}), task_name)
    predictions = _normalized_box_mapping(
        sample.get("extracted_predictions", {}), task_name
    )
    total_gt = sum(len(boxes) for boxes in gt.values())
    total_predictions = sum(len(boxes) for boxes in predictions.values())

    if total_gt == 0:
        score = 1.0 if total_predictions == 0 else 0.0
        mae = abs(total_predictions - total_gt) if task_name == "visual_prompt_detection" else 0.0
        return score, score, float(mae)
    if total_predictions == 0:
        mae = float(total_gt) if task_name == "visual_prompt_detection" else 0.0
        return 0.0, 0.0, mae

    matches = 0
    for label, gt_boxes in gt.items():
        pred_boxes = predictions.get(label, [])
        used_predictions = set()
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_index = -1
            for index, pred_box in enumerate(pred_boxes):
                if index in used_predictions:
                    continue
                overlap = metric.calculate_iou(gt_box, pred_box)
                if overlap >= iou and overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                matches += 1
                used_predictions.add(best_index)

    recall = matches / total_gt
    precision = matches / total_predictions
    mae = (
        float(abs(total_predictions - total_gt))
        if task_name == "visual_prompt_detection"
        else 0.0
    )
    return recall, precision, mae


def _score_category_aware_iou_sweep(sample):
    """Vectorize GAM's ten-threshold sweep over one precomputed IoU table."""

    metric = _metric()
    task_name = sample["task_name"]
    gt = _normalized_box_mapping(sample.get("gt", {}), task_name)
    predictions = _normalized_box_mapping(
        sample.get("extracted_predictions", {}), task_name
    )
    total_gt = sum(len(boxes) for boxes in gt.values())
    total_predictions = sum(len(boxes) for boxes in predictions.values())
    mae = (
        float(abs(total_predictions - total_gt))
        if task_name == "visual_prompt_detection"
        else 0.0
    )
    if total_gt == 0:
        score = 1.0 if total_predictions == 0 else 0.0
        return [score] * len(IOU_THRESHOLDS), [score] * len(IOU_THRESHOLDS), mae
    if total_predictions == 0:
        return [0.0] * len(IOU_THRESHOLDS), [0.0] * len(IOU_THRESHOLDS), mae

    overlap_tables = []
    for label, gt_boxes in gt.items():
        pred_boxes = predictions.get(label, [])
        overlap_tables.append([
            [metric.calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
            for gt_box in gt_boxes
        ])

    recalls = []
    precisions = []
    for threshold in IOU_THRESHOLDS:
        matches = 0
        for table in overlap_tables:
            used_predictions = set()
            for overlaps in table:
                best_iou = 0.0
                best_index = -1
                for index, overlap in enumerate(overlaps):
                    if index in used_predictions:
                        continue
                    if overlap >= threshold and overlap > best_iou:
                        best_iou = overlap
                        best_index = index
                if best_index >= 0:
                    matches += 1
                    used_predictions.add(best_index)
        recalls.append(matches / total_gt)
        precisions.append(matches / total_predictions)
    return recalls, precisions, mae


def _score_sample_iou_sweep(sample):
    """Return per-threshold P/R values and the threshold-independent count MAE."""

    if sample.get("_gam_category_aware"):
        return _score_category_aware_iou_sweep(sample)

    recalls = []
    precisions = []
    mae = 0.0
    for index, threshold in enumerate(IOU_THRESHOLDS):
        recall, precision, threshold_mae = _score_sample(sample, threshold)
        recalls.append(recall)
        precisions.append(precision)
        if index == 0:
            mae = threshold_mae
    return recalls, precisions, mae


def _metric_payload(recalls, precisions, mae):
    """Build the normalized IoU=.50/.95/.50:.05:.95 metric contract."""

    return {
        "recall_iou_50": recalls[0],
        "recall_iou_95": recalls[-1],
        "recall_mean_iou": tuple(recalls),
        "precision_iou_50": precisions[0],
        "precision_iou_95": precisions[-1],
        "precision_mean_iou": tuple(precisions),
        "f1_iou_50": {"recall": recalls[0], "precision": precisions[0]},
        "f1_iou_95": {"recall": recalls[-1], "precision": precisions[-1]},
        "F1mIoU": {
            "recalls": tuple(recalls),
            "precisions": tuple(precisions),
        },
        "mae": mae,
    }


def _failure_metric_payload(marker):
    return {name: dict(marker) for name in (
        "recall_iou_50", "recall_iou_95", "recall_mean_iou",
        "precision_iou_50", "precision_iou_95", "precision_mean_iou",
        "f1_iou_50", "f1_iou_95", "F1mIoU", "mae",
    )}


def _deduplicate_box_mapping(mapping):
    """Stable exact-box deduplication inside each scoring label."""

    output = {}
    for label, boxes in mapping.items():
        seen = set()
        unique = []
        for box in boxes or []:
            key = tuple(float(value) for value in box)
            if key in seen:
                continue
            seen.add(key)
            unique.append(list(key))
        output[label] = unique
    return output


def _process_results(doc, results, *, labelless=False):
    raw = results[0] if results else ""
    try:
        w, h = _image_size(_full_image_path(doc))
    except Exception as e:
        marker = image_failure_marker(
            doc.get("dataset_name", doc.get("task_name", "detection")),
            doc.get("id", doc.get("sample_index", doc.get("image_path", ""))),
            _full_image_path(doc),
            e,
        )
        return _failure_metric_payload(marker)

    if is_native_spatial_mode():
        preds = parse_mode_predictions_for_scoring(raw, TaskType.BBOX)
    else:
        preds = _parse_predictions(raw)
    extracted = {}
    for label, box in preds:
        absolute_box = (
            mode_grid_to_abs(box, w, h)
            if is_native_spatial_mode()
            else _to_abs(box, w, h)
        )
        # A zero-area prediction can never match a positive-area GT box.  Drop
        # it before precision accounting instead of letting repeated decoder
        # collapse artifacts dominate the false-positive denominator.
        if absolute_box[0] == absolute_box[2] or absolute_box[1] == absolute_box[3]:
            continue
        extracted.setdefault(label, []).append(absolute_box)
    extracted = _deduplicate_box_mapping(extracted)

    gt = _load_json_field(doc, "gt_json", "gt") or {}

    # Referring asks for geometry selected by a single textual query, so the
    # response label is metadata rather than a target.  Category detection and
    # visual prompting retain their label contract.
    vlm_family = os.environ.get("GAM_EVAL_VLM_FAMILY", "").strip().lower()
    # RynnBrain's native grounding protocol returns an ordered set of
    # ``<object>`` coordinates without per-box category strings.  The queried
    # categories are already fixed by the input prompt, so its family route is
    # geometry-only across detection tasks.  This does not affect standard
    # VLM/GAM JSON routes, where emitted category labels remain mandatory.
    ignore_labels = (
        labelless
        or vlm_family == "rynn"
        or (
            doc.get("task_name") == "referring_object_detection"
            and is_gam_mode()
        )
    )
    if ignore_labels and isinstance(gt, dict):
        gt = {"object": [box for boxes in gt.values() for box in (boxes or [])]}
        extracted = {
            "object": [box for boxes in extracted.values() for box in (boxes or [])]
        }
        extracted = _deduplicate_box_mapping(extracted)

    # visual_prompt_detection（如 FSC147）是“检测与示例同类的全部目标”，本质单类别。
    # 但 GT 类别键（如 "object1"）与模型输出 label（"object"）不一致，指标引擎按
    # 类别键匹配会全 miss。这里把所有预测框归到 GT 的单一类别键下。
    if doc.get("task_name") == "visual_prompt_detection" and isinstance(gt, dict) and gt:
        gt_cat = next(iter(gt.keys()))
        if is_gam_mode():
            # GAM visual-prompt training uses the canonical opaque label
            # ``object``.  Wrong generated labels must not receive credit.
            all_boxes = list(extracted.get("object", []))
        else:
            all_boxes = [b for boxes in extracted.values() for b in boxes]
        extracted = _deduplicate_box_mapping({gt_cat: all_boxes})
    sample = {
        "gt": gt,
        "extracted_predictions": extracted,
        "task_name": doc["task_name"],
        "dataset_name": doc["dataset_name"],
        # In Referring/Labelless, mappings were explicitly pooled to one
        # ``object`` category above.  Thus this scorer remains geometry-only
        # there, while all other GAM routes are label-constrained.
        "_gam_category_aware": is_gam_mode(),
    }
    recalls, precisions, mae = _score_sample_iou_sweep(sample)
    return _metric_payload(recalls, precisions, mae)


def process_results(doc, results):
    return _process_results(doc, results, labelless=False)


def process_results_labelless(doc, results):
    """Detection metric variant that pools all labels before IoU matching."""

    return _process_results(doc, results, labelless=True)


# --------------------------------------------------------------------------- #
# 聚合（注意：F1 用“平均 P / 平均 R”计算，而非逐样本 F1 再平均）
# --------------------------------------------------------------------------- #
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def agg_recall(results, args=None):
    rows = valid_results(results, context="detection/recall")
    return _mean([float(r) for r in rows])


def agg_precision(results, args=None):
    rows = valid_results(results, context="detection/precision")
    return _mean([float(r) for r in rows])


def agg_f1(results, args=None):
    results = valid_results(results, context="detection/f1")
    if not results:
        return 0.0
    mr = _mean([r["recall"] for r in results])
    mp = _mean([r["precision"] for r in results])
    return 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0


def agg_mean_iou(results, args=None):
    results = valid_results(results, context="detection/mean_iou")
    values = [
        float(value)
        for per_sample in results
        for value in per_sample
    ]
    return _mean(values)


def agg_f1_mean_iou(results, args=None):
    results = valid_results(results, context="detection/F1mIoU")
    if not results:
        return 0.0
    threshold_f1 = []
    for index in range(len(IOU_THRESHOLDS)):
        mean_recall = _mean([float(row["recalls"][index]) for row in results])
        mean_precision = _mean(
            [float(row["precisions"][index]) for row in results]
        )
        threshold_f1.append(
            2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            if mean_precision + mean_recall > 0
            else 0.0
        )
    return _mean(threshold_f1)


def agg_mae(results, args=None):
    rows = valid_results(results, context="detection/mae")
    return _mean([float(r) for r in rows])
