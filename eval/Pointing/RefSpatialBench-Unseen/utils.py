import os
import numpy as np
from PIL import Image
from typing import Callable
import re
import json
import sys

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
    mode_grid_point_to_pixel,
    parse_mode_predictions_for_scoring,
    parse_native_predictions,
)
from eval_io import image_failure_marker, record_image_failure, recorded_image_failure, valid_results
from coordinate_mode import vlm_point_to_pixel
from eval_data_root import dataset_path

def refspatialbench_doc_to_visual(doc):
    try:
        return [doc["image"].convert("RGB")]
    except Exception as exc:
        record_image_failure("gam_refspatial_unseen", doc.get("id", ""), "hf://image", exc)
        if is_gam_mode():
            return []
        return [Image.new("RGB", (28, 28), (0, 0, 0))]

def refspatialbench_doc_to_text(doc):
    """
    根据 doc 生成输入文本
    """
    if is_native_spatial_mode():
        return build_mode_refer_point_prompt(str(doc["object"]))
    suffix = ''' Output the point coordinates in JSON format.
For example:
[
{"point_2d": [x, y], "label": "point_1"}
]'''
    full_input_instruction = "Please locate the points of " + doc["object"] + "." + suffix
    return full_input_instruction


def _get_optim_prompt_text(doc):
    """与训练数据完全一致的 prompt 文本"""
    if is_native_spatial_mode():
        return build_mode_refer_point_prompt(str(doc["object"]))
    suffix = ' Output the point coordinates in JSON format.\nFor example:\n[\n{"point_2d": [x, y], "label": "point_1"}\n]'
    return "Please locate the points of " + doc["object"] + "." + suffix


def refspatialbench_doc_to_messages_optim(doc):
    """
    prompt_optim: 与训练数据格式对齐的 messages 构建函数。
    - 增加 system message "You are a helpful assistant."
    - prompt 文本与训练 JSONL 完全一致
    """
    image = doc["image"].convert("RGB")
    text = _get_optim_prompt_text(doc)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "url": image},
            {"type": "text", "text": text},
        ]},
    ]
    return messages

def parse_json_coords(text: str, width=640, height=480) -> np.ndarray:
    """
    解析 JSON 格式的坐标输出
    支持格式：[{"point_2d": [x, y], "label": "..."}]
    假设坐标是 0-1000 范围，需要转换为像素坐标
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    points = []
    def append_item(item):
        if not isinstance(item, dict):
            return
        coord = item.get("point_2d")
        if not isinstance(coord, list) or len(coord) != 2:
            return
        x, y = (round(value) for value in vlm_point_to_pixel(coord, width, height))
        points.append((x, y))
    
    # 方法1：尝试从代码块中提取 JSON（模型可能用 ```json ... ``` 包裹）
    try:
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 尝试直接解析整个文本
            json_str = text.strip()
        
        # 解析 JSON
        data = json.loads(json_str)
        
        # 处理列表格式 [{"point_2d": [x, y], ...}, ...]
        if isinstance(data, list):
            for item in data:
                append_item(item)
        
        # 处理单个对象格式 {"point_2d": [x, y], ...}
        elif isinstance(data, dict):
            append_item(data)
        
        if points:
            return np.array(points)
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        pass
    
    # 方法2：如果 JSON 解析失败，尝试用正则提取 "point_2d": [x, y]
    pattern = r'"point_2d"\s*:\s*\[\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    for x_str, y_str in matches:
        x_val = float(x_str) 
        y_val = float(y_str)
        
        # 转换 0-1000 坐标到像素坐标（使用四舍五入）
        x, y = (round(value) for value in vlm_point_to_pixel((x_val, y_val), width, height))
        points.append((x, y))
    
    # 方法3：最后尝试匹配任意的 [x, y] 或 (x, y) 格式
    if not points:
        pattern2 = r'[\[\(]\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*[\]\)]'
        matches = re.findall(pattern2, text)
        for x_str, y_str in matches:
            x_val = float(x_str)
            y_val = float(y_str)
            
            # 0-1000 坐标（使用四舍五入）
            x, y = (round(value) for value in vlm_point_to_pixel((x_val, y_val), width, height))
            
            points.append((x, y))
    
    return np.array(points) if points else np.empty((0, 2), dtype=int)


def parse_gam_coords(text: str, width=640, height=480) -> np.ndarray:
    """解析严格的 GAM native point token，并映射到 mask 像素。"""

    predictions = (
        parse_mode_predictions_for_scoring(text, TaskType.POINT)
        if is_native_spatial_mode()
        else parse_native_predictions(text, TaskType.POINT)
    )
    points = []
    seen = set()
    for _, point in predictions:
        pixel = mode_grid_point_to_pixel(point, width, height)
        if pixel not in seen:
            seen.add(pixel)
            points.append(pixel)
    return np.array(points, dtype=int) if points else np.empty((0, 2), dtype=int)

def compute_accuracy(mask_path, id, text, parse_func):
    """
    计算单条样本的准确率
    """
    accuracy = []

    mask = np.array(Image.open(mask_path)) / 255.
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)

    try:
        points = parse_func(text, mask.shape[1], mask.shape[0])
    except Exception as e:
        return 0.0

    acc = 0.0
    try:
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                       (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
    except:
        pass

    accuracy.append(acc)
    return np.mean(accuracy)


def refspatialbench_process_results(doc, model_output):
    """
    处理模型输出，计算单条结果是否正确，并返回包含 score 字段的字典
    """
    failure = recorded_image_failure("gam_refspatial_unseen", doc.get("id", ""))
    if failure:
        return {"score": dict(failure, image_read_failed=True)}
    mask_path = os.path.join(
        str(dataset_path("refspatial", "Unseen", "mask")),
        f"{doc['id']}.png"
    )
    if not os.path.exists(mask_path):
        return {"score": image_failure_marker("gam_refspatial_unseen", doc.get("id", ""), mask_path, FileNotFoundError(mask_path))}
    try:
        with Image.open(mask_path) as image:
            image.verify()
    except Exception as exc:
        return {"score": image_failure_marker("gam_refspatial_unseen", doc.get("id", ""), mask_path, exc)}

    # 如果 model_output 是列表，取第一个元素
    if isinstance(model_output, list):
        model_output = model_output[0]

    # 调用 compute_accuracy 计算准确率（使用新的 JSON 解析函数）
    acc = compute_accuracy(
        mask_path,
        doc["id"],
        model_output,
        lambda text, w, h: (
            parse_gam_coords(text, w, h)
            if is_native_spatial_mode()
            else parse_json_coords(text, w, h)
        )
    )

    # 返回分数
    # return {"score": 1.0 if acc == 1 else 0.0}
    return {"score": acc}  # 线性评分，0.0-1.0 


def aggregate_score(results, args=None):
    rows = valid_results(results, context="gam_refspatial_unseen/score")
    return sum(float(value) for value in rows) / len(rows) if rows else 0.0
