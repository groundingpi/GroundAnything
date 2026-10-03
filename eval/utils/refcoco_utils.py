"""
lmms-lab RefCOCO / RefCOCOg / RefCOCO+ （HF parquet 数据集）任务的共享实现。

被 Referring/utils.py 引用（薄垫片 `from refcoco_utils import *`）。依赖 detection_utils.py
里的 `_maybe_downscale`/`_to_abs`/`_parse_predictions`/`_score_sample`/`REFER_INSTRUCTION`
（调用方的薄垫片已经把 utils/ 目录加进 sys.path，这里直接按模块名 import 即可）。

  数据格式: 每行 {image(PIL), bbox=[x,y,w,h], answer=[表达式列表], ...}
  做法: explode 表达式（一表达式一样本）→ referring_object_detection → 复用检测指标计算引擎。
  坐标: gt bbox[x,y,w,h]→绝对像素 xyxy；预测 [0,1000]→绝对像素（用原图 W/H）。
"""

from loguru import logger

from detection_utils import (
    REFER_INSTRUCTION,
    _maybe_downscale,
    _metric_payload,
    _failure_metric_payload,
    _deduplicate_box_mapping,
    _parse_predictions,
    _score_sample_iou_sweep,
    _to_abs,
    _deepseek_vl2_native_ref_prompt,
)
import os
from prompt_mode import (
    TaskType,
    build_mode_refer_bbox_prompt,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_to_abs,
    parse_mode_predictions_for_scoring,
)
from eval_io import record_image_failure, recorded_image_failure


def lmms_refcoco_process_docs(dataset):
    from datasets import Dataset

    rows = []
    for ex in dataset:
        img = ex["image"]
        try:
            w, h = img.size
        except Exception:
            w, h = 0, 0
        bbox = ex.get("bbox")
        answers = ex.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        qid = ex.get("question_id", "")
        for ans in answers:
            rows.append(
                {
                    "image": img,
                    "answer": ans,
                    "bbox": bbox,
                    "image_width": w,
                    "image_height": h,
                    "question_id": str(qid),
                }
            )
    new_ds = Dataset.from_list(rows)
    logger.info(f"[gam-refcoco] explode {len(dataset)} -> {len(new_ds)} 条")
    return new_ds


def lmms_refcoco_doc_to_visual(doc):
    try:
        return [_maybe_downscale(doc["image"].convert("RGB"))]
    except Exception as exc:
        record_image_failure(
            "refcoco",
            doc.get("question_id", ""),
            "hf://image",
            exc,
        )
        if is_gam_mode():
            return []
        from PIL import Image

        return [Image.new("RGB", (28, 28), (0, 0, 0))]


def lmms_refcoco_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if is_native_spatial_mode():
        return build_mode_refer_bbox_prompt(str(doc["answer"]), multiple=False)
    if os.environ.get("GAM_EVAL_VLM_FAMILY", "").lower() == "deepseek_vl2_native":
        return _deepseek_vl2_native_ref_prompt(str(doc["answer"]))
    return REFER_INSTRUCTION.format(phrase=doc["answer"])


def lmms_refcoco_process_results(doc, results):
    raw = results[0] if results else ""
    w = doc.get("image_width") or 1
    h = doc.get("image_height") or 1
    failure = recorded_image_failure("refcoco", doc.get("question_id", ""))
    if failure:
        return _failure_metric_payload(
            dict(failure, image_read_failed=True)
        )
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
        extracted.setdefault(label, []).append(absolute_box)
    # RefCOCO is referring: the query already identifies the target.  Ignore
    # generated label text and score the globally deduplicated boxes only.
    if is_gam_mode():
        all_boxes = [box for boxes in extracted.values() for box in boxes]
        extracted = _deduplicate_box_mapping({"object": all_boxes})
    else:
        extracted = _deduplicate_box_mapping(extracted)

    bbox = doc.get("bbox") or [0, 0, 0, 0]
    x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
    gt_label = "object" if is_gam_mode() else str(doc.get("answer", "obj"))
    gt = {gt_label: [[x, y, x + bw, y + bh]]}
    sample = {
        "gt": gt,
        "extracted_predictions": extracted,
        "task_name": "referring_object_detection",
        "dataset_name": "refcoco",
    }
    recalls, precisions, mae = _score_sample_iou_sweep(sample)
    return _metric_payload(recalls, precisions, mae)
