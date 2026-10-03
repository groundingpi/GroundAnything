"""Canonical GAM route taxonomy used by the RL V3 data builder.

The post-training YAML exposes five coarse input routes, but the GAM metric
suite contains eleven semantically distinct output contracts.  RL sampling
must preserve the latter; otherwise a nominally balanced ``point`` or
``bbox_grounding`` pool can still omit entire benchmark families.
"""

from __future__ import annotations


ROUTES = (
    "grounding",
    "referring",
    "dense",
    "grounding_point",
    "referring_point",
    "dense_point",
    "robo_point",
    "visual_prompt",
    "gui",
    "layout",
    "ocr",
)


SOURCE_TO_ROUTE = {
    # BBox grounding families.
    "gam_lvis": "grounding",
    "gam_coco": "grounding",
    "gam_refcoco": "referring",
    "gam_refcocog": "referring",
    "gam_refcocoplus": "referring",
    "gam_refcocog_val": "referring",
    "gam_refcocog_test": "referring",
    "gam_humanref": "referring",
    "gam_visdrone": "dense",
    "gam_dense200": "dense",
    # Point-in-mask and robotics point families.
    "gam_rex_point_lvis": "grounding_point",
    "gam_rex_point_coco": "grounding_point",
    "gam_rex_point_refcocog_val": "referring_point",
    "gam_rex_point_refcocog_test": "referring_point",
    "gam_rex_point_humanref": "referring_point",
    "gam_rex_point_visdrone": "dense_point",
    "gam_rex_point_dense200": "dense_point",
    "gam_robospatial_context": "robo_point",
    "gam_refspatial_location": "robo_point",
    "gam_refspatial_placement": "robo_point",
    "gam_refspatial_unseen": "robo_point",
    # Visual-prompt, GUI, document-layout and OCR families.
    "gam_visual_lvis": "visual_prompt",
    "gam_visual_coco": "visual_prompt",
    "gam_visual_dense200": "visual_prompt",
    "gam_fsc147": "visual_prompt",
    "gam_screenspot_pro": "gui",
    "gam_screenspot_v2": "gui",
    "gam_osworld_g": "gui",
    "gam_doclaynet": "layout",
    "gam_m6doc": "layout",
    "gam_hiertext": "ocr",
    "gam_icdar2015": "ocr",
    "gam_sroie": "ocr",
    "gam_totaltext": "ocr",
}


COARSE_ROUTE = {
    "grounding": "bbox_grounding",
    "referring": "bbox_grounding",
    "dense": "bbox_grounding",
    "grounding_point": "point",
    "referring_point": "point",
    "dense_point": "point",
    "robo_point": "point",
    "visual_prompt": "visual_prompt",
    "gui": "gui_layout",
    "layout": "gui_layout",
    "ocr": "ocr",
}


def semantic_route(source_name: str) -> str:
    """Return the exact semantic route for one SFT2 source."""

    try:
        return SOURCE_TO_ROUTE[source_name]
    except KeyError as exc:
        raise ValueError(f"unmapped GAM source: {source_name!r}") from exc


def validate_schema() -> None:
    missing = set(ROUTES) - set(SOURCE_TO_ROUTE.values())
    if missing:
        raise RuntimeError(f"GAM route schema has no source for: {sorted(missing)}")
    if set(COARSE_ROUTE) != set(ROUTES):
        raise RuntimeError("coarse-route mapping does not cover the canonical routes")


validate_schema()
