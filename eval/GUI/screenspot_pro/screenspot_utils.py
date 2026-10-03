"""
ScreenSpot-Pro utilities for lmms_eval
Based on ScreenSpot-Pro-GUI-Grounding evaluation framework

完全对齐原始代码的 prompt 格式和坐标解析逻辑
"""

import json
import os
import re
import glob
import importlib.util
import sys
from pathlib import Path
from loguru import logger as eval_logger
from PIL import Image

_GUI_PROMPT_PATH = Path(__file__).resolve().with_name("prompts.py")
_GUI_PROMPT_SPEC = importlib.util.spec_from_file_location(
    "_gam_screenspot_gui_agent_prompt", _GUI_PROMPT_PATH
)
if _GUI_PROMPT_SPEC is None or _GUI_PROMPT_SPEC.loader is None:
    raise ImportError(f"cannot load ScreenSpot-Pro prompt registry: {_GUI_PROMPT_PATH}")
_GUI_PROMPTS = importlib.util.module_from_spec(_GUI_PROMPT_SPEC)
_GUI_PROMPT_SPEC.loader.exec_module(_GUI_PROMPTS)
GUI_MOBILE_PROMPT = _GUI_PROMPTS.GUI_MOBILE_PROMPT
GUI_PC_PROMPT = _GUI_PROMPTS.GUI_PC_PROMPT
GUI_single_PROMPT = _GUI_PROMPTS.GUI_single_PROMPT

_SHARED_UTILS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "utils",
)
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)

from prompt_mode import (
    TaskType,
    build_mode_gui_prompt,
    is_gam_mode,
    is_native_spatial_mode,
    mode_grid_point_to_norm,
    parse_mode_predictions_for_scoring,
)
from eval_io import record_image_failure, recorded_image_failure, valid_results
from eval_data_root import dataset_path
from coordinate_mode import vlm_point_to_norm


# ============= Model Configuration Mapping =============
MODEL_CONFIGS = {
    # Qwen2.5-VL style models (绝对像素坐标)
    "GUI-Owl-7B": {
        "parse_method": "qwen2_5",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",
        "description": "GUI-Owl-7B with qwen2_5 function calling, absolute pixel coordinates"
    },
    "Qwen2.5-VL": {
        "parse_method": "qwen2_5",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",
        "description": "Qwen2.5-VL with function calling, absolute pixel coordinates"
    },
    
    # Qwen3-VL style models (0-1000 坐标范围)
    "Qwen3-VL": {
        "parse_method": "qwen3",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",  # 使用相同的 prompt 格式
        "description": "Qwen3-VL with function calling, 0-1000 coordinate range"
    },

    # DeepSeek-VL2 may follow its UI-agent computer_use tool-call contract
    # even under the fixed VLM prompt.  Keep this parser family-routed.
    "DeepSeek-VL2": {
        "parse_method": "deepseek_vl2",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",
        "description": "DeepSeek-VL2 computer_use tool calls with routed coordinates",
    },

    # BAGEL's official runtime may return either a computer_use JSON point or
    # a bare point/box.  Route it through the explicit per-prediction auto
    # coordinate contract; do not impersonate Qwen3's fixed 0..1000 parser.
    "BAGEL": {
        "parse_method": "bagel",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",
        "description": "BAGEL computer_use output with per-point auto coordinates",
    },

    # Kimi K3 accepts the canonical Qwen computer_use prompt but emits point
    # coordinates normalized to [0,1].  Keep the prompt identical and route
    # only coordinate restoration through GAM_COORD_MODE=norm01.
    "Kimi-K3": {
        "parse_method": "vlm_routed",
        "use_grounding_doubao": False,
        "prompt_type": "qwen2_5",
        "description": "Kimi K3 computer_use calls with routed VLM coordinates",
    },
    
    # UI-TARS models
    "uitars-1.5": {
        "parse_method": "qwen2_5",  # uitars1_5 uses absolute pixels
        "use_grounding_doubao": True,
        "grounding_doubao_style": "uitars1_5",  # Simplified prompt (without extra instructions)
        "description": "UI-TARS 1.5 with simplified GROUNDING_DOUBAO prompt, absolute pixels"
    },
    "uitars-7b-sft": {
        "parse_method": "uitars_0_1000",
        "use_grounding_doubao": True,
        "grounding_doubao_style": "uitars_7b",  # Full prompt (with "You MUST respond...")
        "description": "UI-TARS 7B SFT with full GROUNDING_DOUBAO prompt, 0-1000 range"
    },
    "uitars-72b-dpo": {
        "parse_method": "uitars_0_1000",
        "use_grounding_doubao": True,
        "grounding_doubao_style": "uitars_7b",  # Full prompt (with "You MUST respond...")
        "description": "UI-TARS 72B DPO with full GROUNDING_DOUBAO prompt, 0-1000 range"
    },
    "gui_agent": {
        "parse_method": "uitars_0_1000",
        "use_grounding_doubao": False,
        "prompt_type": "gui_agent",
        "description": "gui_agent with function calling, 0-1000 coordinate range"
        },
}


DEFAULT_MODEL_CONFIG = {
    "parse_method": "gui_agent",
    "use_grounding_doubao": False,
    "prompt_type": "gui_agent_single_prompt",
    "description": "gui_agent with function calling, 0-1000 coordinate range"
}

def normalize_model_name(name):
    """Normalize model name for matching"""
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def get_model_config(model_name=None):
    """Get model configuration based on model name"""
    if model_name is None:
        model_name = os.environ.get("SCREENSPOT_MODEL_NAME", "")
    
    model_name = model_name.strip()
    if not model_name:
        eval_logger.warning("SCREENSPOT_MODEL_NAME not set, using default config")
        return DEFAULT_MODEL_CONFIG
    
    normalized_model_name = normalize_model_name(model_name)
    
    for config_model_name, config in MODEL_CONFIGS.items():
        normalized_config_name = normalize_model_name(config_model_name)
        if (normalized_config_name in normalized_model_name or 
            normalized_model_name in normalized_config_name):
            return config
    
    eval_logger.warning(f"No config found for model '{model_name}', using default config")
    return DEFAULT_MODEL_CONFIG


# ============= Configuration =============
def get_data_root(lmms_eval_specific_kwargs=None):
    """Get data root directory"""
    if lmms_eval_specific_kwargs is not None and "dataset_path" in lmms_eval_specific_kwargs:
        return lmms_eval_specific_kwargs["dataset_path"]
    
    return os.environ.get(
        "SCREENSPOT_DATA_ROOT", str(dataset_path("screenspot_pro"))
    )


def _task_id(lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs:
        return str(lmms_eval_specific_kwargs.get("task_id", "gam_screenspot_pro"))
    return "gam_screenspot_pro"


def _image_path(doc, lmms_eval_specific_kwargs=None):
    data_root = get_data_root(lmms_eval_specific_kwargs)
    image_subdir = "images"
    if lmms_eval_specific_kwargs:
        image_subdir = str(lmms_eval_specific_kwargs.get("image_subdir", image_subdir))
    return os.path.join(data_root, image_subdir, doc["img_filename"])


# ============= Data Preprocessing =============
def screenspot_pro_process_docs(dataset):
    """Process the loaded JSON dataset"""
    data = [item for item in dataset]
    import datasets
    return datasets.Dataset.from_list(data)


# ============= Data Loading =============
def screenspot_pro_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Load image from document
    
    重要：计算模型实际使用的 resized 图片尺寸，保存到 doc 中供后续使用
    """
    img_path = _image_path(doc, lmms_eval_specific_kwargs)
    
    #from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        record_image_failure(
            _task_id(lmms_eval_specific_kwargs),
            doc.get("id", doc.get("img_filename", "")),
            img_path,
            exc,
        )
        if is_gam_mode():
            return []
        image = Image.new("RGB", (28, 28), (0, 0, 0))
    
    
    # # 保存 resized 尺寸到 doc（用于后续坐标归一化）
    # # 注意：这里直接修改 doc 是安全的，因为 doc 是字典引用
    # doc["resized_img_size"] = [resized_width, resized_height]
    
    # eval_logger.debug(f"Original size: {image.width}x{image.height}, Resized: {resized_width}x{resized_height}")
    
    # 返回原始图片（vLLM 会自行 resize）
    # 或者返回 resized 图片以避免重复 resize（但需要确保 vLLM 配置一致）
    return [image]


def screenspot_pro_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Build messages for ScreenSpot-Pro (推荐使用此函数)
    
    完全对齐本地代码 ScreenSpot-Pro-GUI-Grounding/models/：
    - qwen2_5vl.py: get_qwen2_5vl_prompt_msg (第 20-67 行)
    - uitars.py: GROUNDING_DOUBAO (第 38-53 行)
    
    返回标准的 messages 格式：
    [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "image", "url": "..."},
            {"type": "text", "text": "..."}
        ]}
    ]
    """
    import json
    import os

    # Chat-model backends call doc_to_messages directly and may never invoke
    # doc_to_visual.  Preflight here as well; on failure send text only so the
    # worker survives and process_results can exclude the registered sample.
    if is_gam_mode() and not screenspot_pro_doc_to_visual(
        doc, lmms_eval_specific_kwargs
    ):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": screenspot_pro_doc_to_text(
                            doc, lmms_eval_specific_kwargs
                        ),
                    }
                ],
            }
        ]
    
    # Get instruction
    language = "en"
    if lmms_eval_specific_kwargs:
        language = lmms_eval_specific_kwargs.get("language", "en")
    
    if language == "cn":
        instruction = doc.get("instruction_cn", doc["instruction"])
    else:
        instruction = doc["instruction"]
    
    # Get model config
    model_config = get_model_config()
    prompt_type = model_config.get("prompt_type", "qwen2_5")
    use_grounding_doubao = model_config.get("use_grounding_doubao", False)
    
    
    #img_width, img_height = doc["resized_img_size"]
    img_width, img_height = doc["img_size"]
    
    # Get image path
    image_path = _image_path(doc, lmms_eval_specific_kwargs)

    if is_native_spatial_mode():
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {
                        "type": "text",
                        "text": build_mode_gui_prompt(str(instruction)),
                    },
                ],
            },
        ]
    
    if use_grounding_doubao:
        # UI-TARS style: GROUNDING_DOUBAO format
        # Two different styles based on model version
        grounding_doubao_style = model_config.get("grounding_doubao_style", "uitars_7b")
        
        system_text = "You are a helpful assistant."
        
        if grounding_doubao_style == "uitars1_5":
            # 对齐 uitars1_5.py 第 38-39 行：简化版本（无额外指令）
            user_text = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""
        else:  # uitars_7b style (default)
            # 对齐 uitars.py 第 38-53 行：完整版本（带额外指令）
            user_text = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format

You MUST respond with the Action format shown below. DO NOT provide conversational text.

Action: click(point='<point>x1 y1</point>')

## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{instruction}

IMPORTANT: Respond ONLY with the Action format. DO NOT add any explanation or text before or after the Action."""
        
    elif prompt_type == "qwen2_5":
        # Qwen2.5-VL / Qwen3-VL style: function calling
        # 对齐 qwen2_5vl.py 第 20-67 行
        
        # Build computer_use function (对齐 qwen2_5vl.py 第 40 行)
        computer_use_desc = f"""Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {img_width}x{img_height}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""
        
        action_desc = """The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status."""
        
        computer_use_function = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": computer_use_desc,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_desc,
                            "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", 
                                   "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                            "type": "string"
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {
                            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                            "type": "array"
                        },
                        "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"},
                        "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"},
                        "status": {
                            "description": "The status of the task. Required only by `action=terminate`.",
                            "type": "string",
                            "enum": ["success", "failure"]
                        }
                    },
                    "required": ["action"],
                    "type": "object"
                },
                "args_format": "Format the arguments as a JSON object."
            }
        }
        
        tool_descs_json = json.dumps(computer_use_function, ensure_ascii=False)
        
        # Build system message (对齐 qwen2_5vl.py 第 21-48 行)
        # 注意：本地代码的 system role 包含两个 text content
        system_text_1 = "You are a helpful assistant."
        
        # 是否添加输出格式示例（模拟 guide_text 效果）
        add_output_example = os.environ.get("SCREENSPOT_ADD_OUTPUT_EXAMPLE", "true").lower() == "true"
        
        if add_output_example:
            output_example = """

## Output Format Example

For clicking tasks, you MUST respond in this exact format:
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}
</tool_call>

Where x and y are the pixel coordinates. Do NOT add any explanation before or after the tool_call."""
        else:
            output_example = ""
        
        system_text_2 = f"""


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>{output_example}"""
        
        user_text = instruction
        
        # 模拟 guide_text：在 user message 末尾添加引导前缀
        # 对齐 qwen2_5vl.py 第 174 行的 guide_text
        # 虽然无法在 assistant token 之后直接添加，但可以在 user message 末尾引导
        use_guide_text = os.environ.get("SCREENSPOT_USE_GUIDE_TEXT", "true").lower() == "true"
        
        if use_guide_text:
            # 添加明确的输出格式引导
            guide_hint = """

Please respond with the tool call in the following format (continue from where I left off):
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ["""
            user_text = user_text + guide_hint
        
        # Build messages structure (完全对齐 qwen2_5vl.py 第 21-67 行)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_text_1},
                    {"type": "text", "text": system_text_2}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        
        return messages
    
    
    elif prompt_type == "qwen3":
        
        
        messages=[
            { "role": "system",
             "content": 
                 [
                    {
                    "type": "text", 
                    "text": "You are a precise screen reader agent. You receive a screenshot and a query. You must output the normalized coordinates (0-1000 scale) of the geometric center of the target UI element. Do not output any other text."
                    }
                 ]
            }, 
            { "role": "user", 
             "content": [
                 {"type": "image", "url": image_path}, 
                 {"type": "text", "text": f"Target: {instruction}\n\nIdentify the specific UI element described. Visualize its bounding box borders, then pinpoint the exact center (x,y) on a 1000x1000 coordinate grid.\n\nOutput only the tuple: (x,y)"}
                 ]
            }
            ]
        
        
        return messages
    
    
    elif prompt_type == "qwen3_2":
        system_text = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is {screen_width}x{screen_height}.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y).\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
        messages=[
            { "role": "system",
             "content": 
                 [{"type": "text", "text": system_text}]
            }, 
            { "role": "user", 
             "content": [
                 {"type": "image", "url": image_path}, 
                 { "type": "text", "text": f"{instruction}"}
                 ]
             }
            ]
        
        
        return messages
    
    elif prompt_type == "gui_agent":
        
        system_text = GUI_PC_PROMPT
#         user_text = """<image>
# Click the button or icon based on this task <task>."""

 
#63.31  26000     
        user_text = """<image>
Please help me click the button based on its features and functions: <task>"""

#         user_text = """<image>
# <task>"""

#         user_text = """<image>
# Please generate the next move according to the UI screenshot, task and previous operations.

# Task: Please help me click the button based on its features and functions: <task>

# Previous operations:
# None
# """
        user_text=user_text.replace("<task>",instruction)
        # Build messages structure
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        return messages
    
    elif prompt_type == "gui_agent_single_prompt":
        system_text = GUI_single_PROMPT 
        user_text = """<image>
Please help me click the button based on its features and functions: <task>"""

        user_text=user_text.replace("<task>",instruction)
        # Build messages structure
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        return messages
    
    else:
        # Fallback
        system_text = "You are a helpful assistant."
        user_text = instruction
    
    # Build messages structure
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": user_text}
            ]
        }
    ]
    
    return messages


def screenspot_pro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Build prompt for ScreenSpot-Pro
    
    完全对齐原始代码：
    - qwen2_5vl.py: get_qwen2_5vl_prompt_msg + apply_chat_template + guide_text
    - qwen3vl.py: same as qwen2_5vl.py
    - uitars.py / uitars1_5.py: GROUNDING_DOUBAO + apply_chat_template
    
    注意：lmms_eval 框架限制，无法完全模拟 guide_text 的添加方式
    """
    import json
    
    # Get instruction
    language = "en"
    if lmms_eval_specific_kwargs:
        language = lmms_eval_specific_kwargs.get("language", "en")
    
    if language == "cn":
        instruction = doc.get("instruction_cn", doc["instruction"])
    else:
        instruction = doc["instruction"]

    if is_native_spatial_mode():
        return build_mode_gui_prompt(str(instruction))
    
    # Get model configuration
    model_config = get_model_config()
    
    # Determine prompt format
    env_override = os.environ.get("SCREENSPOT_USE_GROUNDING_DOUBAO", "").lower()
    if env_override == "true":
        use_grounding_doubao = True
    elif env_override == "false":
        use_grounding_doubao = False
    else:
        use_grounding_doubao = model_config.get("use_grounding_doubao", False)
    
    prompt_type = model_config.get("prompt_type", "qwen2_5")
    
    if use_grounding_doubao:
        # UI-TARS style: GROUNDING_DOUBAO format
        # 对齐 uitars.py 第 38-53 行
        grounding_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format

You MUST respond with the Action format shown below. Do NOT provide conversational text.

Action: click(point='<point>x1 y1</point>')

## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{instruction}

IMPORTANT: Respond ONLY with the Action format. DO NOT add any explanation or text before or after the Action.

assistant
"""
        return grounding_prompt.format(instruction=instruction)
    
    elif prompt_type == "qwen2_5":
        # Qwen2.5-VL / Qwen3-VL style: function calling with computer_use
        # 对齐 qwen2_5vl.py 第 20-67 行和 qwen3vl.py 第 20-67 行
        
        
        # img_width, img_height = doc["resized_img_size"]
        img_width, img_height = doc["img_size"]
        
        # Build computer_use function (对齐 qwen2_5vl.py 第 40 行)
        computer_use_desc = f"""Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {img_width}x{img_height}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""
        
        action_desc = """The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status."""
        
        computer_use_function = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": computer_use_desc,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_desc,
                            "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                            "type": "string"
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {
                            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                            "type": "array"
                        },
                        "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"},
                        "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"},
                        "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}
                    },
                    "required": ["action"],
                    "type": "object"
                },
                "args_format": "Format the arguments as a JSON object."
            }
        }
        
        tool_descs_json = json.dumps(computer_use_function, ensure_ascii=False)
        
        # Build complete prompt (对齐 qwen2_5vl.py 第 27-46 行)
        prompt = f"""You are a helpful assistant.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

{instruction}

assistant
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": ["""
        return prompt
    
    else:
        # Fallback
        return instruction


# ============= Coordinate Parsing =============
def parse_coordinates_qwen2_5(response, img_size):
    """
    Parse Qwen2.5-VL coordinates (absolute pixels)
    对齐 qwen2_5vl.py 第 217-228 行和 uitars1_5.py 第 248-289 行
    
    支持格式：
    1. JSON function calling: {"name": "computer_use", "arguments": {"coordinate": [x, y]}}
    2. <point>x y</point>
    3. <|box_start|>(x,y)<|box_end|> (UI-TARS point format)
    4. <|box_start|>(x1,y1),(x2,y2)<|box_end|> (UI-TARS bbox format)
    """
    try:
        # Priority 1: UI-TARS <|box_start|> format (对齐 uitars1_5.py 第 248-275 行)
        if '<|box_start|>' in response and '<|box_end|>' in response:
            # Try bbox format first (4 coordinates)
            pattern_bbox = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
            matches_bbox = re.findall(pattern_bbox, response)
            if matches_bbox:
                x1, y1, x2, y2 = map(int, matches_bbox[-1])
                # Normalize bbox using img_size (resized dimensions)
                bbox_normalized = [x1 / img_size[0], y1 / img_size[1],
                                 x2 / img_size[0], y2 / img_size[1]]
                # Return center point
                point_normalized = [(bbox_normalized[0] + bbox_normalized[2]) / 2,
                                   (bbox_normalized[1] + bbox_normalized[3]) / 2]
                return point_normalized
            
            # Try point format (2 coordinates in box tags)
            pattern_point_in_box = r"<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>"
            matches_point = re.findall(pattern_point_in_box, response)
            if matches_point:
                point_x, point_y = map(int, matches_point[-1])
                # Normalize using img_size (resized dimensions)
                point_normalized = [point_x / img_size[0], point_y / img_size[1]]
                return point_normalized
        
        # Priority 2: JSON function calling format (对齐 qwen2_5vl.py 第 217-228 行)
        if '<tool_call>' in response and '</tool_call>' in response:
            json_str = response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
        else:
            # Fallback: find JSON object
            start_idx = response.find('{"name":')
            if start_idx != -1:
                json_str = response[start_idx:]
                end_idx = json_str.rfind('}')
                if end_idx != -1:
                    json_str = json_str[:end_idx + 1]
                    # Parse JSON
                    action = json.loads(json_str)
                    coordinates = action['arguments']['coordinate']
                    
                    # Handle point or bbox
                    if len(coordinates) == 2:
                        point_x, point_y = coordinates
                    elif len(coordinates) == 4:
                        x1, y1, x2, y2 = coordinates
                        point_x = (x1 + x2) / 2
                        point_y = (y1 + y2) / 2
                    else:
                        return None
                    
                    # Normalize
                    return [point_x / img_size[0], point_y / img_size[1]]
        
        # Priority 3: Fallback to simple number extraction (对齐 uitars1_5.py 第 277-289 行)
        click_point = re.findall(r"\d+", response)
        if len(click_point) >= 2:
            point_x = int(click_point[0])
            point_y = int(click_point[1])
            # Normalize
            return [point_x / img_size[0], point_y / img_size[1]]
        
        return None
        
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        eval_logger.debug(f"Failed to parse qwen2.5 coordinates ({type(e).__name__})")
        return None


# import re

    
#     match = re.search(pattern, text)
    
        

import re

def extract_coordinate(response_text):
    """
    从模型返回的文本中解析 (x, y) 坐标。
    兼容性：
    1. 纯坐标: "(500, 500)" 或 "[500, 500]"
    2. 思维链: "Found button at... center is (500, 500)"
    3. Markdown: "```(500, 500)```" 或 "**(500, 500)**"
    4. 容错: 允许 x,y 之间有空格，允许使用圆括号或方括号
    
    返回:
        tuple(int, int): (x, y) 坐标
        None: 如果无法解析
    """
    if not response_text:
        return None

    # 正则表达式解释：
    # 1. [\(\[]      : 匹配开头的 '(' 或 '['
    # 2. \s*         : 允许任意空白字符
    # 3. (\d+)       : 捕获第一组数字 (x)
    # 4. ,           : 匹配逗号
    # 5. (\d+)       : 捕获第二组数字 (y)
    # 6. [\)\]]      : 匹配结尾的 ')' 或 ']'
    pattern = r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]"
    
    # 使用 findall 找出所有匹配项
    matches = re.findall(pattern, response_text)
    
    if matches:
        # 策略：取最后一个匹配项 (Last Match Strategy)
        # 原因：在 CoT (思维链) 模式下，模型通常先分析，最后才给出确定的结论坐标。
        last_match = matches[-1]
        try:
            x = int(last_match[0])
            y = int(last_match[1])
            return x, y
        except ValueError:
            return None
            
    # --- 备用方案 (Fallback) ---
    # 如果模型完全不听话，没有加括号，只输出了 "500, 500" 这种形式
    # 我们尝试匹配行尾的两个数字
    fallback_pattern = r"(\d+)\s*,\s*(\d+)"
    matches_fallback = re.findall(fallback_pattern, response_text)
    
    if matches_fallback:
        # 同样取最后一个，但这种匹配风险较大，仅作为最后的尝试
        last_match = matches_fallback[-1]
        return int(last_match[0]), int(last_match[1])

    return None

def parse_coordinates_gui_agent_format(response, img_size):
    
    try:
        point_x,point_y=extract_coordinate(response)
        return [point_x / 1000.0, point_y / 1000.0]
        
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        eval_logger.debug(f"Failed to parse qwen3 coordinates ({type(e).__name__})")
        return None

def parse_coordinates_qwen3(response, img_size):
    """
    Parse Qwen3-VL coordinates (0-1000 range)
    对齐 qwen3vl.py 第 238-264 行
    """
    try:
        # Extract JSON
        if '<tool_call>' in response:
            json_str = response.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
        else:
            start_idx = response.find('{"name":')
            if start_idx == -1:
                return None
            json_str = response[start_idx:]
            end_idx = json_str.rfind('}')
            if end_idx != -1:
                json_str = json_str[:end_idx + 1]
        
        # Parse JSON
        action = json.loads(json_str)
        coordinates = action['arguments']['coordinate']
        # Handle point (对齐第 240-246 行)
        if len(coordinates) == 2:
            point_x, point_y = coordinates
            if 0 <= point_x <= 1000 and 0 <= point_y <= 1000:
                return [point_x / 1000.0, point_y / 1000.0]
        # Handle bbox (对齐第 247-256 行)
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            if all(0 <= coord <= 1000 for coord in [x1, y1, x2, y2]):
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
                return [point_x / 1000.0, point_y / 1000.0]
        
        return None

    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        eval_logger.debug(f"Failed to parse qwen3 coordinates ({type(e).__name__})")
        return None


def parse_coordinates_deepseek_vl2(response, img_size):
    """Parse DeepSeek computer_use calls or one standalone numeric point."""

    decoder = json.JSONDecoder()
    payloads = []
    for match in re.finditer(r"\{", response):
        try:
            payload, _ = decoder.raw_decode(response, match.start())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    for payload in reversed(payloads):
        arguments = payload.get("arguments", payload)
        if not isinstance(arguments, dict):
            continue
        coordinate = arguments.get("coordinate") or arguments.get("coordinates")
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) not in (2, 4):
            continue
        try:
            if len(coordinate) == 4:
                x = (float(coordinate[0]) + float(coordinate[2])) / 2.0
                y = (float(coordinate[1]) + float(coordinate[3])) / 2.0
            else:
                x, y = (float(coordinate[0]), float(coordinate[1]))
            point = vlm_point_to_norm((x, y), img_size[0], img_size[1])
        except (TypeError, ValueError):
            continue
        if all(0.0 <= value <= 1.0 for value in point):
            return list(point)

    # The full DeepSeek-VL2 checkpoint occasionally follows the instruction's
    # compact point contract and returns exactly ``x, y`` instead of wrapping
    # it in computer_use JSON.  Accept only a whole-response pair so numbers in
    # explanations, image sizes, or repeated generations cannot be mined as a
    # prediction.
    number = r"[-+]?\d+(?:\.\d+)?"
    bare = re.fullmatch(
        rf"\s*[\(\[]?\s*({number})\s*,\s*({number})\s*[\)\]]?\s*",
        response,
    )
    if bare:
        try:
            point = vlm_point_to_norm(
                (float(bare.group(1)), float(bare.group(2))),
                img_size[0],
                img_size[1],
            )
        except (TypeError, ValueError):
            return None
        if all(0.0 <= value <= 1.0 for value in point):
            return list(point)
    return None


def parse_coordinates_vlm_routed(response, img_size):
    """Parse a computer_use JSON point using the explicit VLM coord mode."""

    return parse_coordinates_deepseek_vl2(response, img_size)


def parse_coordinates_bagel(response, img_size):
    """Parse BAGEL GUI points without assuming a global coordinate scale."""

    parsed = parse_coordinates_deepseek_vl2(response, img_size)
    if parsed is not None:
        return parsed
    # Official BAGEL occasionally emits a bare [x,y] or [x1,y1,x2,y2].  Only
    # accept one complete bracketed numeric value; never mine prose numbers.
    matches = re.findall(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)(?:\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?))?\s*\]", response)
    if not matches:
        return None
    x1, y1, x2, y2 = matches[-1]
    values = [float(x1), float(y1)]
    if x2 and y2:
        values = [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]
    try:
        point = vlm_point_to_norm(values, img_size[0], img_size[1])
    except (TypeError, ValueError):
        return None
    return list(point) if all(0.0 <= value <= 1.0 for value in point) else None


def parse_coordinates_uitars_0_1000(response, img_size):
    """
    Parse UI-TARS coordinates (0-1000 range)
    对齐 uitars.py 第 252-301 行
    """
    try:
        # Check bbox format (对齐第 254-271 行)
        if '<|box_start|>' in response and '<|box_end|>' in response:
            # Try 4-coordinate bbox
            pattern_bbox = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
            matches = re.findall(pattern_bbox, response)
            if matches:
                x1, y1, x2, y2 = map(int, matches[-1])
                if all(0 <= coord <= 1000 for coord in [x1, y1, x2, y2]):
                    # Normalize bbox (对齐第 263 行)
                    bbox_normalized = [pos / 1000.0 for pos in [x1, y1, x2, y2]]
                    point_normalized = [(bbox_normalized[0] + bbox_normalized[2]) / 2,
                                       (bbox_normalized[1] + bbox_normalized[3]) / 2]
                    return point_normalized
            
            # Try 2-coordinate point in box tags (对齐第 272-283 行)
            pattern_point_in_box = r"<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>"
            matches = re.findall(pattern_point_in_box, response)
            if matches:
                point_x, point_y = map(int, matches[-1])
                if 0 <= point_x <= 1000 and 0 <= point_y <= 1000:
                    return [point_x / 1000.0, point_y / 1000.0]
        
        # Fallback: simple point format (对齐第 285-301 行)
        click_point = re.findall(r"\d+", response)
        if len(click_point) >= 2:
            point_x = int(click_point[-2])
            point_y = int(click_point[-1])
            
            if 0 <= point_x <= 1000 and 0 <= point_y <= 1000:
                return [point_x / 1000.0, point_y / 1000.0]
        
        return None
        
    except (IndexError, ValueError) as e:
        eval_logger.debug(f"Failed to parse uitars_0_1000 coordinates ({type(e).__name__})")
        return None


def parse_coordinates(response, img_size, parse_method=None):
    """Universal coordinate parsing"""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    if parse_method is None:
        model_config = get_model_config()
        parse_method = model_config.get("parse_method", "qwen2_5")
    
    if parse_method == "qwen2_5":
        return parse_coordinates_qwen2_5(response, img_size)
    elif parse_method == "qwen3":
        return parse_coordinates_qwen3(response, img_size)
    elif parse_method == "deepseek_vl2":
        return parse_coordinates_deepseek_vl2(response, img_size)
    elif parse_method == "vlm_routed":
        return parse_coordinates_vlm_routed(response, img_size)
    elif parse_method == "bagel":
        return parse_coordinates_bagel(response, img_size)
    elif parse_method == "gui_agent":
        return parse_coordinates_gui_agent_format(response, img_size)
    elif parse_method == "uitars_0_1000":
        return parse_coordinates_uitars_0_1000(response, img_size)
    else:
        eval_logger.warning(f"Unknown parse_method: {parse_method}, using qwen2_5")
        return parse_coordinates_qwen2_5(response, img_size)


# ============= Evaluation =============
def eval_sample_positive_gt(doc, pred_point):
    """
    Evaluate if predicted point falls within ground truth bbox
    对齐 eval_screenspot_pro_parallel.py 第 184-197 行
    """
    if pred_point is None:
        return "wrong_format"
    
    
    
    
    
    
    
    
    #qwen3vl不用resize
    bbox = doc["bbox"]
    original_size = doc["img_size"]  # 原始尺寸
    # 归一化 bbox（现在在 resized 空间）
    bbox_norm = [
        bbox[0] / original_size[0],
        bbox[1] / original_size[1],
        bbox[2] / original_size[0],
        bbox[3] / original_size[1]
    ]
    
    # Check if point is within bbox (对齐第 194-197 行)
    # pred_point 也是基于 resized 尺寸归一化的，现在可以正确比较了
    if (bbox_norm[0] <= pred_point[0] <= bbox_norm[2] and 
        bbox_norm[1] <= pred_point[1] <= bbox_norm[3]):
        return "correct"
    else:
        return "wrong"


# ============= Results Processing =============
def screenspot_pro_process_results(doc, results, lmms_eval_specific_kwargs=None):
    """Process model output and evaluate
    
    对齐 android_control 的嵌套结构：
    返回 {metric_name: {详细信息字典}}
    """
    response = results[0] if isinstance(results, list) else results
    marker = recorded_image_failure(
        _task_id(lmms_eval_specific_kwargs),
        doc.get("id", doc.get("img_filename", "")),
    )
    if marker:
        marker = dict(marker)
        marker.update({"raw_response": response, "img_filename": doc.get("img_filename", "")})
        return {"action_acc": marker, "parse_error_rate": marker}
    
    
    #img_size = doc["resized_img_size"]
    img_size = doc["img_size"]
    gam_mode = is_native_spatial_mode()
    gam_pred_points = None
    if gam_mode:
        native_predictions = parse_mode_predictions_for_scoring(
            response, TaskType.POINT
        )
        gam_pred_points = [
            mode_grid_point_to_norm(point) for _, point in native_predictions
        ]
        pred_point = (
            gam_pred_points[0]
            if gam_pred_points
            else None
        )
    else:
        pred_point = parse_coordinates(response, img_size)

    # GUI is a single-point task.  Score only the first parsed point and do not
    # gate it on the object-ref label; the label is formatting metadata, not GT.
    correctness = eval_sample_positive_gt(doc, pred_point)
    
    bbox = doc["bbox"]
    # 归一化 bbox（现在在 resized 空间）
    bbox_norm = [
        bbox[0] / img_size[0],
        bbox[1] / img_size[1],
        bbox[2] / img_size[0],
        bbox[3] / img_size[1]
    ]
    
    # Each metric receives the same per-sample details for aggregation.
    details = {
        "is_correct": correctness == "correct",
        "correctness": correctness,
        "parse_error": pred_point is None,
        "answer": bbox_norm,
        "pred_point": pred_point,
        "raw_response": response,
        "platform": doc["platform"],
        "application": doc["application"],
        "ui_type": doc["ui_type"],
        "group": doc.get("group", "unknown"),
        "img_filename": doc.get("img_filename", ""),
        "instruction": doc.get("instruction", ""),
    }
    if gam_mode:
        details["pred_points"] = gam_pred_points or []
    return {"action_acc": details, "parse_error_rate": details}


def screenspot_pro_aggregate_parse_error_rate(
    results, args=None, lmms_eval_specific_kwargs=None
):
    """Fraction of scored GUI rows without a parseable point."""

    task_id = _task_id(lmms_eval_specific_kwargs)
    results = valid_results(results, context=f"{task_id}/parse_error_rate")
    if not results:
        return 0.0
    return sum(bool(result.get("parse_error")) for result in results) / len(results)


def screenspot_pro_aggregate_results(results, args=None, lmms_eval_specific_kwargs=None):
    """Aggregate results across all samples
    
    对齐原始代码 eval_screenspot_pro_parallel.py 第 163-182 行
    计算 action_acc, text_acc, icon_acc，并保存详细统计结果到 submissions 目录
    
    对齐 android_control 模式：
    当 YAML 中指定 metric: action_acc 时，lmms_eval 会提取 action_acc 字段（内层字典）
    所以 results 是: [{is_correct:..., correctness:..., platform:...}, ...]
    
    Args:
        results: List of result dictionaries from process_results (内层字典)
        args: Evaluation arguments (for file saving)
        lmms_eval_specific_kwargs: Additional evaluation kwargs
        
    Returns:
        float: Overall action_acc (exact accuracy)
    """
    task_id = _task_id(lmms_eval_specific_kwargs)
    results = valid_results(results, context=f"{task_id}/action_acc")
    if not results:
        return 0.0
    
    # 检查数据格式
    if not isinstance(results[0], dict):
        eval_logger.error(f"Unexpected results format: {type(results[0])}")
        return 0.0
    
    # Results 是内层字典列表：[{is_correct:..., correctness:..., platform:...}, ...]
    num_total = len(results)
    num_correct_action = sum(1 for r in results if r.get("correctness") == "correct")
    wrong_format_num = sum(1 for r in results if r.get("correctness") == "wrong_format")
    
    # Calculate text and icon accuracy
    text_results = [r for r in results if r.get("ui_type") == "text"]
    icon_results = [r for r in results if r.get("ui_type") == "icon"]
    
    text_correct = sum(1 for r in text_results if r.get("correctness") == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for r in icon_results if r.get("correctness") == "correct")
    icon_total = len(icon_results)
    
    # Calculate metrics
    action_acc = num_correct_action / num_total if num_total > 0 else 0.0
    text_acc = text_correct / text_total if text_total > 0 else 0.0
    icon_acc = icon_correct / icon_total if icon_total > 0 else 0.0
    
    # Build detailed statistics dictionary
    result_dict = {
        "overall": {
            "num_correct_action": num_correct_action,
            "num_total": num_total,
            "wrong_format_num": wrong_format_num,
            "action_acc": action_acc,
            "text_acc": text_acc,
            "icon_acc": icon_acc
        },
        "by_platform": {},
        "by_application": {}
    }
    
    # Per-platform statistics
    platforms = sorted(set(r.get("platform", "unknown") for r in results))
    for platform in platforms:
        platform_results = [r for r in results if r.get("platform") == platform]
        platform_correct = sum(1 for r in platform_results if r.get("correctness") == "correct")
        platform_total = len(platform_results)
        platform_wrong_format = sum(1 for r in platform_results if r.get("correctness") == "wrong_format")
        platform_acc = platform_correct / platform_total if platform_total > 0 else 0.0
        
        # Platform text/icon stats
        platform_text_results = [r for r in platform_results if r.get("ui_type") == "text"]
        platform_icon_results = [r for r in platform_results if r.get("ui_type") == "icon"]
        platform_text_correct = sum(1 for r in platform_text_results if r.get("correctness") == "correct")
        platform_icon_correct = sum(1 for r in platform_icon_results if r.get("correctness") == "correct")
        platform_text_acc = platform_text_correct / len(platform_text_results) if platform_text_results else 0.0
        platform_icon_acc = platform_icon_correct / len(platform_icon_results) if platform_icon_results else 0.0
        
        result_dict["by_platform"][platform] = {
            "num_correct_action": platform_correct,
            "num_total": platform_total,
            "wrong_format_num": platform_wrong_format,
            "action_acc": platform_acc,
            "text_acc": platform_text_acc,
            "icon_acc": platform_icon_acc
        }
    
    # Per-application statistics
    applications = sorted(set(r.get("application", "unknown") for r in results))
    for application in applications:
        app_results = [r for r in results if r.get("application") == application]
        app_correct = sum(1 for r in app_results if r.get("correctness") == "correct")
        app_total = len(app_results)
        app_wrong_format = sum(1 for r in app_results if r.get("correctness") == "wrong_format")
        app_acc = app_correct / app_total if app_total > 0 else 0.0
        
        # Application text/icon stats
        app_text_results = [r for r in app_results if r.get("ui_type") == "text"]
        app_icon_results = [r for r in app_results if r.get("ui_type") == "icon"]
        app_text_correct = sum(1 for r in app_text_results if r.get("correctness") == "correct")
        app_icon_correct = sum(1 for r in app_icon_results if r.get("correctness") == "correct")
        app_text_acc = app_text_correct / len(app_text_results) if app_text_results else 0.0
        app_icon_acc = app_icon_correct / len(app_icon_results) if app_icon_results else 0.0
        
        result_dict["by_application"][f"app:{application}"] = {
            "num_correct_action": app_correct,
            "num_total": app_total,
            "wrong_format_num": app_wrong_format,
            "action_acc": app_acc,
            "text_acc": app_text_acc,
            "icon_acc": app_icon_acc
        }
    
    # Print to console
    eval_logger.info("=" * 60)
    eval_logger.info("ScreenSpot-Pro Overall Results:")
    eval_logger.info("=" * 60)
    eval_logger.info(f"  num_correct_action: {num_correct_action}")
    eval_logger.info(f"  num_total: {num_total}")
    eval_logger.info(f"  wrong_format_num: {wrong_format_num}")
    eval_logger.info(f"  action_acc: {action_acc:.4f} ({action_acc*100:.2f}%)")
    eval_logger.info(f"  text_acc: {text_acc:.4f} ({text_acc*100:.2f}%)")
    eval_logger.info(f"  icon_acc: {icon_acc:.4f} ({icon_acc*100:.2f}%)")
    eval_logger.info("=" * 60)
    
    # Per-platform statistics
    eval_logger.info("\nPer-Platform Results:")
    for platform in platforms:
        platform_stats = result_dict["by_platform"][platform]
        eval_logger.info(f"  {platform}: {platform_stats['action_acc']:.4f} "
                       f"({platform_stats['num_correct_action']}/{platform_stats['num_total']}) "
                       f"[text: {platform_stats['text_acc']:.4f}, icon: {platform_stats['icon_acc']:.4f}]")
    
    # Per-application statistics
    eval_logger.info("\nPer-Application Results:")
    for app_key in sorted(result_dict["by_application"].keys()):
        app_stats = result_dict["by_application"][app_key]
        eval_logger.info(f"  {app_key}: {app_stats['action_acc']:.4f} "
                       f"({app_stats['num_correct_action']}/{app_stats['num_total']})")
    
    eval_logger.info("=" * 60)
    
    # Save detailed results to submissions directory
    if args is not None:
        try:
            from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
            
            # Generate filename with model name
            model_name = os.environ.get("SCREENSPOT_MODEL_NAME", "")
            task_slug = _task_id(lmms_eval_specific_kwargs).removeprefix("gam_")
            if model_name:
                # Clean model name for filename (remove special chars)
                clean_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                filename = f"{task_slug}_results_{clean_model_name}.json"
            else:
                filename = f"{task_slug}_results.json"
            
            output_file = generate_submission_file(filename, args)
            
            with open(output_file, "w") as f:
                json.dump(result_dict, f, indent=2)
            
            eval_logger.info(f"\nDetailed results saved to: {output_file}")
        except Exception as e:
            eval_logger.warning(f"Failed to save detailed results: {e}")
    
    # Return action_acc as the main metric
    return action_acc
