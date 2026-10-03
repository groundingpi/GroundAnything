#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parallel task evaluation with explicit dataset and output configuration.

Use scripts/evaluate.py or run.py eval to select a recipe. Workers share
the configured service and write logs beneath outputs/eval."""

import json
import importlib
import os
import sys
import time
import subprocess
import signal
from eval.process_control import cancellation_exit, finish_cleanup, defer_cancellation
import multiprocessing
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_config import (MODELS_TASK_CONFIG,
                         DEFAULT_CPU_NUM,
                         DEFAULT_TASK_PIXEL_MIN_MAX,
                         DEFAULT_GENERATION_KWARGS,
                         DEFAULT_TASK_CONFIG)

_SHARED_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
if _SHARED_UTILS_DIR not in sys.path:
    sys.path.insert(0, _SHARED_UTILS_DIR)
from prompt_mode import VALID_EVAL_MODES, normalize_eval_mode
from coordinate_mode import coordinate_mode_for_model, vlm_coord_mode

# Task logs live outside the source tree.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAM_EVAL_ROOT = _THIS_DIR
EVAL_LOG_ROOT = os.environ.get("GAM_EVAL_LOG_ROOT", os.path.join(os.path.dirname(GAM_EVAL_ROOT), "outputs", "eval"))

@dataclass
class TaskProcess:
    """任务进程信息"""
    task_name: str
    process: subprocess.Popen
    port: int
    cpu_count: int
    log_file: str
    log_fd: Optional[object] = None  # 日志文件句柄
    start_time: float = 0.0


class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self, tasks: List[TaskProcess], check_interval: float = 60.0):
        self.tasks = tasks
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """启动监控线程"""
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            for task in self.tasks:
                if task.process.poll() is None:  # 进程仍在运行
                    elapsed = time.time() - task.start_time
                    print(f"[监控] {task.task_name}: 运行中 (已运行 {elapsed:.0f}s, PID: {task.process.pid})")
                else:  # 进程已结束
                    elapsed = time.time() - task.start_time
                    status = "完成" if task.process.returncode == 0 else f"失败(退出码: {task.process.returncode})"
                    print(f"[监控] {task.task_name}: {status} (运行时长: {elapsed:.0f}s)")
            self._stop_event.wait(self.check_interval)
    
    def print_status(self):
        """打印当前状态"""
        running = sum(1 for t in self.tasks if t.process.poll() is None)
        completed = sum(1 for t in self.tasks if t.process.poll() is not None and t.process.returncode == 0)
        failed = sum(1 for t in self.tasks if t.process.poll() is not None and t.process.returncode != 0)
        total = len(self.tasks)
        
        print(f"\n[状态] 总任务: {total}, 运行中: {running}, 已完成: {completed}, 失败: {failed}")


def print_stage(title: str):
    """打印阶段标题"""
    bar = "=" * 12
    print(f"\n{bar} {title} {bar}")


def print_step(msg: str):
    """打印步骤信息"""
    print(f"[步骤] {msg}")


def print_ok(msg: str):
    """打印成功信息"""
    print(f"[成功] {msg}")


def print_warn(msg: str):
    """打印警告信息"""
    print(f"[警告] {msg}")


def print_err(msg: str):
    """打印错误信息"""
    print(f"[错误] {msg}")


def apply_smoke_generation_overrides(gen_kwargs: Dict) -> Dict:
    """Apply a bounded max-token override only for explicitly limited sample counts."""

    result = dict(gen_kwargs)
    raw = os.environ.get("GAM_EVAL_SMOKE_MAX_TOKENS")
    if raw is None:
        return result
    if not os.environ.get("GAM_EVAL_LIMIT"):
        raise ValueError(
            "GAM_EVAL_SMOKE_MAX_TOKENS 只能与 GAM_EVAL_LIMIT 一起使用，正式评测禁止截断"
        )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GAM_EVAL_SMOKE_MAX_TOKENS 必须是整数") from exc
    if not 1 <= value <= 4096:
        raise ValueError("GAM_EVAL_SMOKE_MAX_TOKENS 必须位于 1..4096")
    result["max_tokens"] = value
    return result


def apply_mode_generation_contract(gen_kwargs: Dict, mode: str) -> Dict:
    """Keep GAM native tokens verbatim while preserving historical VLM decoding."""

    result = dict(gen_kwargs)
    normalized_mode = normalize_eval_mode(mode)
    if normalized_mode in {"VLM", "GROUNDINGDINO"}:
        # VLM baselines retain their historical deterministic decoding. GAM's
        # category-specific two-profile contract must never leak into a VLM run.
        result.update(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
    else:
        # Native spatial coordinates and wrappers are AddedToken(special=True).
        # vLLM's OpenAI protocol defaults to dropping them and to inserting spaces
        # between adjacent special tokens, either of which makes canonical parsing
        # impossible.  This is a hard mode contract, not a caller override.
        result["skip_special_tokens"] = False
        result["spaces_between_special_tokens"] = False
    if normalized_mode == "REXOMNI":
        result.update(
            temperature=0.0,
            top_p=0.8,
            top_k=1,
            repetition_penalty=1.05,
        )
    elif normalized_mode == "LOCATEANYTHING":
        result.update(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        generation_mode = os.environ.get("GAM_LOCATEANYTHING_GENERATION_MODE")
        if generation_mode is not None:
            if generation_mode not in {"fast", "slow", "hybrid"}:
                raise ValueError(
                    "GAM_LOCATEANYTHING_GENERATION_MODE must be fast/slow/hybrid"
                )
            result["generation_mode"] = generation_mode
        max_tokens_override = os.environ.get("GAM_LOCATEANYTHING_MAX_TOKENS")
        if max_tokens_override is not None:
            try:
                max_tokens = int(max_tokens_override)
            except ValueError as exc:
                raise ValueError(
                    "GAM_LOCATEANYTHING_MAX_TOKENS must be an integer"
                ) from exc
            if not 1 <= max_tokens <= 8192:
                raise ValueError(
                    "GAM_LOCATEANYTHING_MAX_TOKENS must be within 1..8192"
                )
            result["max_tokens"] = max_tokens
    temperature_override = os.environ.get("GAM_EVAL_TEMPERATURE_OVERRIDE")
    if temperature_override is not None:
        try:
            temperature = float(temperature_override)
        except ValueError as exc:
            raise ValueError(
                "GAM_EVAL_TEMPERATURE_OVERRIDE must be a number"
            ) from exc
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "GAM_EVAL_TEMPERATURE_OVERRIDE must be within 0..2"
            )
        result["temperature"] = temperature
    top_p_override = os.environ.get("GAM_EVAL_TOP_P_OVERRIDE")
    if top_p_override is not None:
        try:
            top_p = float(top_p_override)
        except ValueError as exc:
            raise ValueError("GAM_EVAL_TOP_P_OVERRIDE must be a number") from exc
        if not 0.0 < top_p <= 1.0:
            raise ValueError("GAM_EVAL_TOP_P_OVERRIDE must be within (0, 1]")
        result["top_p"] = top_p
    top_k_override = os.environ.get("GAM_EVAL_TOP_K_OVERRIDE")
    if top_k_override is not None:
        try:
            top_k = int(top_k_override)
        except ValueError as exc:
            raise ValueError("GAM_EVAL_TOP_K_OVERRIDE must be an integer") from exc
        if top_k < 0:
            raise ValueError("GAM_EVAL_TOP_K_OVERRIDE must be non-negative")
        result["top_k"] = top_k
    repetition_penalty_override = os.environ.get(
        "GAM_EVAL_REPETITION_PENALTY_OVERRIDE"
    )
    if repetition_penalty_override is not None:
        try:
            repetition_penalty = float(repetition_penalty_override)
        except ValueError as exc:
            raise ValueError(
                "GAM_EVAL_REPETITION_PENALTY_OVERRIDE must be a number"
            ) from exc
        if not 0.0 < repetition_penalty <= 2.0:
            raise ValueError(
                "GAM_EVAL_REPETITION_PENALTY_OVERRIDE must be within (0, 2]"
            )
        result["repetition_penalty"] = repetition_penalty
    seed_override = os.environ.get("GAM_EVAL_SEED_OVERRIDE")
    if seed_override is not None:
        try:
            seed = int(seed_override)
        except ValueError as exc:
            raise ValueError("GAM_EVAL_SEED_OVERRIDE must be an integer") from exc
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("GAM_EVAL_SEED_OVERRIDE must be within uint32 range")
        result["seed"] = seed
    max_tokens_override = os.environ.get("GAM_EVAL_MAX_TOKENS_OVERRIDE")
    if max_tokens_override is not None:
        try:
            max_tokens = int(max_tokens_override)
        except ValueError as exc:
            raise ValueError(
                "GAM_EVAL_MAX_TOKENS_OVERRIDE must be an integer"
            ) from exc
        if not 1 <= max_tokens <= 65536:
            raise ValueError(
                "GAM_EVAL_MAX_TOKENS_OVERRIDE must be within 1..65536"
            )
        result["max_tokens"] = max_tokens
    until_override = os.environ.get("GAM_EVAL_UNTIL_OVERRIDE_JSON")
    if until_override is not None:
        try:
            until_values = json.loads(until_override)
        except json.JSONDecodeError as exc:
            raise ValueError("GAM_EVAL_UNTIL_OVERRIDE_JSON must be JSON") from exc
        if not (
            isinstance(until_values, list)
            and until_values
            and all(isinstance(value, str) and value for value in until_values)
        ):
            raise ValueError(
                "GAM_EVAL_UNTIL_OVERRIDE_JSON must be a non-empty string list"
            )
        result["until"] = until_values
    return result


_QWEN2_OCR_TASKS = frozenset({
    "gam_hiertext",
    "gam_icdar2015",
    "gam_totaltext",
    "gam_sroie",
    "gam_hiertext_Boxonly",
    "gam_icdar2015_Boxonly",
    "gam_totaltext_Boxonly",
    "gam_sroie_Boxonly",
})


def apply_model_family_generation_contract(
    gen_kwargs: Dict, task: str, env: Dict[str, str]
) -> Dict:
    """Apply family-specific VLM contracts after the generic mode contract."""

    result = dict(gen_kwargs)
    if (
        env.get("GAM_EVAL_MODE") == "VLM"
        and env.get("GAM_EVAL_PRESERVE_SPECIAL_TOKENS") == "1"
    ):
        # Opt-in family contract for models such as DeepSeek-VL2 whose native
        # grounding protocol uses AddedToken(special=True) delimiters.
        result["skip_special_tokens"] = False
        result["spaces_between_special_tokens"] = False
    if (
        env.get("GAM_EVAL_MODE") == "VLM"
        and env.get("GAM_COORD_MODE") == "qwen2"
        and task in _QWEN2_OCR_TASKS
    ):
        result["until"] = []
        result["repetition_penalty"] = 1.10
        result["max_tokens"] = max(8192, int(result.get("max_tokens", 0)))
        result["skip_special_tokens"] = False
        result["spaces_between_special_tokens"] = False
    mimo_gui_min_tokens = env.get("GAM_EVAL_MIMO_GUI_MIN_TOKENS")
    if (
        env.get("GAM_EVAL_MODE") == "VLM"
        and mimo_gui_min_tokens is not None
        and task in {"gam_screenspot_pro", "gam_screenspot_v2", "gam_osworld_g"}
    ):
        # MiMo may emit an empty <think> block and immediately select EOS before
        # its computer_use call.  A small model-family-only minimum generation
        # floor masks that premature EOS without changing any benchmark prompt.
        try:
            minimum = int(mimo_gui_min_tokens)
        except ValueError as exc:
            raise ValueError("GAM_EVAL_MIMO_GUI_MIN_TOKENS must be an integer") from exc
        if not 1 <= minimum <= 512:
            raise ValueError("GAM_EVAL_MIMO_GUI_MIN_TOKENS must be within 1..512")
        result["min_tokens"] = minimum
    return result


def run_kill_eval_script() -> int:
    """
    执行 kill_eval.sh 清理脚本，终止所有相关进程
    
    Returns:
        脚本的退出码，0表示成功，非0表示失败
    """
    # 获取脚本路径（eval_runner.py 在 GAM eval 根目录，kill_eval.sh 在 scripts/ 子目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kill_script = os.path.join(script_dir, "scripts", "kill_eval.sh")
    
    if not os.path.exists(kill_script):
        print_warn(f"清理脚本不存在: {kill_script}")
        return 1
    
    if not os.access(kill_script, os.X_OK):
        print_warn(f"清理脚本无执行权限: {kill_script}")
        return 1
    
    print_step(f"执行清理脚本: {kill_script}")
    
    try:
        # 执行脚本，直接输出到标准输出
        process = subprocess.run(
            ["bash", kill_script],
            check=False
        )
        
        return_code = process.returncode
        
        if return_code == 0:
            print_ok("清理脚本执行成功")
        else:
            print_warn(f"清理脚本返回非零退出码: {return_code}")
        
        return return_code
        
    except Exception as e:
        print_err(f"执行清理脚本失败: {e}")
        return 1


def setup_environment(
    model_path: str, api_url: str = None, mode: str = "VLM"
) -> Dict[str, str]:
    """设置环境变量"""
    env = os.environ.copy()
    env["GAM_EVAL_MODE"] = normalize_eval_mode(mode)
    if env["GAM_EVAL_MODE"] in {"VLM", "GROUNDINGDINO"}:
        if "GAM_COORD_MODE" not in env:
            env["GAM_COORD_MODE"] = (
                "qwen3"
                if env["GAM_EVAL_MODE"] == "GROUNDINGDINO"
                else coordinate_mode_for_model(model_path)
            )
        # Validate explicit overrides before starting costly model servers.
        previous = os.environ.get("GAM_COORD_MODE")
        try:
            os.environ["GAM_COORD_MODE"] = env["GAM_COORD_MODE"]
            vlm_coord_mode()
        finally:
            if previous is None:
                os.environ.pop("GAM_COORD_MODE", None)
            else:
                os.environ["GAM_COORD_MODE"] = previous
        if env["GAM_EVAL_MODE"] == "GROUNDINGDINO":
            # The local adapter emits the existing Qwen3-compatible 0..1000
            # JSON/tool-call contract.  This selects only the established GUI
            # parser; prompts and metrics remain unchanged.
            env["SCREENSPOT_MODEL_NAME"] = "Qwen3-VL"
        elif env["GAM_COORD_MODE"] == "qwen2":
            env.setdefault("GAM_EVAL_QWEN25_OCR_STRUCTURED", "1")
            env.setdefault("SCREENSPOT_MODEL_NAME", "Qwen2.5-VL")
        else:
            env.setdefault("SCREENSPOT_MODEL_NAME", "Qwen3-VL")
    
    # API 配置。Credential 只能由调用环境注入；代码中不提供真实 key/token，
    # 也不覆盖已经注入的值。Judge key 并非当前 grounding 指标的必需项，缺失时
    # 保持 unset 是最安全的默认行为。
    env.setdefault('API_TYPE', 'openai')
    env.setdefault('MODEL_VERSION', 'gpt-5-mini')
    
    # 本地API配置
    env['OPENAI_API_BASE'] = (
        api_url
        if api_url
        else env.get('OPENAI_API_BASE', 'http://127.0.0.1:8888/v1')
    )
    env.setdefault('OPENAI_API_KEY', 'EMPTY')
    env['OPENAI_API_VERSION'] = model_path
    
    # HuggingFace配置
    env.setdefault('HF_HOME', 'outputs/cache/huggingface')
    
    return env


PROMPT_OPTIM_TASK_MAP = {
    "RefSpatialBench_Location_qwen_align": "RefSpatialBench_Location_prompt_optim",
    "RefSpatialBench_Placement_qwen_align": "RefSpatialBench_Placement_prompt_optim",
    "RefSpatialBench_Unseen_qwen_align": "RefSpatialBench_Unseen_prompt_optim",
}


def parse_tasks(tasks_str: str, prompt_optim: bool = False) -> List[str]:
    """解析任务列表，如果启用 prompt_optim 则自动映射到优化版任务名"""
    tasks = [t.strip() for t in tasks_str.split(',') if t.strip()]
    if prompt_optim:
        mapped = []
        for t in tasks:
            if t in PROMPT_OPTIM_TASK_MAP:
                new_t = PROMPT_OPTIM_TASK_MAP[t]
                print_step(f"prompt_optim: {t} -> {new_t}")
                mapped.append(new_t)
            else:
                mapped.append(t)
        tasks = mapped
    return tasks


def calculate_cpu_allocation(tasks: List[str],model_eval_config, available_cpu: int) -> Dict[str, int]:
    """计算每个任务的CPU分配数量"""
    # Optional per-run weights let heterogeneous shards give more in-flight
    # capacity to their genuine long-tail tasks without changing the global
    # task contract used by every other evaluator.
    weight_overrides = {}
    override_json = os.environ.get("GAM_EVAL_CPU_WEIGHT_OVERRIDES", "").strip()
    if override_json:
        try:
            parsed = json.loads(override_json)
            if not isinstance(parsed, dict):
                raise ValueError("override must be a JSON object")
            weight_overrides = {
                str(task): int(weight)
                for task, weight in parsed.items()
                if int(weight) > 0
            }
            print_step(f"CPU weight overrides: {weight_overrides}")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid GAM_EVAL_CPU_WEIGHT_OVERRIDES={override_json!r}: {exc}"
            ) from exc

    # 计算总权重
    total_weight = 0
    task_weights = {}
    
    for task in tasks:
        weight = weight_overrides.get(
            task, model_eval_config.get(task, {}).get('cpu_num', 1)
        )
        task_weights[task] = weight
        total_weight += weight
        if weight == 1 and task not in model_eval_config:
            print_warn(f"任务 '{task}' 未找到CPU权重配置，使用默认值1")
    
    # 分配CPU
    cpu_allocation = {}
    for task in tasks:
        weight = task_weights[task]
        cpu_count = max(1, int(available_cpu * weight / total_weight))
        cpu_allocation[task] = cpu_count
    
    return cpu_allocation


def start_eval_process(
    model_type: str,
    task: str,
    port: int,
    cpu_count: int,
    model_path: str,
    log_dir: str,
    log_dir_timestamp: str,
    cache_dir: str,
    env: Dict[str, str],
    model_eval_config,
    include_path: Optional[str] = None,
) -> Tuple[subprocess.Popen, str]:
    """启动单个评测进程"""
    api_concurrency_override = env.get("GAM_EVAL_API_CONCURRENCY_PER_TASK", "").strip()
    if api_concurrency_override:
        cpu_count = int(api_concurrency_override)
        if cpu_count < 1:
            raise ValueError("GAM_EVAL_API_CONCURRENCY_PER_TASK must be positive")
    # 创建日志文件路径（任务日志保存在job_id目录下）
    log_file = os.path.join(log_dir_timestamp, f"{task}.log")
    
    # 任务缓存文件
    # VLM keeps the historical cache filename. GAM uses an isolated file so a
    # reused explicit job_id can never replay responses generated for old
    # prompts (or vice versa).
    cache_mode = env.get("GAM_EVAL_MODE", "VLM").lower()
    decode_mode = env.get("GAM_LOCATEANYTHING_GENERATION_MODE", "")
    mode_cache_suffix = "" if cache_mode == "vlm" else f"_{cache_mode}"
    if env.get('GAM_EVAL_DECODER'):
        mode_cache_suffix += '_' + env['GAM_EVAL_DECODER']
    if cache_mode == "locateanything" and decode_mode:
        mode_cache_suffix += f"_{decode_mode}"
    task_cache = os.path.join(
        cache_dir, f"sqlite_cache_{task}{mode_cache_suffix}.db"
    )
    
    # 获取任务像素数量
    task_pixel=model_eval_config.get(task, {}).get("task_pixel_min_max",None)
    if task_pixel is None:
        print_warn(f"任务 '{task}' 未找到像素范围配置，使用默认值")
        task_pixel=DEFAULT_TASK_PIXEL_MIN_MAX
    min_pixels=task_pixel[0]
    max_pixels=task_pixel[1]
    print(f"{task}[像素范围] {min_pixels} ~ {max_pixels}")
    #gen kwargs 超参
    gen_kwargs=model_eval_config.get(task, {}).get("generation_kwargs",None)
    if gen_kwargs is None:
        print_warn(f"任务 '{task}' 未找到生成参数配置，使用默认值")
        gen_kwargs=DEFAULT_GENERATION_KWARGS

    # 从 generation_kwargs 中提取 enable_thinking（不传给 gen_kwargs 字符串，通过 model_args 传递）
    if env.get('GAM_EVAL_DECODER'):
        if env.get('GAM_EVAL_MODE') not in ('GAM', 'DLM', 'RLV2'):
            raise ValueError('DLM decoder policy requires GAM/DLM/RLV2 evaluation')
        # The DLM request policy is an optional project-local provider. Pi has
        # no DLM package and never selects this branch.
        policy = importlib.import_module('infer.decoding')
        gen_kwargs = policy.evaluation_generation(task, gen_kwargs, env['GAM_EVAL_DECODER'])
    gen_kwargs = apply_smoke_generation_overrides(gen_kwargs)
    gen_kwargs = apply_mode_generation_contract(
        gen_kwargs, env.get("GAM_EVAL_MODE", "VLM")
    )
    gen_kwargs = apply_model_family_generation_contract(gen_kwargs, task, env)
    enable_thinking = gen_kwargs.pop("enable_thinking", True)

    gen_kwargs_str=""
    for k,v in gen_kwargs.items():
        gen_kwargs_str+=f"{k}={v},"
    print(f"{task}[生成参数] {gen_kwargs_str} enable_thinking={enable_thinking}")
    
    #判断是否是qwenvl3 特殊处理
    is_qwen3_vl=False
    if model_type in ("qwen3vl", "qwen35"):
        is_qwen3_vl=True
    
    # 构建命令
    # Bind accelerate to the exact Python runtime selected by the caller.
    # H800 vLLM uses an isolated venv; a PATH-resolved system ``accelerate``
    # would silently start a different torch/transformers ABI.
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes=1",
        f"--main_process_port={port}",
        "-m", "lmms_eval",
        "--model", "async_openai",
        "--gen_kwargs",f"{gen_kwargs_str}",
        "--model_args", f"model_type={model_type},model_version={env.get('GAM_EVAL_MODEL_ID', model_path)},min_pixels={min_pixels},max_pixels={max_pixels},base_url={env['OPENAI_API_BASE']},num_cpus={cpu_count},timeout=6000,is_qwen3_vl={is_qwen3_vl},enable_thinking={enable_thinking}",
        "--tasks", task,
        "--output_path", log_dir,
        "--log_samples",
        "--write_out",
        "--seed", "42",
        "--verbosity=DEBUG",
        "--use_cache", task_cache,
        "--batch_size", "1",
    ]

    # --include_path：指向 eval 任务定义目录，让 lmms_eval 递归发现 gam_* 任务
    # （必须传，否则该 task 名字在引擎自带 tasks/ 里找不到会直接报错 “task not found”）
    if include_path:
        cmd += ["--include_path", include_path]

    # 可选样本数量上限：设 GAM_EVAL_LIMIT=N 时只跑前 N 条
    _limit = os.environ.get("GAM_EVAL_LIMIT")
    if _limit:
        cmd += ["--limit", str(_limit)]

    # 启动进程
    # 注意：使用文件对象时，subprocess不会自动关闭，需要在进程结束时手动关闭
    log_fd = open(log_file, 'w')
    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True  # Own the entire worker group without preexec_fn
        )
    except BaseException:
        log_fd.close()
        raise
    
    return process, log_file, log_fd


def cleanup_task_processes(tasks, grace_seconds=2.0):
    """Reap only worker groups created by this invocation, including descendants."""
    groups = [task.process.pid for task in tasks]
    def signal_group(pid, sig):
        try:
            os.killpg(pid, sig)
            return True
        except ProcessLookupError:
            return False
    try:
        active = [pid for pid in groups if signal_group(pid, signal.SIGTERM)]
        deadline = time.monotonic() + grace_seconds
        while active and time.monotonic() < deadline:
            for task in tasks: task.process.poll()  # Reap exited worker parents.
            active = [pid for pid in active if signal_group(pid, 0)]
            if active: time.sleep(0.05)
        for pid in active: signal_group(pid, signal.SIGKILL)
        for task in tasks:
            task.process.wait(timeout=5)
    finally:
        for task in tasks:
            if task.log_fd is not None and not task.log_fd.closed:
                task.log_fd.close()


@cancellation_exit
def run_parallel_eval(
    model_type: str,
    model_path: str,
    tasks: str,
    job_id: Optional[str] = None,
    base_port: int = 12345,
    cpu_usage_ratio: float = 0.8,
    start_delay: float = 2.0,
    api_url: str = None,
    skip_cleanup: bool = False,
    prompt_optim: bool = False,
    include_path: Optional[str] = None,
    mode: str = "VLM",
) -> int:
    """
    并行运行多个评测任务
    
    Args:
        model_path: 模型路径
        tasks: 逗号分隔的任务列表，如 "gam_coco,gam_lvis"
        job_id: 任务ID，用于日志目录命名。如果为None，使用时间戳
        base_port: 基础端口号（每个任务使用不同的端口）
        cpu_usage_ratio: CPU使用比例（默认80%）
        start_delay: 启动任务之间的延迟（秒）
        include_path: 传给 lmms_eval --include_path 的任务定义目录（如 eval/Grounding），
            不传则退化为只能跑引擎自带 tasks/ 下的任务（本仓库 gam_* 任务基本都需要这个参数）
        mode: VLM/GAM/DLM/RLV2/REXOMNI/LOCATEANYTHING/GROUNDINGDINO；
            DLM/RLV2 与 GAM 共用提示词和输出协议，但 cache 按 mode 独立。
    
    Returns:
        退出码：0表示所有任务成功，非0表示有任务失败
    """
    print_stage("初始化并行评测")
    
    # 参数校验
    if not model_path:
        print_err("未提供模型路径")
        return 1
    
    if not tasks:
        print_err("未提供任务列表")
        return 1

    try:
        mode = normalize_eval_mode(mode)
    except ValueError as exc:
        print_err(str(exc))
        return 1
    
    # 解析任务列表
    task_list = parse_tasks(tasks, prompt_optim=prompt_optim)
    if not task_list:
        print_err("任务列表为空")
        return 1
    
    print_step(f"模型路径: {model_path}")
    print_step(f"任务列表: {', '.join(task_list)}")
    print_step(f"任务数量: {len(task_list)}")
    print_step(f"评测模式: {mode}")
    
    # 生成job_id
    if job_id is None:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_step(f"任务ID: {job_id}")
    
    # 设置环境变量
    env = setup_environment(model_path, api_url=api_url, mode=mode)
    
    # 创建日志和缓存目录（锚定在 outputs/eval/ 下，与调用者 cwd 无关）
    # cache 按 job_id 隔离，保证并发评测（不同模型/不同 job）互不污染缓存响应
    log_dir = os.path.join(EVAL_LOG_ROOT, "log_eval") + "/"
    cache_dir = os.path.join(EVAL_LOG_ROOT, "cache", job_id)
    log_dir_timestamp = os.path.join(log_dir, job_id)
    
    os.makedirs(log_dir_timestamp, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 清理缓存
    gpt_response_dir = os.path.join(EVAL_LOG_ROOT, "log", "gpt_response")
    if not skip_cleanup and os.path.exists(gpt_response_dir):
        import shutil
        try:
            shutil.rmtree(gpt_response_dir)
            print_step(f"已清理缓存目录: {gpt_response_dir}")
        except Exception as e:
            print_warn(f"清理缓存目录失败: {e}")
    
    # 计算CPU分配
    print_stage(" 计算CPU分配")
    
    model_eval_config= MODELS_TASK_CONFIG.get(model_type,None)
    
    if model_eval_config==None:
        print_warn(f"未找到模型配置使用默认配置!")
        model_eval_config=DEFAULT_TASK_CONFIG
    
    
    cpu_count = multiprocessing.cpu_count()
    available_cpu = int(cpu_count * cpu_usage_ratio)
    print_step(f"CPU总数: {cpu_count}")
    print_step(f"可用CPU数量（{int(cpu_usage_ratio*100)}%）: {available_cpu}")
    
    cpu_allocation = calculate_cpu_allocation(task_list,model_eval_config, available_cpu)
    print_step(f"CPU分配: {cpu_allocation}")
    
    # A single ownership scope covers startup, stagger delays and waiting.
    print_stage("启动评测任务")
    task_processes: List[TaskProcess] = []
    monitor = None
    failed_tasks = []
    success_tasks = []
    try:
        for i, task in enumerate(task_list):
            port = base_port + i
            with defer_cancellation():
                process, log_file, log_fd = start_eval_process(
                    model_type=model_type, task=task, port=port,
                    cpu_count=cpu_allocation[task], model_path=model_path,
                    log_dir=log_dir, log_dir_timestamp=log_dir_timestamp,
                    cache_dir=cache_dir, env=env, model_eval_config=model_eval_config,
                    include_path=include_path)
                task_processes.append(TaskProcess(
                    task_name=task, process=process, port=port,
                    cpu_count=cpu_allocation[task], log_file=log_file,
                    log_fd=log_fd, start_time=time.time()))
            print_ok(f"任务 {task} 已启动 (PID: {process.pid}, 日志: {log_file})")
            if i < len(task_list) - 1: time.sleep(start_delay)
        monitor = ProcessMonitor(task_processes)
        monitor.start_monitoring()
        for task_proc in task_processes:
            code = task_proc.process.wait()
            if code == 0:
                success_tasks.append(task_proc.task_name)
            else:
                failed_tasks.append(task_proc.task_name)
                print_err(f"任务失败: {task_proc.task_name}, 退出码: {code}, 日志: {task_proc.log_file}")
    except KeyboardInterrupt:
        print_warn("收到中断信号，回收本次评测的进程组")
        return 130
    except Exception as exc:
        print_err(f"评测启动或等待失败: {exc}")
        return 1
    finally:
        with finish_cleanup():
            try:
                if monitor is not None: monitor.stop_monitoring()
            finally:
                cleanup_task_processes(task_processes)
    print_stage("评测完成")
    print_step(f"成功任务数: {len(success_tasks)}/{len(task_processes)}")
    print_step(f"失败任务数: {len(failed_tasks)}/{len(task_processes)}")
    return 1 if failed_tasks else 0


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="并行评测脚本（Python版本）")
    parser.add_argument("--model_type", type=str,default="qwen3vl", help="模型类型")
    parser.add_argument("--model_path", help="模型路径")
    parser.add_argument("--tasks", type=str, default="ocrbench", help="逗号分隔的任务列表")
    parser.add_argument("--job_id", type=str, default=None, help="任务ID，用于日志目录命名")
    parser.add_argument("--base_port", type=int, default=12345, help="基础端口号")
    parser.add_argument("--cpu_ratio", type=float, default=0.8, help="CPU使用比例（0-1）")
    parser.add_argument("--start_delay", type=float, default=2.0, help="启动任务之间的延迟（秒）")
    parser.add_argument(
        "--api_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL；外部凭据应由安全代理持有，此处只传本机代理地址",
    )
    parser.add_argument("--prompt_optim", action="store_true", help="使用与训练数据对齐的优化 prompt（含 system message，格式对齐）")
    parser.add_argument("--include_path", type=str, default=None, help="lmms_eval --include_path：本仓库任务定义目录（如 eval/Grounding）")
    parser.add_argument(
        "--skip_cleanup",
        action="store_true",
        help="跳过任务后的进程清理；用于共享主机上的外部 API 评测",
    )
    parser.add_argument(
        "--mode",
        type=str.upper,
        choices=VALID_EVAL_MODES,
        default="VLM",
        help="VLM/GAM/DLM/RLV2/REXOMNI/LOCATEANYTHING/GROUNDINGDINO 评测协议",
    )
    
    args = parser.parse_args()
    
    exit_code = run_parallel_eval(
        model_type=args.model_type,
        model_path=args.model_path,
        tasks=args.tasks,
        job_id=args.job_id,
        base_port=args.base_port,
        cpu_usage_ratio=args.cpu_ratio,
        start_delay=args.start_delay,
        api_url=args.api_url,
        prompt_optim=args.prompt_optim,
        include_path=args.include_path,
        mode=args.mode,
        skip_cleanup=args.skip_cleanup,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
