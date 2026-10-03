"""Explicit, read-only dataset locations shared by task loaders and the launcher."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r"\$\{EVAL_DATA:([a-zA-Z0-9_]+)\}")


def input_path(base: Path, value: str) -> Path:
    """Dataset inputs may live outside the repository; never use this for outputs."""
    if not isinstance(value, str) or not value or '\x00' in value or '${' in value or '://' in value:
        raise ValueError('dataset paths must be explicit local filesystem paths')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_data_paths(root: Path, data_root: str, config_path: str = 'configs/datasets.yaml') -> dict:
    base = input_path(root, data_root)
    source = input_path(root, config_path)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or not config:
        raise ValueError('dataset configuration must be a nonempty mapping')
    resolved = {}
    for key, value in config.items():
        if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9_]+', key):
            raise ValueError('invalid dataset key')
        if isinstance(value, list):
            if key in {'images', 'screenspot_pro', 'screenspot_v2', 'osworld_g', 'refspatial', 'refcoco', 'refcocoplus', 'refcocog', 'icdar2015_dontcare'}:
                raise ValueError(f'{key} must name a single path')
            if not value: raise ValueError(f'empty dataset file list: {key}')
            resolved[key] = [str(input_path(base, item)) for item in value]
        else:
            resolved[key] = str(input_path(base, value))
    return resolved


def configured_paths() -> dict:
    prepared = os.getenv('EVAL_DATA_PATHS')
    if prepared:
        return json.loads(prepared)
    return load_data_paths(ROOT, os.getenv('GAM_EVAL_DATA_ROOT', 'data/eval'))


def dataset_path(key: str, *parts: str) -> Path:
    value = configured_paths()[key]
    if not isinstance(value, str):
        raise ValueError(f'{key} must name a single path')
    return Path(value).joinpath(*parts)


def expand_data_paths(value):
    """Expand only declared dataset tokens, preserving lists of annotation files."""
    if isinstance(value, str):
        match = TOKEN.fullmatch(value)
        if match: return configured_paths()[match.group(1)]
        def replace(match):
            path = configured_paths()[match.group(1)]
            if not isinstance(path, str): raise ValueError('a file list cannot be embedded in a path')
            return path
        return TOKEN.sub(replace, value)
    if isinstance(value, dict): return {key: expand_data_paths(item) for key, item in value.items()}
    if isinstance(value, list): return [expand_data_paths(item) for item in value]
    if isinstance(value, tuple): return tuple(expand_data_paths(item) for item in value)
    return value


def task_data_inputs(root: Path, tasks: list[str], inventory: dict, paths: dict) -> list[str]:
    """Collect selected tasks' inputs, including inherited YAML and image roots."""
    keys = set()
    visited = set()
    def read(path):
        path = path.resolve()
        if path in visited: return
        if not path.is_relative_to((root / 'eval').resolve()):
            raise ValueError('task include escapes evaluation directory')
        visited.add(path)
        text = path.read_text()
        keys.update(TOKEN.findall(text))
        node = yaml.compose(text)
        for key, value in node.value:
            if key.value == 'include':
                includes = [value.value] if isinstance(value, yaml.ScalarNode) else [v.value for v in value.value]
                for include in includes: read(path.parent / include)
        if 'eval/rexomni_jsonl.py' in text: keys.add('images')
    for task in tasks:
        read(root / inventory[task])
    if 'gam_osworld_g' in tasks: keys.add('osworld_g')
    if 'gam_icdar2015' in keys: keys.add('icdar2015_dontcare')
    missing_keys = keys - paths.keys()
    if missing_keys: raise ValueError(f'missing dataset keys: {sorted(missing_keys)}')
    return sorted({item for key in keys for item in (paths[key] if isinstance(paths[key], list) else [paths[key]])})


def missing_data_inputs(inputs: list[str]) -> list[str]:
    return [value for value in inputs if not glob.glob(value)]
