"""Admit only the verified bundled runtime before importing model backends."""
import importlib
import importlib.util
import json
from pathlib import Path
from models.dependency_contract import verify_dependency

ROOT = Path(__file__).resolve().parents[1]


def bundled_transformers(root=ROOT):
    guidance = ('Native DLM requires the bundled Transformers 5.7.0 source. '
                'Use the documented train environment; run '
                'python scripts/prepare_dependencies.py --name transformers --apply, then '
                'python -m pip install -e vendor/transformers '
                'with the Python used to start this server.')
    try:
        revision = verify_dependency('transformers', root)
        entry = json.loads((root / 'third_party/manifest.json').read_text())['dependencies']['transformers']
        expected = (root / entry['destination'] / 'src/transformers').resolve()
        spec = importlib.util.find_spec('transformers')
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(expected):
            raise ValueError('Transformers import resolves outside the bundled source')
        module = importlib.import_module('transformers')
        if not Path(module.__file__).resolve().is_relative_to(expected) or module.__version__ != '5.7.0':
            raise ValueError('unexpected Transformers import path or version')
    except (ValueError, OSError, KeyError, ImportError) as exc:
        raise ValueError(f'{guidance} Validation failed: {exc}') from exc
    return module, revision
