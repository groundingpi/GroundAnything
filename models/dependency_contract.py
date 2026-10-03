"""Verify the bundled dependency source snapshots against their manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_dependency(name, root=None):
    root = Path(root or ROOT).resolve()
    entry = json.loads((root / 'third_party/manifest.json').read_text())['dependencies'][name]
    directory = (root / entry['destination']).resolve()
    if not directory.is_relative_to(root): raise ValueError('dependency directory escapes project')
    if not directory.is_dir(): raise ValueError(f'prepare dependency snapshot first: {name}')
    for relative, digest in entry['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'missing dependency source: {name}/{relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'dependency source drift: {name}/{relative}')
    extras = [str(p.relative_to(directory)) for p in directory.rglob('*.py')
              if '__pycache__' not in p.parts and str(p.relative_to(directory)) not in entry['files']]
    if extras: raise ValueError(f'unrecorded Python source in {name}: {extras[:3]}')
    return entry['source_revision']
