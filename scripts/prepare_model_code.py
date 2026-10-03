"""Attach the bundled VLM model Python files to a prepared local model directory."""
import argparse
import hashlib
import json
from pathlib import Path
from config_contract import relative

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='project-relative model directory')
    parser.add_argument('--apply', action='store_true', help='copy missing code after checking every existing file')
    ns = parser.parse_args()
    model = relative(ROOT, ns.model)
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'groundinganything_vlm' or config.get('text_config', {}).get('model_type') != 'qwen3':
        raise ValueError('model must be a supported VLM checkpoint with a Qwen3 text backbone')
    source = ROOT / 'models/vlm'
    manifest = json.loads((source / 'manifest.json').read_text())
    missing = []
    for name, digest in manifest['files'].items():
        if Path(name).name != name: raise ValueError('invalid runtime filename')
        payload = (source / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest: raise ValueError(f'runtime code drift: {name}')
        target = relative(ROOT, str(Path(ns.model) / name))
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f'existing model code differs; no files replaced: {name}')
        else:
            missing.append((target, payload))
    print(json.dumps({'model': ns.model, 'missing_code': [p.name for p, _ in missing], 'apply': ns.apply}, indent=2))
    if ns.apply:
        for path, payload in missing:
            # Exclusive creation protects a file that appeared after preflight.
            with path.open('xb') as stream: stream.write(payload)


if __name__ == '__main__': main()
