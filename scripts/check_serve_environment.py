"""Check the SGLang ABI override without hiding unrelated metadata failures."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys


def classify(lines, versions):
    known, errors = [], []
    for line in lines:
        if not line.strip() or line == 'No broken requirements found.':
            continue
        if (versions.get('torch') == '2.9.1' and versions.get('nvidia-cudnn-cu12') == '9.16.0.29'
                and line.startswith('torch 2.9.1 has requirement nvidia-cudnn-cu12==9.10.2.21;')
                and line.endswith('but you have nvidia-cudnn-cu12 9.16.0.29.')):
            known.append(line)
        else:
            errors.append(line)
    return known, errors


def main():
    versions = {n: metadata.version(n) for n in ('torch', 'nvidia-cudnn-cu12')}
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    known, errors = classify(result.stdout.splitlines(), versions)
    report = dict(pip_check_exit=result.returncode, known_metadata_override=known,
                  unexpected_errors=errors, stderr=result.stderr, versions=versions,
                  reason='SGLang requires cuDNN >=9.15 for Torch 2.9.1 Conv3d; use the explicitly pinned 9.16 runtime.')
    Path(sys.prefix, 'serve-dependency-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if errors or result.stderr or (result.returncode and not known):
        raise SystemExit('Unexpected dependency conflict in the serving environment')
    import torch
    if torch.backends.cudnn.version() < 91500:
        raise SystemExit('The loaded cuDNN is older than 9.15')


if __name__ == '__main__':
    main()
