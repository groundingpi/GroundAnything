"""Explicit data binding and preflight for the supported RL training routes."""
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]


def load_mixture(args, seed, kind, *, root=ROOT, factory=None):
    """Read and audit the chosen dataset before device/distributed initialization."""
    selected = ('grounding_data', 'ocr_data') if kind == 'grounding_ocr' else ('multiroute_data',)
    if kind not in {'grounding_ocr', 'multiroute'}:
        raise ValueError(f'unknown RL data kind: {kind}')
    paths = {}
    for key in ('grounding_data', 'ocr_data', 'multiroute_data'):
        value = getattr(args, key, None)
        if key not in selected:
            if value is not None:
                raise ValueError(f'{key} is not supported by the {kind} route')
            continue
        if value is None:
            raise ValueError(f'provide --{key.replace("_", "-")} for the {kind} route')
        path = Path(value)
        if path.is_absolute() or PureWindowsPath(str(value)).is_absolute() or '..' in path.parts or '\\' in str(value):
            raise ValueError(f'{key} must be project-relative')
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f'{key} escapes the project root')
        if not resolved.is_file():
            raise FileNotFoundError(f'{key}: missing JSONL file: {path}')
        paths[key] = resolved
    if kind == 'grounding_ocr':
        if factory is None:
            from train.rl.corrected_data import CorrectedGroundingOCRMixture
            factory = CorrectedGroundingOCRMixture
        return factory(seed, grounding=paths['grounding_data'], ocr=paths['ocr_data'])
    if factory is None:
        from train.rl.shared.causal_data import MultirouteRLV2Mixture
        factory = MultirouteRLV2Mixture
    return factory(seed, path=paths['multiroute_data'])
