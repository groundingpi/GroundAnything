"""Short source-tree commands for setup, training, serving and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parent

def parser_for(root):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    setup = commands.add_parser('setup', help='install a task environment')
    setup.add_argument('profile', choices=('train', 'serve', 'eval'))
    train = commands.add_parser('train', help='start training')
    recipe = train.add_mutually_exclusive_group()
    recipe.add_argument('--config', help='project-relative training launch YAML')
    recipe.add_argument('--resume', action='store_true', help='resume from the configured training checkpoint')
    serve = commands.add_parser('serve', help='start the model service')
    engine = serve.add_mutually_exclusive_group()
    engine.add_argument('--config', help='project-relative service launch YAML')
    engine.add_argument('--decoder', choices=('denoise', 'causal', 'speculative'), help='SGLang decoder (default: denoise / DecodeV4)')
    evaluate = commands.add_parser('eval', help='evaluate an existing model service')
    evaluation = evaluate.add_mutually_exclusive_group()
    evaluation.add_argument('--config', help='project-relative evaluation YAML')
    evaluation.add_argument('--decoder', choices=('denoise', 'causal', 'speculative'), help='request policy matching the running service (default: denoise)')
    actions = [setup, train, serve, evaluate]
    bundle = commands.add_parser('prepare-model', help='prepare the DLM model bundle for SGLang')
    bundle.add_argument('--base', default='weights/base_model')
    bundle.add_argument('--source', default='weights/dlm')
    bundle.add_argument('--output', default='weights/dlm_bundle', help='new project-relative model directory')
    actions.append(bundle)
    for action in actions:
        action.add_argument('--venv', help='project-relative environment directory (default: .venv-<profile>)')
        action.add_argument('--dry-run', action='store_true', help='print the downstream command without executing it')
    return parser

def relative(root, value):
    path = Path(value)
    target = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or (not target.is_relative_to(root.resolve())) or (target == root.resolve()):
        raise ValueError('use a path inside the project: ' + value)
    return path.as_posix()

def plan(args, root=ROOT):
    if not (root / 'grounding_anything').is_dir() or not (root / 'configs/release').is_dir():
        raise ValueError('run.py must be in a complete GroundAnything source tree')
    profile = args.profile if args.action == 'setup' else {'train': 'train', 'serve': 'serve', 'eval': 'eval', 'prepare-model': 'serve'}[args.action]
    venv = relative(root, args.venv or '.venv-' + profile)
    python = str(root / venv / 'bin/python')
    if args.action == 'setup':
        python = sys.executable if sys.version_info[:2] == (3, 12) else shutil.which('python3.12')
        if not python:
            if not args.dry_run:
                raise ValueError('setup requires the interpreter described in environments/README.md (python3.12)')
            python = 'python3.12'
        command = [python, str(root / 'scripts/setup_environment.py'), '--profile', profile, '--venv', venv, '--apply']
    elif args.action == 'prepare-model':
        command = [python, '-m', 'infer.dlm.build_public_model_bundle', '--kind', 'dlm', '--label', 'groundinganything', '--base', relative(root, args.base), '--source', relative(root, args.source), '--destination', relative(root, args.output)]
    else:
        if args.config:
            name = None
        elif args.action == 'train':
            name = 'dlm_resume_h800_smoke' if args.resume else 'dlm_train_h800_smoke'
        elif args.action == 'serve':
            name = 'dlm_sglang_' + (args.decoder or 'denoise')
        else:
            name = 'dlm' if args.decoder in (None, 'denoise') else 'dlm_' + args.decoder
        config = relative(root, args.config or f"configs/{('eval' if args.action == 'eval' else 'release')}/{name}.yaml")
        if not (root / config).is_file() or Path(config).suffix not in ('.yaml', '.yml'):
            raise ValueError('configuration YAML not found: ' + config)
        script = 'evaluate.py' if args.action == 'eval' else 'run.py'
        command = [python, str(root / 'scripts' / script), config]
    if args.action != 'setup' and (not args.dry_run) and (not Path(python).is_file()):
        hint = f'python3 run.py setup {profile} --venv {venv}'
        raise ValueError('environment missing; run: ' + hint)
    return {'profile': profile, 'cwd': str(root), 'command': command}

def main(argv=None):
    parser = parser_for(ROOT)
    args = parser.parse_args(argv)
    try:
        selected = plan(args, ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return
    os.chdir(selected['cwd'])
    os.execv(selected['command'][0], selected['command'])
if __name__ == '__main__':
    main()
