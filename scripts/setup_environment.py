"""Plan or install one environment in a fresh, explicit virtual environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from config_contract import relative

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--venv', required=True, help='new project-relative virtual environment directory')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--inherit-container', action='store_true', help='inherit packages from the documented GPU base image')
    parser.add_argument('--index-url', default='https://pypi.org/simple')
    args = parser.parse_args()
    profiles = yaml.safe_load((ROOT / 'environments/install.yaml').read_text())['profiles']
    if args.profile not in profiles: parser.error('unknown profile')
    profile = profiles[args.profile]
    if profile.get('platform') == 'Linux x86_64':
        import platform
        if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
            parser.error(f'{args.profile} profile requires Linux x86_64 and Python 3.12 for its wheels')
    venv = relative(ROOT, args.venv)
    if venv.exists(): parser.error('use a new environment directory; existing environments are never modified')
    if profile.get('requires_container') and not args.inherit_container:
        parser.error('this profile requires the documented base image and --inherit-container')
    python = str(venv / 'bin/python')
    constraints = str(ROOT / profile['requirements'])
    commands = [[sys.executable, '-m', 'venv', *(['--system-site-packages'] if args.inherit_container else []), str(venv)],
                [python, '-m', 'pip', 'install', '-c', constraints, 'setuptools>=68', 'wheel', 'packaging', 'ninja', 'psutil', 'einops']]
    if profile.get('accelerator_packages'):
        commands.append([python, '-m', 'pip', 'install', '-c', constraints, '--index-url', profile.get('accelerator_index', args.index_url), *profile['accelerator_packages']])
    names = profile.get('snapshots', [])
    entries = json.loads((ROOT / 'third_party/manifest.json').read_text())['dependencies']
    if names:
        commands.append([sys.executable, str(ROOT / 'scripts/prepare_dependencies.py'), '--apply',
                         *[arg for name in names for arg in ('--name', name)]])
    if profile.get('source_extensions'):
        commands.append([python, '-m', 'pip', 'install', '--no-build-isolation', '--no-cache-dir', '-c', constraints, *profile['source_extensions']])
    commands.append([python, '-m', 'pip', 'install', '-c', constraints, '-r', str(ROOT / profile['requirements']),
                     *[arg for name in names for arg in ('-e', str(ROOT / entries[name]['destination']))]])
    if profile.get('post_install_overrides'):
        commands.append([python, '-m', 'pip', 'install', '--no-deps', *profile['post_install_overrides']])
    check_command = ([python, str(ROOT / 'scripts/check_serve_environment.py')]
                     if args.profile == 'serve' else [python, '-m', 'pip', 'check'])
    commands += [[python, '-m', 'pip', 'install', '--no-deps', '-e', str(ROOT)], check_command]
    print(json.dumps({'profile': args.profile, 'base_image': profile.get('base_image'),
                      'commands': commands, 'build_env': profile.get('build_env', {}), 'index_url': args.index_url, 'apply': args.apply, 'gpu_verified': False}, indent=2))
    if not args.apply: return
    if profile.get('requires_container'):
        import importlib.metadata
        version = importlib.metadata.version('torch')
        if not version.startswith('2.8.0a0+5228986c39'):
            raise ValueError('training requires the documented Torch source build; no environment was created')
    # Container overlays intentionally replace selected packages inside the venv.
    # Remove the base image's global pip constraint hook only for these subprocesses.
    env = os.environ.copy(); env.pop('PIP_CONSTRAINT', None)
    env['PIP_INDEX_URL'] = args.index_url
    env['PIP_CONFIG_FILE'] = os.devnull
    env.update({str(k): str(v) for k, v in profile.get('build_env', {}).items()})
    env.pop('PIP_EXTRA_INDEX_URL', None)
    for command in commands: subprocess.run(command, cwd=ROOT, env=env, check=True)
    report = subprocess.run([python, '-m', 'pip', 'freeze', '--all'], cwd=ROOT, env=env,
                            check=True, text=True, capture_output=True)
    (venv / 'installed.freeze.txt').write_text(report.stdout)


if __name__ == '__main__': main()
