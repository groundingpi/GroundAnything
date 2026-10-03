"""CPU-only configuration checks. These checks are not a filesystem sandbox."""
from pathlib import Path, PureWindowsPath
import re

PATH_KEYS = {'path', 'manifest', 'external_plugin', 'source_manifest', 'source_recipe',
             'completed_checkpoint', 'native_config', 'initial_dlm_checkpoint',
             'resume_from_checkpoint', 'deepspeed_config', 'model', 'BASE_MODEL',
             'DLM_CHECKPOINT'}


def relative(root, value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('resource paths must be nonempty strings')
    path = Path(value)
    if (path.is_absolute() or PureWindowsPath(value).is_absolute()
            or '..' in path.parts or '\\' in value or value.startswith('~')
            or re.search(r'\$\{|\$\(', value) or '://' in value):
        raise ValueError(f'expected project-relative path: {value}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes project through a symlink: {value}')
    return resolved


def path_key(key):
    key = str(key)
    return key in PATH_KEYS or key.lower().endswith(('_path', '_dir', '_root', '_manifest', '_audit', '_file', '_checkpoint', '_gate'))


def validate_argv(root, argv):
    for index, item in enumerate(argv):
        if item.startswith('--'):
            option, separator, value = item.partition('=')
            key = option[2:].replace('-', '_')
            if key in {'output', 'config', 'checkpoint'} or path_key(key):
                if not separator:
                    if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
                        raise ValueError(f'{option} requires a relative path')
                    value = argv[index + 1]
                relative(root, value)


def validate_tree(root, obj, key='', trail='config'):
    """Inspect nested YAML without importing model code or following resource manifests."""
    if isinstance(obj, dict):
        for k, value in obj.items():
            validate_tree(root, value, k, f'{trail}.{k}')
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            validate_tree(root, value, key, f'{trail}[{i}]')
    elif isinstance(obj, str):
        # Free prose and registry names are not resource paths. Leading absolute
        # paths and traversal are nevertheless rejected regardless of the key.
        if (path_key(key) or obj.startswith(('/', '~')) or '..' in Path(obj).parts
                or PureWindowsPath(obj).is_absolute()):
            try:
                relative(root, obj)
            except ValueError as exc:
                raise ValueError(f'{trail}: {exc}') from exc


def native_inputs(native):
    """Resource roots read directly by the supported training entrypoints."""
    result = []
    model = native.get('model', {})
    for key in ('path', 'manifest'):
        if model.get(key): result.append(model[key])
    for item in native.get('datasets', []):
        for key in ('path', 'manifest'):
            if item.get(key): result.append(item[key])
    for key, value in native.get('data', {}).items():
        if path_key(key) and isinstance(value, str): result.append(value)
    for key, value in native.get('runtime', {}).items():
        if path_key(key) and isinstance(value, str): result.append(value)
    if native.get('training',{}).get('resume_from_checkpoint'):
        result.append(native['training']['resume_from_checkpoint'])
    return result


def validate_native(root, native, world_size):
    validate_tree(root, native)
    training = native.get('training', {})
    runtime = native.get('runtime', {})
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps'):
        if key in training and (type(training[key]) is not int or training[key] < 1):
            raise ValueError(f'training.{key} must be a positive integer')
    nodes, gpus = runtime.get('expected_nodes'), runtime.get('expected_gpus_per_node')
    if nodes is not None and gpus is not None and nodes * gpus != world_size:
        raise ValueError('native runtime topology differs from launcher topology')
    expected = training.get('expected_global_batch_size')
    if expected is not None:
        actual = world_size * training['per_device_train_batch_size'] * training['gradient_accumulation_steps']
        if expected != actual: raise ValueError('expected_global_batch_size differs from launch batch size')


def checkpoint_args(root, cfg):
    checkpoint = cfg.get('checkpoint', {})
    if not isinstance(checkpoint, dict) or set(checkpoint) - {'initial_dlm_checkpoint', 'resume_from_checkpoint'}:
        raise ValueError('checkpoint accepts initial_dlm_checkpoint or resume_from_checkpoint')
    if len(checkpoint) > 1:
        raise ValueError('initial_dlm_checkpoint and resume_from_checkpoint are mutually exclusive')
    if checkpoint and cfg['entrypoint'] not in {'dlm-train', 'vlm-train'}:
        raise ValueError('checkpoint is supported only for training')
    if cfg['entrypoint'] == 'vlm-train' and 'initial_dlm_checkpoint' in checkpoint:
        raise ValueError('initial_dlm_checkpoint is not a VLM checkpoint')
    result = []
    for key, value in checkpoint.items():
        relative(root, value)
        result.extend(['--' + (key if cfg['entrypoint']=='vlm-train' else key.replace('_', '-')), value])
    return result


def rl_data_paths(root, recipe, entry):
    """Require explicit dataset paths without importing training or accelerator code."""
    keys = ('grounding_path', 'ocr_path') if entry == 'rl-24' else ('multiroute_path',)
    data = recipe.get('data')
    if not isinstance(data, dict) or set(data) != set(keys):
        raise ValueError(f'{entry} data must contain exactly {keys}')
    for value in data.values():
        relative(root, value)
    return {key: data[key] for key in keys}


def rl_command(root, cfg, target, python):
    import yaml
    expected = {'rl-24': 24, 'rl-56': 56, 'rl-64': 64}[cfg['entrypoint']]
    native = relative(root, cfg.get('native_config'))
    recipe = yaml.safe_load(native.read_text())
    if not isinstance(recipe, dict) or not isinstance(recipe.get('causal_justgrpo'), dict):
        raise ValueError('RLV2 requires a causal_justgrpo native configuration')
    validate_tree(root, recipe)
    data = rl_data_paths(root, recipe, cfg['entrypoint'])
    data_args = [part for key, value in data.items()
                 for part in ('--' + key.replace('_path', '-data'), value)]
    args = cfg.get('args', [])
    if any(a.split('=')[0] in {'--grounding-data', '--ocr-data', '--multiroute-data'} for a in args):
        raise ValueError('RL data paths come only from native_config.data')
    validate_argv(root, args)
    if any(a == '--config' or a.startswith('--config=') for a in args):
        raise ValueError('RLV2 --config comes only from native_config')
    topology = cfg.get('distributed', {})
    if not isinstance(topology, dict) or set(topology) - {'nproc_per_node', 'nnodes', 'node_rank', 'master_addr', 'master_port'}:
        raise ValueError('invalid distributed configuration')
    count, nodes, rank = (topology.get(k) for k in ('nproc_per_node', 'nnodes', 'node_rank'))
    if (any(type(x) is not int for x in (count, nodes, rank)) or count < 1 or nodes < 1
            or count * nodes != expected or not 0 <= rank < nodes):
        raise ValueError(f'{cfg["entrypoint"]} requires world_size={expected} and a valid node rank')
    command = [python, '-m', 'torch.distributed.run', f'--nproc_per_node={count}']
    if nodes == 1: command.append('--standalone')
    else:
        address, port = topology.get('master_addr'), topology.get('master_port', 29500)
        if not isinstance(address, str) or not address or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('multi-node RLV2 requires master_addr and a valid master_port')
        command.extend([f'--nnodes={nodes}', f'--node_rank={rank}', f'--master_addr={address}', f'--master_port={port}'])
    return [*command, *target, '--config', cfg['native_config'], *data_args, *args]
