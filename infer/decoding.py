"""Public SGLang decoder selection and matching evaluation policies.

This module has no model/runtime dependencies so launch and request contracts can
be checked before allocating a device. Historical algorithm registration names
remain internal; ``denoise`` selects the canonical DecodeV4 task profiles.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {'denoise': 'GAMDecodeV2Block', 'causal': None,
              'speculative': 'GAMSpeculativeBlock'}
PROFILE_PATH = ROOT / 'infer/decode/configs/task_profiles.json'


def decoder_settings(decoder):
    if decoder not in ALGORITHMS:
        raise ValueError('decoder must be denoise, causal or speculative')
    settings = {'decoder': decoder, 'algorithm': ALGORITHMS[decoder],
                'algorithm_config': None, 'decode_profile': None}
    if decoder == 'denoise':
        profile = json.loads(PROFILE_PATH.read_text())
        from infer.decode.config import DecodeConfig
        config = dict(profile['shared_decode'], token_shift=1, temperature=0.0, top_p=1.0)
        DecodeConfig.from_mappings(config)
        settings.update(algorithm_config=config, decode_profile=profile['revision'],
                        task_profile_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest())
    elif decoder == 'speculative':
        settings['algorithm_config'] = {'block_size': 32, 'token_shift': 1, 'debug': False}
    return settings


def evaluation_generation(task, generation, decoder):
    """Apply the selected request policy without changing the shared task table."""
    decoder_settings(decoder)  # Reject unsupported names even outside the YAML frontend.
    result = copy.deepcopy(generation)
    if decoder == 'denoise':
        if os.environ.get('GAM_DLM_DECODE_PROFILE') != 'task_profiles':
            raise ValueError('denoise evaluation requires the canonical task_profiles setting')
        from infer.engines.dlm_task_profiles import build_dlm_task_config
        result = build_dlm_task_config({task: {'generation_kwargs': result}})[task]['generation_kwargs']
    else:
        # Strict Spec implements greedy longest-prefix verification. Its causal
        # control must also start with a greedy anchor and no repetition penalty.
        result = {k: v for k, v in result.items() if not k.startswith('gam_dlm_')}
        result.update(temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        result.pop('top_k', None)
    return result


def validate_server_decoder(decoder, info):
    """Check the actual bundled SGLang /server_info before evaluation workers."""
    expected = decoder_settings(decoder)['algorithm']
    if not isinstance(info, dict) or 'dllm_algorithm' not in info:
        raise ValueError('SGLang /server_info must expose dllm_algorithm')
    actual = info['dllm_algorithm']
    if actual != expected:
        raise ValueError(f'decoder mismatch: requested {decoder} ({expected}), server reports {actual}')
    return {'decoder': decoder, 'algorithm': actual,
            'cuda_graph_disabled': info.get('disable_cuda_graph'),
            'max_running_requests': info.get('max_running_requests')}
