"""Check the actual served model and native budgets before launching workers."""
import json
import importlib
import os
from urllib.request import Request, urlopen


def get_json(url):
    headers = {}
    key = os.environ.get('OPENAI_API_KEY')
    if key: headers['Authorization'] = 'Bearer ' + key
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        return json.load(response)


def validate_service(config, model_ids, health):
    model_id = config.get('model_id', config['model_path'])
    if model_id not in model_ids:
        raise ValueError(f'model_id {model_id!r} is not listed by /v1/models')
    record = {'model_id': model_id}
    if config.get('service_contract', 'openai') != 'native':
        return record
    if health.get('service_contract') != 'native-v1':
        raise ValueError('native service must advertise its generation contract in /health')
    from eval.task_config import MODELS_TASK_CONFIG, DEFAULT_GENERATION_KWARGS
    table = MODELS_TASK_CONFIG[config.get('model_type', 'qwen3vl')]
    maxima = {task: config.get('max_tokens', table.get(task, {}).get('generation_kwargs', DEFAULT_GENERATION_KWARGS)['max_tokens']) for task in config['tasks']}
    cap, context = health.get('max_new_tokens'), health.get('max_model_len')
    if type(cap) is not int or type(context) is not int:
        raise ValueError('native service must expose integer output/context limits')
    if any(n > cap or n >= context for n in maxima.values()):
        raise ValueError('task output budget exceeds service output/context capacity; update matching YAMLs')
    if config['mode'] in ('GAM', 'DLM', 'RLV2') and health.get('non_thinking') is not True:
        raise ValueError('spatial evaluation requires a non-thinking native service')
    record.update(task_max_tokens=maxima, max_new_tokens=cap, max_model_len=context,
                  non_thinking=health.get('non_thinking'),
                  input_budget_check='expanded prompt plus requested output checked by server per request')
    return record


def preflight(config, prepared):
    base = config['api_url'].rstrip('/')
    models = get_json(base + '/models')
    health = {}
    if config.get('service_contract', 'openai') == 'native':
        if not base.endswith('/v1'):
            raise ValueError('native API URL must end in /v1')
        health = get_json(base[:-3] + '/health')
    record = validate_service(config, [item['id'] for item in models['data']], health)
    if config.get('decoder') is not None:
        # Optional Anything policy; generic OpenAI/native services need no DLM module.
        policy = importlib.import_module('infer.decoding')
        info = get_json(base[:-3] + '/server_info')
        record.update(policy.validate_server_decoder(config['decoder'], info))
    record['api_url'] = base
    if prepared['effective_runtime']['api_urls'] != [base]:
        raise ValueError('preflight and worker endpoints differ')
    return record
