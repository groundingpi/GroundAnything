"""Shared validation for native HTTP generation contracts (no model imports)."""

def validate_thinking(body, non_thinking):
    if 'chat_template_kwargs' not in body:
        return
    value = body['chat_template_kwargs']
    if not isinstance(value, dict) or set(value) != {'enable_thinking'} or type(value['enable_thinking']) is not bool:
        raise ValueError('chat_template_kwargs supports only boolean enable_thinking')
    if value['enable_thinking'] != (not non_thinking):
        raise ValueError('enable_thinking conflicts with the served model configuration')


def validate_budget(requested, cap, prompt_tokens=None, context=None):
    if type(requested) is not int or not 1 <= requested <= cap:
        raise ValueError(f'max_tokens must be an integer in 1..{cap}')
    if prompt_tokens is not None and prompt_tokens + requested > context:
        raise ValueError('prompt plus max_tokens exceeds max_model_len')
    return requested
