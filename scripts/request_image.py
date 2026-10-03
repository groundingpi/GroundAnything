"""Send one local image to an already running native service (standard library only)."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen


def make_body(image, model, prompt, max_tokens=256):
    path = Path(image)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {'image/png', 'image/jpeg', 'image/webp'}:
        raise ValueError('image must have a PNG, JPEG or WebP extension')
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return {'model': model, 'max_tokens': max_tokens, 'temperature': 0,
            'skip_special_tokens': False,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}},
                {'type': 'text', 'text': prompt},
            ]}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--base-url', required=True, help='service URL ending in /v1')
    parser.add_argument('--model', required=True, help='model ID from service configuration')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    body = make_body(args.image, args.model, args.prompt, args.max_tokens)
    request = Request(args.base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(body).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    choice = result['choices'][0]
    print(json.dumps({'model': result.get('model'), 'content': choice['message']['content'],
                      'finish_reason': choice['finish_reason'], 'usage': result.get('usage')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
