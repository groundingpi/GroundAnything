"""Rebuild a selected dependency export from a pinned public base and file patch."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe archive member path')
    return path.as_posix()


def read_export(archive, expected):
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members = stream.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('export file inventory mismatch')
        result = {}
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError('export must contain regular files only')
            data = stream.extractfile(member).read()
            if digest(data) != expected[member.name]:
                raise ValueError('export file checksum mismatch')
            result[member.name] = data
        return result


def rebuild(name, base_archive, output, root=ROOT):
    root = Path(root).resolve()
    entry = json.loads((root / 'third_party/manifest.json').read_text())['dependencies'][name]
    recipe_path = root / safe_name(entry['upstream_patch'])
    recipe = json.loads(recipe_path.read_text())
    if digest(recipe_path.read_bytes()) != entry['upstream_patch_sha256']:
        raise ValueError('patch recipe checksum mismatch')
    base_blob = Path(base_archive).read_bytes()
    if digest(base_blob) != recipe['base']['archive_sha256']:
        raise ValueError('public base archive checksum mismatch')
    source = root / safe_name(entry['archive'])
    blob = source.read_bytes()
    if digest(blob) != entry['archive_sha256']:
        raise ValueError('bundled export checksum mismatch')
    replacements = read_export(blob, entry['files'])
    base_files = {}
    with tarfile.open(fileobj=io.BytesIO(base_blob)) as stream:
        for member in stream:
            # Only selected regular source files are read; nothing is extracted.
            if member.isfile():
                parts = PurePosixPath(safe_name(member.name)).parts
                if len(parts) > 1:
                    base_files['/'.join(parts[1:])] = member
        required = {}
        for path in entry['files']:
            mapped = recipe.get('path_overrides', {}).get(path, recipe['source_prefix'] + path)
            safe_name(mapped)
            if mapped in base_files:
                required[path] = stream.extractfile(base_files[mapped]).read()
    if set(recipe['operations']) - set(entry['files']):
        raise ValueError('patch contains unselected files')
    result = {}
    for path, target_sha in entry['files'].items():
        operation = recipe['operations'].get(path)
        if operation is None:
            if path not in required:
                raise ValueError('unchanged upstream file is missing')
            data = required[path]
        else:
            if operation['op'] == 'replace':
                if path not in required or digest(required[path]) != operation['base_sha256']:
                    raise ValueError('replacement base checksum mismatch')
            elif operation['op'] == 'add':
                if path in required:
                    raise ValueError('added file already exists in selected base')
            else:
                raise ValueError('unknown patch operation')
            data = replacements[path]
        if digest(data) != target_sha:
            raise ValueError('reconstructed source checksum mismatch')
        result[path] = data
    output = Path(output)
    if output.resolve() == source.resolve():
        raise ValueError('output must differ from bundled source archive')
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as stream:
                for path in sorted(result, key=lambda name: PurePosixPath(name).parts):
                    member = tarfile.TarInfo(path)
                    member.mode = 0o644
                    member.size = len(result[path])
                    stream.addfile(member, io.BytesIO(result[path]))
    return {'dependency': name, 'files': len(result), 'base_commit': recipe['base']['commit'],
            'archive_sha256': digest(output.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--base-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.name, args.base_archive, args.output), indent=2))


if __name__ == '__main__':
    main()
