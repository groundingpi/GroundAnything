"""Inspect or extract local dependency archives. Does not install packages or access the network."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from config_contract import relative

ROOT = Path(__file__).resolve().parents[1]


def unpack(entry, root=ROOT):
    destination = relative(root, entry['destination'])
    archive = relative(root, entry['archive'])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != entry['archive_sha256']:
        raise ValueError('dependency archive checksum mismatch')
    if destination.exists():
        sys.path.insert(0, str(root))
        from models.dependency_contract import verify_dependency
        verify_dependency(entry['name'], root)
        return 'already_verified'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.dependency-', dir=destination.parent))
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if {m.name for m in members} != set(entry['files']) or len(members) != len(entry['files']):
                raise ValueError('archive file list differs from manifest')
            for member in members:
                if not member.isfile(): raise ValueError('dependency archive must contain regular files only')
                path = relative(temporary, member.name)
                payload = tar.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != entry['files'][member.name]:
                    raise ValueError(f'dependency file checksum mismatch: {member.name}')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return 'prepared'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='append', help='dependency name; may be repeated, defaults to all')
    parser.add_argument('--apply', action='store_true', help='extract verified snapshots inside this project')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'third_party/manifest.json').read_text())['dependencies']
    names = args.name or sorted(entries)
    if set(names) - entries.keys(): parser.error('unknown dependency name')
    reports = []
    for name in names:
        entry = entries[name]
        archive = relative(ROOT, entry['archive'])
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry['archive_sha256']:
            raise ValueError(f'archive checksum mismatch: {name}')
        reports.append({'name': name, 'files': len(entry['files']), 'destination': entry['destination'],
                        'status': unpack(entry) if args.apply else 'verified_archive'})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__': main()
