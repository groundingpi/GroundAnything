"""Read OpenImages ``tar@offset=&size=&member=`` virtual image paths."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
from typing import Optional
from urllib.parse import unquote


TAR_URI_RE = re.compile(
    r"^(?P<tar>.+\.tar)@offset=(?P<offset>[0-9]+)&size=(?P<size>[0-9]+)"
    r"&member=(?P<member>.+)$"
)
MAX_MEMBER_BYTES = 128 << 20


class TarURIError(OSError):
    pass


def parse_tar_uri(value: str) -> Optional[tuple[Path, int, int, str]]:
    if not isinstance(value, str):
        return None
    match = TAR_URI_RE.fullmatch(value.strip())
    if match is None:
        return None
    tar_path = Path(match.group("tar")).expanduser()
    offset = int(match.group("offset"))
    size = int(match.group("size"))
    member = unquote(match.group("member"))
    if offset < 0 or not 0 < size <= MAX_MEMBER_BYTES:
        raise TarURIError(
            f"invalid tar virtual range: offset={offset} size={size} uri={value!r}"
        )
    if not member or member.startswith("/") or "\x00" in member:
        raise TarURIError(f"unsafe tar member metadata: {member!r}")
    return tar_path, offset, size, member


def read_tar_uri(value: str) -> bytes:
    parsed = parse_tar_uri(value)
    if parsed is None:
        raise TarURIError(f"not a GAM tar URI: {value!r}")
    tar_path, offset, size, member = parsed
    if not tar_path.is_file():
        raise TarURIError(f"tar file does not exist for {member!r}: {tar_path}")
    file_size = os.stat(tar_path).st_size
    if offset + size > file_size:
        raise TarURIError(
            f"tar virtual range exceeds file for {member!r}: "
            f"offset={offset} size={size} file_size={file_size}"
        )
    with tar_path.open("rb", buffering=0) as handle:
        handle.seek(offset)
        payload = handle.read(size)
    if len(payload) != size:
        raise TarURIError(
            f"short tar virtual read for {member!r}: expected={size} got={len(payload)}"
        )
    return payload


def install_swift_tar_uri_patch() -> None:
    """Patch Swift once; ordinary paths retain the original implementation."""

    from swift.template import vision_utils

    current = vision_utils.load_file
    if getattr(current, "_gam_tar_uri_patch", False):
        return

    original = current

    def gam_load_file(path):
        if isinstance(path, str) and parse_tar_uri(path) is not None:
            return BytesIO(read_tar_uri(path))
        return original(path)

    gam_load_file._gam_tar_uri_patch = True
    gam_load_file._gam_original = original
    vision_utils.load_file = gam_load_file

