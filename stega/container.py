"""Payload container: file name, original size, SHA-256 and optional zlib compression."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass

MAGIC = b"STGC"
VERSION = 1
FLAG_ZLIB = 0x01
_HDR = struct.Struct(">4sBBHQ32s")  # magic, version, flags, name_len, original_size, sha256
HEADER_SIZE = _HDR.size
MAX_SIZE = 1 << 28  # 256 MiB: far beyond what this channel carries; bounds a crafted zlib bomb


@dataclass
class Payload:
    name: str
    data: bytes


def pack(name: str, data: bytes, compress: bool = True) -> bytes:
    flags, body = 0, data
    if compress:
        z = zlib.compress(data, 9)
        if len(z) < len(data) * 0.98:
            flags, body = FLAG_ZLIB, z
    name_b = name.encode("utf-8")[:1024]
    hdr = _HDR.pack(MAGIC, VERSION, flags, len(name_b), len(data), hashlib.sha256(data).digest())
    return hdr + name_b + body


def unpack(blob: bytes) -> Payload:
    if len(blob) < _HDR.size:
        raise ValueError("not a stega container (too short)")
    magic, version, flags, name_len, size, digest = _HDR.unpack_from(blob)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a stega container (bad magic/version)")
    if size > MAX_SIZE:
        raise ValueError(f"declared size {size} is implausible")
    off = _HDR.size + name_len
    name = blob[_HDR.size : off].decode("utf-8", errors="replace")
    body = blob[off:]
    if flags & FLAG_ZLIB:
        d = zlib.decompressobj()
        try:
            data = d.decompress(body, size + 1)  # bounded: never inflate past the declared size
        except zlib.error as e:
            raise ValueError(f"corrupt compressed payload: {e}") from None
    else:
        data = body[:size]
    if len(data) != size or hashlib.sha256(data).digest() != digest:
        raise ValueError("payload integrity check failed (SHA-256 mismatch)")
    return Payload(name=name, data=data)
