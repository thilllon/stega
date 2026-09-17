"""Inner code: one code frame = header + fountain symbol + CRC32, split over Reed-Solomon codewords,
byte-interleaved across the whole frame, whitened with a fixed scrambler, mapped to cell symbols.

  frame bytes  : [hdr 18B][symbol S B][crc32 4B]              (= D data bytes)
  RS           : D bytes cut into n_blocks shortened RS(n_i, n_i - nsym) codewords, n_i <= 255
  interleave   : codeword byte j of every block, round robin -> spatial bursts hit all blocks evenly
  scramble     : XOR with a fixed pseudo-random mask -> ~50% cell balance whatever the content
  cells        : bits (MSB first) grouped per `bits_per_cell` -> palette index per data cell
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import reedsolo

from .layout import Layout, build_layout, prng_bytes

FRAME_MAGIC = b"SG"
FRAME_VERSION = 1
HEADER = struct.Struct(">2sBBIIHHH")  # magic, version, profile id, session, total_len, gen_size, gen, esi
CRC_LEN = 4


@dataclass(frozen=True)
class FrameHeader:
    pid: int
    session: int
    total_len: int
    gen_size: int
    gen: int
    esi: int


@dataclass(frozen=True)
class FrameCodec:
    layout: Layout
    block_lens: tuple[int, ...]  # codeword length of each RS block
    nsym: int
    scramble: np.ndarray  # uint8 mask, capacity_bytes
    interleave: np.ndarray  # interleave[pos] = flat index into concatenated codewords
    filler_bits: np.ndarray  # bits for the cells beyond capacity_bytes*8

    @property
    def data_bytes(self) -> int:
        return sum(n - self.nsym for n in self.block_lens)

    @property
    def symbol_size(self) -> int:
        return self.data_bytes - HEADER.size - CRC_LEN

    # ---- block bookkeeping -------------------------------------------------------------
    def block_slices(self) -> list[slice]:
        """Slices of each block inside the concatenated-codeword space."""
        out, off = [], 0
        for n in self.block_lens:
            out.append(slice(off, off + n))
            off += n
        return out

    def data_slices(self) -> list[slice]:
        out, off = [], 0
        for n in self.block_lens:
            k = n - self.nsym
            out.append(slice(off, off + k))
            off += k
        return out

    # ---- encode ------------------------------------------------------------------------
    def encode(self, hdr: FrameHeader, symbol: bytes) -> np.ndarray:
        """-> palette index per data cell (uint8, len n_data_cells)."""
        assert len(symbol) == self.symbol_size, (len(symbol), self.symbol_size)
        body = (
            HEADER.pack(FRAME_MAGIC, FRAME_VERSION, hdr.pid, hdr.session, hdr.total_len, hdr.gen_size, hdr.gen, hdr.esi)
            + symbol
        )
        data = body + struct.pack(">I", zlib.crc32(body))
        rs = _rs(self.nsym)
        codewords = b"".join(bytes(rs.encode(data[s])) for s in self.data_slices())
        cw = np.frombuffer(codewords, np.uint8)
        stream = cw[self.interleave] ^ self.scramble
        bits = np.concatenate([np.unpackbits(stream), self.filler_bits])
        return bits_to_symbols(bits, self.layout.profile.bits_per_cell)

    # ---- decode ------------------------------------------------------------------------
    def deinterleave(self, stream: np.ndarray) -> np.ndarray:
        cw = np.empty_like(stream)
        cw[self.interleave] = stream
        return cw

    def decode_block(self, cw: np.ndarray, conf: np.ndarray, i: int) -> bytes | None:
        """RS-decode block i from concatenated codewords; retries with low-confidence bytes as erasures."""
        s = self.block_slices()[i]
        block = bytearray(cw[s].tobytes())
        rs = _rs(self.nsym)
        k = len(block) - self.nsym
        try:
            return bytes(rs.decode(block, only_erasures=False)[0])[:k]
        except (reedsolo.ReedSolomonError, ZeroDivisionError, IndexError, ValueError):
            pass
        order = np.argsort(conf[s], kind="stable")
        for n_erase in (self.nsym // 2, (self.nsym * 3) // 4, self.nsym - 2):
            try:
                pos = sorted(int(x) for x in order[:n_erase])
                return bytes(rs.decode(bytearray(block), erase_pos=pos)[0])[:k]
            except (reedsolo.ReedSolomonError, ZeroDivisionError, IndexError, ValueError):
                continue
        return None

    def decode(self, stream: np.ndarray, byte_conf: np.ndarray, skip=None) -> tuple[FrameHeader, bytes] | None:
        """stream: bytes as sampled from the cells (still scrambled + interleaved, capacity_bytes long);
        byte_conf: per-byte reliability in the same order. Returns (header, symbol) or None.
        `skip(header) -> bool` lets the caller abort after block 0 for already-known symbols."""
        cw = self.deinterleave(stream ^ self.scramble)
        conf = np.empty_like(byte_conf)
        conf[self.interleave] = byte_conf
        first = self.decode_block(cw, conf, 0)
        if first is None or len(first) < HEADER.size:
            return None
        magic, ver, pid, session, total_len, gen_size, gen, esi = HEADER.unpack_from(first)
        if magic != FRAME_MAGIC or ver != FRAME_VERSION or pid != self.layout.profile.pid:
            return None
        hdr = FrameHeader(pid, session, total_len, gen_size, gen, esi)
        if skip is not None and skip(hdr):
            return hdr, b""
        parts = [first]
        for i in range(1, len(self.block_lens)):
            b = self.decode_block(cw, conf, i)
            if b is None:
                return None
            parts.append(b)
        data = b"".join(parts)
        body, crc = data[:-CRC_LEN], data[-CRC_LEN:]
        if struct.unpack(">I", crc)[0] != zlib.crc32(body):
            return None
        return hdr, body[HEADER.size :]


@lru_cache(maxsize=None)
def _rs(nsym: int) -> reedsolo.RSCodec:
    return reedsolo.RSCodec(nsym, nsize=255)


def bits_to_symbols(bits: np.ndarray, bpc: int) -> np.ndarray:
    if bpc == 1:
        return bits.astype(np.uint8)
    b = bits.reshape(-1, bpc).astype(np.uint8)
    w = (1 << np.arange(bpc - 1, -1, -1)).astype(np.uint8)
    return (b * w).sum(axis=1).astype(np.uint8)


def symbols_to_bits(sym: np.ndarray, bpc: int) -> np.ndarray:
    if bpc == 1:
        return sym.astype(np.uint8)
    shifts = np.arange(bpc - 1, -1, -1)
    return ((sym[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


@lru_cache(maxsize=None)
def build_codec(profile_name: str) -> FrameCodec:
    layout = build_layout(profile_name)
    p = layout.profile
    cap = layout.capacity_bytes
    n_blocks = math.ceil(cap / 255)
    base, extra = divmod(cap, n_blocks)
    lens = tuple(base + (1 if i < extra else 0) for i in range(n_blocks))
    if min(lens) - p.ecc_nsym < 32:
        raise ValueError(f"profile {p.name}: RS blocks too short for nsym={p.ecc_nsym}")
    # round-robin interleave over codeword byte index
    offs = np.cumsum((0,) + lens[:-1])
    order = [offs[i] + j for j in range(max(lens)) for i in range(n_blocks) if j < lens[i]]
    interleave = np.array(order, dtype=np.int64)
    scramble = np.frombuffer(prng_bytes(f"stega-scramble-{p.name}", cap), np.uint8)
    n_filler = layout.n_data_cells * p.bits_per_cell - cap * 8
    filler = np.unpackbits(np.frombuffer(prng_bytes(f"stega-filler-{p.name}", n_filler // 8 + 1), np.uint8))[:n_filler]
    return FrameCodec(layout, lens, p.ecc_nsym, scramble, interleave, filler)
