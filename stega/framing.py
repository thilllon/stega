"""Inner code. A code frame is split into `bands` horizontal bands (Profile.bands, default 1); each band
is an independent *unit* carrying one fountain symbol:

  unit bytes   : [hdr 18B][symbol S B][crc32 4B]              (= D data bytes)
  RS           : D bytes cut into n_blocks shortened RS(n_i, n_i - nsym) codewords, n_i <= 255
  interleave   : codeword byte j of every block, round robin -> spatial bursts hit all blocks evenly
  scramble     : XOR with a fixed pseudo-random mask -> ~50% cell balance whatever the content
  cells        : bits (MSB first) grouped per `bits_per_cell` -> palette index per data cell

Data cells are numbered row-major, so a band (a contiguous range of them) is a horizontal strip. A
capture torn by the rolling shutter (top rows from frame t, bottom rows from t+1) therefore still yields
every band that lies entirely above or below the tear, and a frame with ~100 codewords is no longer lost
because one of them fails. With bands=1 the bitstream is identical to the original single-unit format.
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
class UnitCodec:
    """Codes one fountain symbol into `n_cells` consecutive data cells starting at `cell_start`."""

    pid: int
    bits_per_cell: int
    cell_start: int
    n_cells: int
    block_lens: tuple[int, ...]  # codeword length of each RS block
    nsym: int
    scramble: np.ndarray  # uint8 mask, capacity_bytes
    interleave: np.ndarray  # interleave[pos] = flat index into concatenated codewords
    filler_bits: np.ndarray  # bits for the cells beyond capacity_bytes*8

    @property
    def capacity_bytes(self) -> int:
        return len(self.scramble)

    @property
    def cells(self) -> slice:
        return slice(self.cell_start, self.cell_start + self.n_cells)

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
        """-> palette index per cell of this unit (uint8, len n_cells)."""
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
        return bits_to_symbols(bits, self.bits_per_cell)

    # ---- decode ------------------------------------------------------------------------
    def stream_from_cells(self, sym: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell decisions/confidences of the whole frame -> (byte stream, per-byte confidence) of this unit."""
        bpc, cap = self.bits_per_cell, self.capacity_bytes
        s, m = sym[self.cells], margin[self.cells]
        bits = symbols_to_bits(s, bpc)[: cap * 8]
        conf = np.repeat(m, bpc)[: cap * 8].reshape(cap, 8).min(axis=1)
        return np.packbits(bits), conf

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
        if magic != FRAME_MAGIC or ver != FRAME_VERSION or pid != self.pid:
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


@dataclass(frozen=True)
class FrameCodec:
    """A whole code frame: `bands` units side by side (stacked top to bottom)."""

    layout: Layout
    units: tuple[UnitCodec, ...]

    @property
    def bands(self) -> int:
        return len(self.units)

    @property
    def symbol_size(self) -> int:
        """Bytes of fountain symbol per unit (identical for every band)."""
        return self.units[0].symbol_size

    @property
    def frame_payload(self) -> int:
        return self.symbol_size * self.bands

    def encode_frame(self, items: list[tuple[FrameHeader, bytes]]) -> np.ndarray:
        """One (header, symbol) per band -> palette index per data cell of the frame."""
        assert len(items) == self.bands, (len(items), self.bands)
        return np.concatenate([u.encode(h, s) for u, (h, s) in zip(self.units, items)])

    # Single-band conveniences (the original one-symbol-per-frame API).
    def _only_unit(self) -> UnitCodec:
        if self.bands != 1:
            raise ValueError(f"profile {self.layout.profile.name} has {self.bands} bands; use encode_frame/units")
        return self.units[0]

    def encode(self, hdr: FrameHeader, symbol: bytes) -> np.ndarray:
        return self._only_unit().encode(hdr, symbol)

    def decode(self, stream: np.ndarray, byte_conf: np.ndarray, skip=None):
        return self._only_unit().decode(stream, byte_conf, skip=skip)

    def __getattr__(self, name):  # block_lens, nsym, interleave, ... of a single-band codec
        if name in ("block_lens", "nsym", "interleave", "scramble", "block_slices", "data_slices", "capacity_bytes"):
            return getattr(self._only_unit(), name)
        raise AttributeError(name)


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
    n_bands = max(1, p.bands)
    per_band = layout.n_data_cells // n_bands  # equal capacity -> one fountain symbol size for all bands
    cap = per_band * p.bits_per_cell // 8
    n_blocks = math.ceil(cap / 255)
    base, extra = divmod(cap, n_blocks)
    lens = tuple(base + (1 if i < extra else 0) for i in range(n_blocks))
    if min(lens) - p.ecc_nsym < 32:
        raise ValueError(f"profile {p.name}: RS blocks too short for nsym={p.ecc_nsym} with {n_bands} bands")
    if cap - p.ecc_nsym * n_blocks - HEADER.size - CRC_LEN < 64:
        raise ValueError(f"profile {p.name}: {n_bands} bands leave no room for a payload")
    # round-robin interleave over codeword byte index (within a band)
    offs = np.cumsum((0,) + lens[:-1])
    order = [offs[i] + j for j in range(max(lens)) for i in range(n_blocks) if j < lens[i]]
    interleave = np.array(order, dtype=np.int64)
    units = []
    for b in range(n_bands):
        start = b * per_band
        n_cells = per_band if b < n_bands - 1 else layout.n_data_cells - start  # leftovers -> last band filler
        tag = "" if n_bands == 1 else f"-b{b}"  # bands=1 keeps the original labels -> identical bitstream
        scramble = np.frombuffer(prng_bytes(f"stega-scramble-{p.name}{tag}", cap), np.uint8)
        n_filler = n_cells * p.bits_per_cell - cap * 8
        filler = np.unpackbits(np.frombuffer(prng_bytes(f"stega-filler-{p.name}{tag}", n_filler // 8 + 1), np.uint8))[:n_filler]
        units.append(UnitCodec(p.pid, p.bits_per_cell, start, n_cells, lens, p.ecc_nsym, scramble, interleave, filler))
    return FrameCodec(layout, tuple(units))
