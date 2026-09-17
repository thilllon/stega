"""Outer erasure code: generation-based random linear fountain code over GF(2).

The container is cut into K source symbols of S bytes, grouped into generations of <= `gen_size`
symbols. Every transmitted symbol belongs to one generation and is identified by an ESI:
  * esi < k_g  -> systematic: the source symbol itself
  * esi >= k_g -> repair: XOR of a pseudo-random subset of the generation's source symbols
A generation is recovered from ANY k_g linearly independent symbols (on average ~k_g + 1.6),
so lost / blurred / rolling-shutter-torn frames are simply erasures. Generations keep Gaussian
elimination cost O(gen_size^2 * S) instead of O(K^2 * S).
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FountainParams:
    total_len: int  # container length in bytes
    symbol_size: int
    gen_size: int

    @property
    def k(self) -> int:
        return max(1, math.ceil(self.total_len / self.symbol_size))

    @property
    def n_gens(self) -> int:
        return math.ceil(self.k / self.gen_size)

    def gen_k(self, gen: int) -> int:
        return min(self.gen_size, self.k - gen * self.gen_size)


def coefficients(session: int, gen: int, esi: int, k_g: int) -> np.ndarray:
    """0/1 coefficient vector (uint8, length k_g) of symbol `esi` in generation `gen`."""
    if esi < k_g:
        v = np.zeros(k_g, np.uint8)
        v[esi] = 1
        return v
    seed = struct.pack(">III", session, gen, esi)
    raw = hashlib.shake_128(b"stega-fountain" + seed).digest((k_g + 7) // 8)
    v = np.unpackbits(np.frombuffer(raw, np.uint8))[:k_g].copy()
    if not v.any():
        v[esi % k_g] = 1
    return v


class FountainEncoder:
    def __init__(self, blob: bytes, symbol_size: int, gen_size: int, session: int):
        self.params = FountainParams(len(blob), symbol_size, gen_size)
        self.session = session
        padded = blob + bytes(self.params.k * symbol_size - len(blob))
        self.src = np.frombuffer(padded, np.uint8).reshape(self.params.k, symbol_size)

    def symbol(self, gen: int, esi: int) -> bytes:
        p = self.params
        start = gen * p.gen_size
        k_g = p.gen_k(gen)
        block = self.src[start : start + k_g]
        if esi < k_g:
            return block[esi].tobytes()
        mask = coefficients(self.session, gen, esi, k_g).astype(bool)
        return np.bitwise_xor.reduce(block[mask], axis=0).tobytes()

    def schedule(self, overhead: float, min_repair: int = 4) -> list[tuple[int, int]]:
        """(gen, esi) transmission order: generations interleaved round-robin so a burst of lost
        frames is spread over all generations; systematic symbols first, then repair symbols."""
        p = self.params
        per_gen = []
        for g in range(p.n_gens):
            k_g = p.gen_k(g)
            n_rep = max(min_repair, math.ceil(k_g * overhead))
            per_gen.append([(g, e) for e in range(k_g + n_rep)])
        order = []
        for i in range(max(len(x) for x in per_gen)):
            order.extend(x[i] for x in per_gen if i < len(x))
        return order


class GenerationDecoder:
    """Online Gauss-Jordan elimination over GF(2); rows kept fully reduced at all times."""

    def __init__(self, k_g: int, symbol_size: int):
        self.k = k_g
        self.coef = np.zeros((0, k_g), np.uint8)
        self.data = np.zeros((0, symbol_size), np.uint8)
        self.pivots: list[int] = []

    @property
    def rank(self) -> int:
        return len(self.pivots)

    @property
    def done(self) -> bool:
        return self.rank == self.k

    def add(self, coef: np.ndarray, data: bytes) -> bool:
        """Returns True if the symbol was innovative."""
        if self.done:
            return False
        c = coef.astype(np.uint8).copy()
        d = np.frombuffer(data, np.uint8).copy()
        if self.pivots:
            sel = c[self.pivots].astype(bool)
            if sel.any():
                c ^= np.bitwise_xor.reduce(self.coef[sel], axis=0)
                d ^= np.bitwise_xor.reduce(self.data[sel], axis=0)
        nz = np.flatnonzero(c)
        if len(nz) == 0:
            return False
        p = int(nz[0])
        hit = self.coef[:, p].astype(bool)
        if hit.any():
            self.coef[hit] ^= c
            self.data[hit] ^= d
        self.coef = np.vstack([self.coef, c])
        self.data = np.vstack([self.data, d])
        self.pivots.append(p)
        return True

    def recover(self) -> bytes:
        assert self.done
        out = np.empty_like(self.data)
        out[self.pivots] = self.data
        return out.tobytes()


class FountainDecoder:
    def __init__(self, params: FountainParams, session: int):
        self.params = params
        self.session = session
        self.gens = [GenerationDecoder(params.gen_k(g), params.symbol_size) for g in range(params.n_gens)]
        self.seen: set[tuple[int, int]] = set()

    def add(self, gen: int, esi: int, data: bytes) -> bool:
        if (gen, esi) in self.seen or not 0 <= gen < len(self.gens):
            return False
        self.seen.add((gen, esi))
        g = self.gens[gen]
        return g.add(coefficients(self.session, gen, esi, g.k), data)

    @property
    def done(self) -> bool:
        return all(g.done for g in self.gens)

    def progress(self) -> tuple[int, int]:
        return sum(g.rank for g in self.gens), self.params.k

    def recover(self) -> bytes:
        return b"".join(g.recover() for g in self.gens)[: self.params.total_len]
