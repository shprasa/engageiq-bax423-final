from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def stable_hash64(s: str) -> int:
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little", signed=False)


class BloomFilter:
    def __init__(self, capacity: int, fp_rate: float = 0.01):
        capacity = max(1, int(capacity))
        fp_rate = min(max(fp_rate, 1e-6), 0.25)

        m = int(-(capacity * math.log(fp_rate)) / (math.log(2) ** 2))
        k = max(1, int((m / capacity) * math.log(2)))

        self.m = max(8, m)
        self.k = k
        self.bits = np.zeros(self.m, dtype=np.bool_)

    def _hashes(self, key: str) -> Iterable[int]:
        h = stable_hash64(key)
        h2 = stable_hash64("salt|" + key) | 1
        for i in range(self.k):
            yield (h + i * h2) % self.m

    def add(self, key: str) -> None:
        for idx in self._hashes(key):
            self.bits[idx] = True

    def __contains__(self, key: str) -> bool:
        return all(self.bits[idx] for idx in self._hashes(key))


@dataclass
class CountMinSketch:
    width: int = 4096
    depth: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.depth = int(self.depth)
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)

    def _hash(self, s: str, i: int) -> int:
        return stable_hash64(f"{self.seed + i}|{s}") % self.width

    def add(self, key: str, count: int = 1) -> None:
        c = int(count)
        for i in range(self.depth):
            self.table[i, self._hash(key, i)] += c

    def estimate(self, key: str) -> int:
        return int(
            min(self.table[i, self._hash(key, i)] for i in range(self.depth))
        )


class HyperLogLog:
    def __init__(self, p: int = 10):
        if not (4 <= p <= 16):
            raise ValueError("p must be between 4 and 16")
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)

        # bias-correction constant
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        if w == 0:
            return max_bits + 1
        return (w ^ (w - 1)).bit_length()

    def add(self, key: str) -> None:
        x = stable_hash64(key)
        idx = x & (self.m - 1)
        w = x >> self.p
        rank = self._rho(w, 64 - self.p)
        self.registers[idx] = max(self.registers[idx], rank)

    def count(self) -> int:
        z = np.sum(2.0 ** (-self.registers.astype(np.float64)))
        e = self.alpha * (self.m**2) / z

        # small range correction
        v = int(np.sum(self.registers == 0))
        if e <= 2.5 * self.m and v > 0:
            e = self.m * math.log(self.m / v)
        return int(e)

