"""Perceptual hash (difference hash)."""

from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image

_U64_MASK: Final[int] = (1 << 64) - 1
_I64_MAX: Final[int] = (1 << 63) - 1
_U64_MOD: Final[int] = 1 << 64


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a 64-bit dHash stored as *signed* int64-compatible value for SQLite."""
    img = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(img)
    diff = pixels[:, 1:] > pixels[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)

    # For hash_size=8 we have 64 bits; SQLite INTEGER is signed int64.
    # Convert unsigned 64-bit into signed range [-2^63, 2^63-1].
    value &= _U64_MASK
    if value > _I64_MAX:
        value -= _U64_MOD
    return int(value)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance for signed-stored hashes: mask back to 64-bit then XOR."""
    x = (a & _U64_MASK) ^ (b & _U64_MASK)
    return int(x.bit_count())
