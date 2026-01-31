"""Color histogram features."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def color_histogram(image: Image.Image, bins: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB image")

    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 255), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 255), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 255), density=True)
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    return hist


def serialize_hist(hist: np.ndarray) -> bytes:
    return hist.astype(np.float32).tobytes()


def deserialize_hist(blob: bytes, bins: int) -> np.ndarray:
    expected = bins * 3
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != expected:
        raise ValueError("Histogram dimension mismatch")
    return arr
