"""General utilities."""

from __future__ import annotations

import os
import random
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def batch_iter(items: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_val_split(
    items: Sequence[T], train_ratio: float, seed: int, val_ratio: Optional[float] = None
) -> Tuple[List[T], List[T]]:
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    if val_ratio is None:
        val_ratio = 1.0 - train_ratio
    val_ratio = max(0.0, min(val_ratio, 1.0))
    train_ratio = max(0.0, min(train_ratio, 1.0))
    if train_ratio + val_ratio > 1.0:
        val_ratio = 1.0 - train_ratio
    split = int(len(items) * train_ratio)
    train_idx = indices[:split]
    val_count = int(len(items) * val_ratio)
    val_idx = indices[split : split + val_count]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items
