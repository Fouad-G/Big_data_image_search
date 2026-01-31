"""Approximate nearest neighbor index using hnswlib."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import hnswlib
import numpy as np

from imagerec.db import ImageDB
from imagerec.features.embedding import deserialize_embedding

logger = logging.getLogger(__name__)


class AnnIndex:
    def __init__(self, space: str, dim: int) -> None:
        self.index = hnswlib.Index(space=space, dim=dim)
        self.dim = dim
        self.space = space

    def init(self, max_elements: int, m: int, ef_construction: int) -> None:
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m)

    def add_items(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        self.index.add_items(vectors, labels)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(path)

    def load(self, path: str) -> None:
        self.index.load_index(path)

    def set_ef(self, ef: int) -> None:
        self.index.set_ef(ef)

    def query(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        labels, distances = self.index.knn_query(vector, k=k)
        return labels[0], distances[0]


def build_ann(db: ImageDB, index_path: str, space: str, dim: int, m: int, ef_construction: int) -> None:
    total = db.count_embeddings()
    logger.info("Building ANN index for %d embeddings", total)

    ann = AnnIndex(space=space, dim=dim)
    ann.init(max_elements=total, m=m, ef_construction=ef_construction)

    batch_vectors: List[np.ndarray] = []
    batch_labels: List[int] = []

    for image_id, _, blob in db.iter_embeddings():
        vec = deserialize_embedding(blob, dim)
        batch_vectors.append(vec)
        batch_labels.append(image_id)
        if len(batch_vectors) >= 2048:
            ann.add_items(np.vstack(batch_vectors), np.array(batch_labels))
            batch_vectors = []
            batch_labels = []

    if batch_vectors:
        ann.add_items(np.vstack(batch_vectors), np.array(batch_labels))

    ann.save(index_path)
    logger.info("Saved ANN index to %s", index_path)


def load_ann(index_path: str, space: str, dim: int) -> AnnIndex:
    ann = AnnIndex(space=space, dim=dim)
    ann.load(index_path)
    return ann
