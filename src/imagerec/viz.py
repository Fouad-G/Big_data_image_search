"""2D visualization of embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from imagerec.config import Config
from imagerec.db import ImageDB
from imagerec.features.embedding import deserialize_embedding

logger = logging.getLogger(__name__)


def viz_embeddings(cfg: Config, output_path: str, sample_size: int = 2000) -> None:
    db = ImageDB(cfg.db_path)
    embeddings: List[np.ndarray] = []
    ids: List[int] = []

    for image_id, dim, blob in db.iter_embeddings():
        embeddings.append(deserialize_embedding(blob, dim))
        ids.append(image_id)
        if len(embeddings) >= sample_size:
            break

    if not embeddings:
        raise ValueError("No embeddings available for visualization")

    data = np.vstack(embeddings)
    tsne = TSNE(n_components=2, init="random", random_state=cfg.seed)
    reduced = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=8, alpha=0.7)
    plt.title("Embedding t-SNE")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    db.close()
    logger.info("Saved visualization to %s", output_path)
