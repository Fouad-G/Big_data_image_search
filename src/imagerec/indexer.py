"""Dataset indexing and feature extraction."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from imagerec.config import Config
from imagerec.db import ImageDB
from imagerec.features.color import color_histogram, serialize_hist
from imagerec.features.embedding import EmbeddingExtractor, serialize_embedding
from imagerec.features.phash import dhash
from imagerec.io import iter_image_paths, load_image, image_metadata
from imagerec.utils import batch_iter

logger = logging.getLogger(__name__)


def index_dataset(cfg: Config) -> None:
    cfg.ensure_dirs()
    db = ImageDB(cfg.db_path)
    db.init_schema()

    device = _device()
    extractor = EmbeddingExtractor.load(cfg.embedding_model_path, cfg.embedding_dim, device, cfg.image_size)

    paths_iter = iter_image_paths(cfg.dataset_path)
    for batch_paths in tqdm(batch_iter(paths_iter, cfg.batch_size), desc="Indexing", unit="batch"):
        images = []
        metadata_rows: List[Tuple[str, int, int, str]] = []
        valid_paths: List[str] = []

        for path in batch_paths:
            try:
                width, height, fmt = image_metadata(path)
                img = load_image(path, cfg.image_size)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Skipping %s: %s", path, exc)
                continue
            images.append(img)
            metadata_rows.append((path, width, height, fmt))
            valid_paths.append(path)

        if not metadata_rows:
            continue

        id_map = db.insert_images(metadata_rows)
        paired = [(img, id_map[path]) for img, path in zip(images, valid_paths) if path in id_map]
        if not paired:
            continue

        image_ids = [image_id for _, image_id in paired]

        hist_rows = []
        phash_rows = []
        for img, image_id in paired:
            hist = color_histogram(img, cfg.color_hist_bins)
            hist_rows.append((image_id, cfg.color_hist_bins, serialize_hist(hist)))
            phash_rows.append((image_id, dhash(img, cfg.phash_size)))

        db.insert_color_hist(hist_rows)
        db.insert_phash(phash_rows)

        embeddings = extractor.encode_batch([img for img, _ in paired])
        emb_rows = []
        for image_id, vec in zip(image_ids, embeddings):
            emb_rows.append((image_id, cfg.embedding_dim, serialize_embedding(vec)))
        db.insert_embeddings(emb_rows)

    db.close()
    logger.info("Indexing complete")


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
