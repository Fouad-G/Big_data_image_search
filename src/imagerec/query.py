"""Query logic for image similarity."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from imagerec.ann import load_ann
from imagerec.config import Config
from imagerec.db import ImageDB
from imagerec.features.color import color_histogram, deserialize_hist
from imagerec.features.embedding import EmbeddingExtractor, deserialize_embedding
from imagerec.features.phash import dhash
from imagerec.io import load_image
from imagerec.similarity import combine_scores, color_similarity, embedding_similarity, mean_scores, phash_similarity

logger = logging.getLogger(__name__)


def query_images(cfg: Config, query_paths: Sequence[str], top_k: int, weights: Dict[str, float]) -> List[Tuple[str, float]]:
    db = ImageDB(cfg.db_path)

    device = _device()
    extractor = EmbeddingExtractor.load(cfg.embedding_model_path, cfg.embedding_dim, device, cfg.image_size)
    ann = load_ann(cfg.ann_index_path, cfg.ann_space, cfg.embedding_dim)
    ann.set_ef(cfg.ann_ef_search)

    query_features = _compute_query_features(cfg, extractor, query_paths)

    candidate_ids = _ann_candidates(ann, query_features["embedding"][0], top_k=top_k * 20)
    if not candidate_ids:
        return []

    color_map = db.fetch_color_hist(candidate_ids)
    emb_map = db.fetch_embeddings(candidate_ids)
    phash_map = db.fetch_phash(candidate_ids)

    results: List[Tuple[str, float]] = []
    for image_id in candidate_ids:
        if image_id not in emb_map:
            continue

        emb_vec = deserialize_embedding(emb_map[image_id], cfg.embedding_dim)
        emb_scores = [embedding_similarity(vec, emb_vec) for vec in query_features["embedding"]]
        emb_score = mean_scores(emb_scores)

        color_score = 0.0
        if image_id in color_map:
            hist = deserialize_hist(color_map[image_id], cfg.color_hist_bins)
            color_scores = [color_similarity(h, hist) for h in query_features["color_hist"]]
            color_score = mean_scores(color_scores)

        phash_score = 0.0
        if image_id in phash_map:
            phash_scores = [phash_similarity(h, phash_map[image_id], cfg.phash_size) for h in query_features["phash"]]
            phash_score = mean_scores(phash_scores)

        score = combine_scores(
            {"embedding": emb_score, "color": color_score, "phash": phash_score},
            weights,
        )
        path = db.fetch_image(image_id)[1]
        results.append((path, float(score)))

    db.close()
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def _compute_query_features(cfg: Config, extractor: EmbeddingExtractor, query_paths: Sequence[str]) -> Dict[str, List]:
    embeddings: List[np.ndarray] = []
    color_hists: List[np.ndarray] = []
    phashes: List[int] = []

    images = []
    for path in query_paths:
        img = load_image(path, cfg.image_size)
        images.append(img)
        color_hists.append(color_histogram(img, cfg.color_hist_bins))
        phashes.append(dhash(img, cfg.phash_size))

    embeddings = list(extractor.encode_batch(images))

    return {"embedding": embeddings, "color_hist": color_hists, "phash": phashes}


def _ann_candidates(ann, query_vec: np.ndarray, top_k: int) -> List[int]:
    labels, _ = ann.query(query_vec, k=top_k)
    return [int(x) for x in labels]


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
