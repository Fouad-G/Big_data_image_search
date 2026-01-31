"""Similarity metrics and combining logic."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from imagerec.features.phash import hamming_distance


def color_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    denom = hist_a + hist_b + 1e-8
    num = (hist_a - hist_b) ** 2
    return 0.5 * float(np.sum(num / denom))


def color_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    dist = color_distance(hist_a, hist_b)
    return 1.0 / (1.0 + dist)


def embedding_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def phash_similarity(hash_a: int, hash_b: int, hash_size: int = 8) -> float:
    max_bits = hash_size * hash_size
    dist = hamming_distance(hash_a, hash_b)
    return 1.0 - (dist / max_bits)


def combine_scores(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    weighted = 0.0
    for key, weight in weights.items():
        weighted += scores.get(key, 0.0) * weight
    return weighted / total_weight


def mean_scores(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(np.mean(values_list))
