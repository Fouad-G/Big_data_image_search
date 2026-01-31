import numpy as np

from imagerec.similarity import color_similarity, embedding_similarity, phash_similarity
from imagerec.features.phash import hamming_distance


def test_color_similarity_identity():
    hist = np.array([0.1, 0.2, 0.3, 0.4] * 3, dtype=np.float32)
    score = color_similarity(hist, hist)
    assert score == 1.0


def test_embedding_similarity_identity():
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    score = embedding_similarity(vec, vec)
    assert abs(score - 1.0) < 1e-6


def test_phash_similarity():
    a = 0b101010
    b = 0b101011
    dist = hamming_distance(a, b)
    score = phash_similarity(a, b, hash_size=2)
    assert dist == 1
    assert 0.0 <= score <= 1.0
