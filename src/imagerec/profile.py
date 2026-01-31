"""Profiling utilities."""

from __future__ import annotations

import cProfile
import logging
from pathlib import Path
from typing import Optional

from imagerec.config import Config
from imagerec.indexer import index_dataset
from imagerec.query import query_images

logger = logging.getLogger(__name__)


def profile_index(cfg: Config, output: Optional[str] = None) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    index_dataset(cfg)
    profiler.disable()
    out = output or str(Path(cfg.cache_dir) / "profile_index.prof")
    profiler.dump_stats(out)
    logger.info("Wrote index profile to %s", out)


def profile_query(cfg: Config, query_path: str, output: Optional[str] = None) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    query_images(cfg, [query_path], top_k=5, weights={"embedding": 0.7, "color": 0.2, "phash": 0.1})
    profiler.disable()
    out = output or str(Path(cfg.cache_dir) / "profile_query.prof")
    profiler.dump_stats(out)
    logger.info("Wrote query profile to %s", out)
