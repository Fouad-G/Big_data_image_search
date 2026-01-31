"""Command line interface."""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List

from imagerec.ann import build_ann
from imagerec.config import Config, load_config, override_config
from imagerec.db import ImageDB
from imagerec.indexer import index_dataset
from imagerec.logging_utils import setup_logging
from imagerec.profile import profile_index, profile_query
from imagerec.query import query_images
from imagerec.train import train_embeddings
from imagerec.viz import viz_embeddings

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="imagerec")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index dataset and compute features")

    train_parser = subparsers.add_parser("train-embeddings", help="Train or fine-tune embedding model")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)

    ann_parser = subparsers.add_parser("build-ann", help="Build ANN index")

    query_parser = subparsers.add_parser("query", help="Query similar images")
    query_parser.add_argument("--image", dest="images", nargs="+", required=True, help="Path(s) to query image(s)")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--w-embedding", type=float, default=0.7)
    query_parser.add_argument("--w-color", type=float, default=0.2)
    query_parser.add_argument("--w-phash", type=float, default=0.1)

    profile_parser = subparsers.add_parser("profile", help="Profile indexing or query")
    profile_parser.add_argument("--mode", choices=["index", "query"], default="query")
    profile_parser.add_argument("--image", help="Query image path for profiling")

    viz_parser = subparsers.add_parser("viz-2d", help="Visualize embeddings in 2D")
    viz_parser.add_argument("--output", default="./data/embeddings_tsne.png")
    viz_parser.add_argument("--sample-size", type=int, default=2000)

    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    overrides = {}
    if getattr(args, "batch_size", None):
        overrides["batch_size"] = args.batch_size
    train_overrides = {}
    if getattr(args, "epochs", None):
        train_overrides["epochs"] = args.epochs
    if train_overrides:
        overrides["train"] = train_overrides
    cfg = override_config(cfg, overrides)

    if args.command == "index":
        index_dataset(cfg)
    elif args.command == "train-embeddings":
        train_embeddings(cfg)
    elif args.command == "build-ann":
        db = ImageDB(cfg.db_path)
        build_ann(db, cfg.ann_index_path, cfg.ann_space, cfg.embedding_dim, cfg.ann_m, cfg.ann_ef_construction)
        db.close()
    elif args.command == "query":
        weights = {"embedding": args.w_embedding, "color": args.w_color, "phash": args.w_phash}
        results = query_images(cfg, args.images, args.top_k, weights)
        for path, score in results:
            print(f"{score:.4f}\t{path}")
    elif args.command == "profile":
        if args.mode == "index":
            profile_index(cfg)
        else:
            if not args.image:
                raise SystemExit("--image is required for profile query")
            profile_query(cfg, args.image)
    elif args.command == "viz-2d":
        viz_embeddings(cfg, args.output, args.sample_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
