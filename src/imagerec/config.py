"""Configuration loader and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml




@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    temperature: float = 0.5
    resume: bool = True
    train_split: float = 0.9
    val_split: float = 0.1
    max_images: int | None = None          # NEU
    freeze_backbone: bool = False          # NEU



@dataclass
class Config:
    dataset_path: str = ""
    db_path: str = "./data/images.db"
    cache_dir: str = "./data/cache"
    ann_index_path: str = "./data/ann_hnsw.bin"
    embedding_model_path: str = "./data/embedding_model.pt"
    checkpoint_dir: str = "./data/checkpoints"
    image_size: int = 224
    color_hist_bins: int = 8
    phash_size: int = 8
    embedding_dim: int = 128
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    ann_space: str = "cosine"
    ann_m: int = 32
    ann_ef_construction: int = 200
    ann_ef_search: int = 64
    train: TrainConfig = field(default_factory=TrainConfig)

    def ensure_dirs(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return _from_dict(data)


def _from_dict(data: Dict[str, Any]) -> Config:
    train_data = data.get("train", {})
    train = TrainConfig(**train_data)

    cfg = Config(
        dataset_path=data.get("dataset_path", ""),
        db_path=data.get("db_path", "./data/images.db"),
        cache_dir=data.get("cache_dir", "./data/cache"),
        ann_index_path=data.get("ann_index_path", "./data/ann_hnsw.bin"),
        embedding_model_path=data.get("embedding_model_path", "./data/embedding_model.pt"),
        checkpoint_dir=data.get("checkpoint_dir", "./data/checkpoints"),
        image_size=int(data.get("image_size", 224)),
        color_hist_bins=int(data.get("color_hist_bins", 8)),
        phash_size=int(data.get("phash_size", 8)),
        embedding_dim=int(data.get("embedding_dim", 128)),
        batch_size=int(data.get("batch_size", 64)),
        num_workers=int(data.get("num_workers", 0)),
        seed=int(data.get("seed", 42)),
        ann_space=str(data.get("ann_space", "cosine")),
        ann_m=int(data.get("ann_m", 32)),
        ann_ef_construction=int(data.get("ann_ef_construction", 200)),
        ann_ef_search=int(data.get("ann_ef_search", 64)),
        train=train,
    )
    return cfg


def override_config(cfg: Config, overrides: Dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "train":
            for tkey, tval in value.items():
                if tval is not None:
                    setattr(cfg.train, tkey, tval)
        else:
            setattr(cfg, key, value)
    return cfg
