"""Training pipeline for embedding model (contrastive / SimCLR-style).

Changes vs your current version:
- Live terminal feedback via tqdm (loss + imgs/s).
- Cleaner imports + removed unused list_images_recursive.
- Avoid loading ALL paths into RAM by default (optional cap via cfg.train.max_images).
- Safer macOS defaults: persistent_workers only when num_workers>0.
- Removed redundant Resize before RandomResizedCrop.
- Optional: freeze backbone (transfer learning) via cfg.train.freeze_backbone.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from PIL import Image as PILImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from imagerec.config import Config
from imagerec.features.embedding import EmbeddingNet
from imagerec.io import iter_image_paths
from imagerec.utils import set_seed, train_val_split

logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """Returns two augmented views of the same image path."""

    def __init__(self, paths: List[str], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                # RandomResizedCrop already performs resizing; no need for a separate Resize before it.
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                view1 = self.transform(img)
                view2 = self.transform(img)
            return view1, view2
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            # Return a different sample instead of crashing
            # (simple retry: move to next index)
            new_idx = (idx + 1) % len(self.paths)
            return self.__getitem__(new_idx)



def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent loss (SimCLR). Complexity is O(B^2) due to similarity matrix."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    similarity = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask out self-similarity
    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)

    # positives are the off-diagonal pairs (i, i+B) and (i+B, i)
    positives = torch.cat([torch.diag(similarity, batch_size), torch.diag(similarity, -batch_size)])
    negatives = similarity[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=similarity.device)
    return F.cross_entropy(logits, labels)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _take_first_n_paths(cfg: Config) -> List[str]:
    """Collect training paths, optionally capped to avoid huge RAM usage.

    Add to config.yaml under train:
      max_images: 200000   # or null/omit for all
    """
    max_images = getattr(cfg.train, "max_images", None)
    paths: List[str] = []

    for p in iter_image_paths(cfg.dataset_path):
        paths.append(p)
        if max_images and len(paths) >= int(max_images):
            break

    return paths


def _maybe_freeze_backbone(model: EmbeddingNet, cfg: Config) -> None:
    """Optional transfer learning: freeze backbone, train only projection head.

    Add to config.yaml under train:
      freeze_backbone: true
    """
    freeze = bool(getattr(cfg.train, "freeze_backbone", False))
    if not freeze:
        return

    # In your EmbeddingNet, backbone is nn.Sequential and proj is nn.Linear
    for p in model.backbone.parameters():
        p.requires_grad = False
    logger.info("Backbone frozen: training only projection head.")


def train_embeddings(cfg: Config) -> None:
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    logger.info("Collecting image paths (may take time on first run)...")
    paths = _take_first_n_paths(cfg)
    if not paths:
        raise ValueError("No images found in dataset_path")

    train_paths, val_paths = train_val_split(paths, cfg.train.train_split, cfg.seed, cfg.train.val_split)
    logger.info("Training images: %d | validation images: %d", len(train_paths), len(val_paths))

    train_ds = ContrastiveDataset(train_paths, cfg.image_size)
    val_ds = ContrastiveDataset(val_paths, cfg.image_size)

    # macOS: num_workers>0 can sometimes feel like a "hang" during worker spin-up.
    # Start with num_workers=0 and increase only if stable.
    persistent_workers = bool(cfg.num_workers and cfg.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )

    device = _device()
    logger.info("Using device: %s", device)

    model = EmbeddingNet(embedding_dim=cfg.embedding_dim, pretrained=True).to(device)
    _maybe_freeze_backbone(model, cfg)

    # Only optimize trainable params (important if backbone frozen)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)

    start_epoch = 0
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if cfg.train.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        logger.info("Resumed from checkpoint at epoch %d", start_epoch)

    logger.info("Train batches per epoch: %d | Val batches: %d", len(train_loader), len(val_loader))
    logger.info("Starting training... (you will see a progress bar per epoch)")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", unit="batch")
        t0 = time.time()

        for step, (view1, view2) in enumerate(pbar, start=1):
            view1 = view1.to(device, non_blocking=False)
            view2 = view2.to(device, non_blocking=False)

            optimizer.zero_grad(set_to_none=True)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=cfg.train.temperature)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            # Update terminal feedback every ~10 steps
            if step % 10 == 0:
                dt = time.time() - t0
                imgs_per_sec = (cfg.batch_size / max(dt, 1e-6))
                t0 = time.time()
                pbar.set_postfix(loss=f"{loss.item():.4f}", ips=f"{imgs_per_sec:.1f}")

        avg_loss = total_loss / max(1, len(train_loader))
        val_loss = _eval(model, val_loader, cfg.train.temperature, device)
        logger.info(
            "Epoch %d/%d done - train loss: %.4f | val loss: %.4f",
            epoch + 1,
            cfg.train.epochs,
            avg_loss,
            val_loss,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    torch.save({"model_state": model.state_dict()}, cfg.embedding_model_path)
    logger.info("Saved embedding model to %s", cfg.embedding_model_path)


def _eval(model: nn.Module, loader: DataLoader, temperature: float, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for view1, view2 in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            total_loss += float(loss.item())
    return total_loss / max(1, len(loader))
