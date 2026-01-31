"""Embedding model and extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        emb = self.proj(feats)
        return emb


class EmbeddingExtractor:
    def __init__(self, model: EmbeddingNet, device: torch.device, image_size: int) -> None:
        self.model = model
        self.device = device
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def load(
        cls, model_path: str, embedding_dim: int, device: torch.device, image_size: int
    ) -> "EmbeddingExtractor":
        model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
        path = Path(model_path)
        if path.exists():
            state = torch.load(path, map_location=device)
            model.load_state_dict(state["model_state"] if "model_state" in state else state)
            logger.info("Loaded embedding model from %s", model_path)
        else:
            logger.warning("Embedding model not found at %s, using random init", model_path)
        model.to(device)
        return cls(model, device, image_size)

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([self.transform(img) for img in images]).to(self.device)
            embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, dim=1)
            return embeddings.cpu().numpy().astype(np.float32)


def serialize_embedding(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        raise ValueError("Embedding dimension mismatch")
    return arr
