"""Dataset scanning and image loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def iter_image_paths(dataset_path: str) -> Iterator[str]:
    root = Path(dataset_path)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield str(path)


def load_image(path: str, image_size: Optional[int] = None) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size is not None:
            img = img.resize((image_size, image_size))
        return img


def iter_images(dataset_path: str, image_size: Optional[int] = None) -> Generator[Tuple[str, Image.Image], None, None]:
    for path in iter_image_paths(dataset_path):
        try:
            img = load_image(path, image_size=image_size)
            yield path, img
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load %s: %s", path, exc)
            continue


def image_metadata(path: str) -> Tuple[int, int, str]:
    with Image.open(path) as img:
        return img.width, img.height, img.format or ""
