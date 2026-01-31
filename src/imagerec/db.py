"""SQLite database layer."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    width INTEGER,
    height INTEGER,
    format TEXT
);

CREATE TABLE IF NOT EXISTS color_hist (
    image_id INTEGER PRIMARY KEY,
    bins INTEGER NOT NULL,
    hist BLOB NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    image_id INTEGER PRIMARY KEY,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS phash (
    image_id INTEGER PRIMARY KEY,
    hash INTEGER NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
CREATE INDEX IF NOT EXISTS idx_color_hist_image ON color_hist(image_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_image ON embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_phash_image ON phash(image_id);
"""


class ImageDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def insert_images(self, rows: Sequence[Tuple[str, int, int, str]]) -> Dict[str, int]:
        if not rows:
            return {}
        with self.transaction() as cur:
            cur.executemany(
                "INSERT OR IGNORE INTO images(path, width, height, format) VALUES (?, ?, ?, ?)",
                rows,
            )
        paths = [row[0] for row in rows]
        mapping: Dict[str, int] = {}
        for chunk in _chunk(paths, 900):
            placeholders = ",".join(["?"] * len(chunk))
            query = f"SELECT id, path FROM images WHERE path IN ({placeholders})"
            for image_id, path in self.conn.execute(query, chunk):
                mapping[path] = image_id
        return mapping

    def insert_color_hist(self, rows: Sequence[Tuple[int, int, bytes]]) -> None:
        if not rows:
            return
        with self.transaction() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO color_hist(image_id, bins, hist) VALUES (?, ?, ?)",
                rows,
            )

    def insert_embeddings(self, rows: Sequence[Tuple[int, int, bytes]]) -> None:
        if not rows:
            return
        with self.transaction() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO embeddings(image_id, dim, vector) VALUES (?, ?, ?)",
                rows,
            )

    def insert_phash(self, rows: Sequence[Tuple[int, int]]) -> None:
        if not rows:
            return
        with self.transaction() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO phash(image_id, hash) VALUES (?, ?)" ,
                rows,
            )

    def fetch_image(self, image_id: int) -> Tuple[int, str]:
        row = self.conn.execute("SELECT id, path FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(f"Image id not found: {image_id}")
        return int(row[0]), str(row[1])

    def fetch_color_hist(self, image_ids: Sequence[int]) -> Dict[int, bytes]:
        return _fetch_blob_map(self.conn, "color_hist", "hist", image_ids)

    def fetch_embeddings(self, image_ids: Sequence[int]) -> Dict[int, bytes]:
        return _fetch_blob_map(self.conn, "embeddings", "vector", image_ids)

    def fetch_phash(self, image_ids: Sequence[int]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for chunk in _chunk(image_ids, 900):
            placeholders = ",".join(["?"] * len(chunk))
            query = f"SELECT image_id, hash FROM phash WHERE image_id IN ({placeholders})"
            for image_id, value in self.conn.execute(query, chunk):
                mapping[int(image_id)] = int(value)
        return mapping

    def count_embeddings(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0]) if row else 0

    def iter_embeddings(self) -> Iterator[Tuple[int, int, bytes]]:
        query = "SELECT image_id, dim, vector FROM embeddings"
        for image_id, dim, vector in self.conn.execute(query):
            yield int(image_id), int(dim), vector


def _chunk(items: Iterable, size: int) -> Iterator[List]:
    batch: List = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fetch_blob_map(conn: sqlite3.Connection, table: str, col: str, image_ids: Sequence[int]) -> Dict[int, bytes]:
    mapping: Dict[int, bytes] = {}
    for chunk in _chunk(image_ids, 900):
        placeholders = ",".join(["?"] * len(chunk))
        query = f"SELECT image_id, {col} FROM {table} WHERE image_id IN ({placeholders})"
        for image_id, blob in conn.execute(query, chunk):
            mapping[int(image_id)] = blob
    return mapping
