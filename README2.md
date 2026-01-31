````md
# Image Recommender (DAISY – Big Data Engineering)

A local, modular image recommender system designed for large-scale image datasets (≈550,000 images).  
The project follows Big Data Engineering principles: generator-based data loading, disk-backed storage via SQLite, self-implemented similarity measures, ANN-based retrieval, profiling, and reproducible configuration.

---

## Requirements

- **macOS on Apple Silicon (M1/M2)** (tested on M1, 8 GB RAM)
- **Python 3.10+** (tested with Python 3.13)
- Local SSD / external SSD recommended for large datasets
- Optional: CUDA support on Linux/Windows (not used in this project)

---

## Setup

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

3. Install the package (editable mode):

   ```bash
   pip install -e .
   ```
4. Prepare configuration files:

   * `config_miniset.yaml` → small subset for testing
   * `config_full.yaml` → full dataset pipeline (≈550k images)

---

## Configuration Profiles

This project uses **multiple config files** for reproducibility and scalability.

### `config_miniset.yaml`

* Small subset (≈1–2k images)
* Fast testing of the full pipeline
* Output written to:

  ```
  ./data/miniset/
  ```

### `config_full.yaml`

* Full dataset (`/Volumes/BigDataB/data/image_data`)
* Separate output directories:

  ```
  ./data/full/images.db
  ./data/full/embedding_model.pt
  ./data/full/ann_hnsw.bin
  ./data/full/cache/
  ```
* Training uses a **controlled subset (e.g. 50,000 images)** via:

  ```yaml
  train:
    max_images: 50000
  ```

  This constraint is necessary due to limited hardware resources (MacBook M1, 8 GB RAM).

---

## CLI Usage

All commands are executed via:

```bash
python -m imagerec --config <config_file> <command> [options]
```

---

## 1) Train Embedding Model (SimCLR-style)

```bash
python -m imagerec --config config_full.yaml train-embeddings
```

Optional overrides:

```bash
python -m imagerec --config config_full.yaml train-embeddings --epochs 1 --batch-size 64
```

**Description**

* Contrastive self-supervised training (SimCLR-style)
* Backbone: **ResNet-18**
* Two augmented views per image (crop, flip, color jitter)
* NT-Xent loss
* Checkpointing + resume support
* Training/validation split (e.g. 45k / 5k images)

**Output**

* `embedding_model.pt`
* `checkpoint_dir/checkpoint.pt`

---

## 2) Index Dataset

```bash
python -m imagerec --config config_full.yaml index
```

**Description**

* Generator-based dataset scan (no full RAM load)
* Assigns image IDs
* Computes and stores:

  * Color histograms
  * Perceptual hashes (dHash)
  * Embeddings (using trained model)
* Stores everything in SQLite

**Output**

* `images.db` (SQLite database)

---

## 3) Build ANN Index (HNSW)

```bash
python -m imagerec --config config_full.yaml build-ann
```

**Description**

* Builds an HNSW ANN index over stored embeddings
* Enables fast approximate nearest neighbor search
* Built once and reused

**Output**

* `ann_hnsw.bin`

---

## 4) Query Similar Images

```bash
python -m imagerec --config config_full.yaml query \
  --image /path/to/query.jpg \
  --top-k 5
```

Multiple query images with weighted fusion:

```bash
python -m imagerec --config config_full.yaml query \
  --image img1.jpg img2.jpg \
  --top-k 5 \
  --w-embedding 0.7 \
  --w-color 0.2 \
  --w-phash 0.1
```

**Description**

* ANN generates candidate set
* Similarity computed using:

  * Embedding cosine similarity
  * Color histogram similarity
  * Perceptual hash similarity
* Weighted fusion and ranking

---

## 5) Profiling

```bash
python -m imagerec --config config_full.yaml profile --mode query --image img.jpg
python -m imagerec --config config_full.yaml profile --mode index
```

**Output**

* `profile_query.prof`
* `profile_index.prof`

Inspect with:

```bash
python -m pstats data/full/cache/profile_index.prof
```

---

## 6) 2D Embedding Visualization

```bash
python -m imagerec --config config_full.yaml viz-2d \
  --sample-size 2000 \
  --output ./data/full/embeddings_tsne.png
```

**Description**

* Samples embeddings from the SQLite database
* Dimensionality reduction via t-SNE
* Visual explainability of learned embedding space

**Output**

* `embeddings_tsne.png`

---

## Project Structure

```
.
├── config_miniset.yaml
├── config_full.yaml
├── README.md
├── requirements.txt
├── src/imagerec/
│   ├── cli.py
│   ├── config.py
│   ├── io.py
│   ├── db.py
│   ├── indexer.py
│   ├── train.py
│   ├── ann.py
│   ├── query.py
│   ├── similarity.py
│   ├── viz.py
│   ├── profile.py
│   └── features/
├── tests/
│   ├── test_db.py
│   └── test_similarity.py
```

---

## Artifacts Produced

* `images.db` – SQLite database with metadata and features
* `embedding_model.pt` – trained embedding model
* `ann_hnsw.bin` – ANN index
* `embeddings_tsne.png` – visualization
* `profile_*.prof` – profiling outputs

---

## Data Cleaning & Robustness

* Skips `.DS_Store` and AppleDouble `._*` files
* Handles:

  * Corrupted images
  * Truncated files
  * PIL `DecompressionBombWarning`
* Failed images are skipped without stopping the pipeline

---

## Testing & Quality Assurance

Automated tests:

```bash
pytest
```

Tests cover:

* Database insert/fetch correctness
* Similarity and distance functions

Manual integration tests:

* Full CLI pipeline tested on:

  * Mini dataset
  * Full dataset

---

## Scalability & Engineering Notes

* Streaming iterators keep RAM usage bounded
* SQLite enables disk-backed feature storage
* ANN indexing scales linearly with dataset size
* Training limited to 50k images due to hardware constraints (MacBook M1, 8 GB RAM)
* System designed to scale further on stronger hardware

```
```
