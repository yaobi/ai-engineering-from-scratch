# Data Helper

## 1. Dataset loading

Use the Hugging Face datasets library to load datasets.

Example:

load_dataset("stanfordnlp/imdb")

The dataset is downloaded once and then cached locally.

---

## 2. Dataset splits

Machine learning projects usually use three splits:

- Train: used for model learning
- Validation: used to check progress during training
- Test: used only for final evaluation

Always set a seed when splitting data so the result is reproducible.

---

## 3. Streaming

Streaming reads large datasets row by row instead of downloading the whole dataset.

This is useful when the dataset is too large for local storage.

---

## 4. File formats

CSV:

Readable, but large and slow.

JSON:

Useful for APIs and nested data.

Parquet:

Smaller and faster. Better for AI and analytics workflows.

Arrow:

Used internally by the datasets library for fast in-memory processing.

---

## 5. Large files

Do not commit large datasets or model weights to Git.

Use .gitignore for personal projects.

Use Git LFS or DVC when a team needs reproducible large-file versioning.
