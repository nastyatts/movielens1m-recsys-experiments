from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "ml-1m"
PROCESSED = ROOT / "data" / "processed_1m"
OUT = ROOT / "outputs"
LLM_DATA = ROOT / "data" / "llm"

RATINGS = RAW / "ratings.dat"
MOVIES = RAW / "movies.dat"
USERS = RAW / "users.dat"
METADATA = RAW / "metadata.pkl"

MOVIE_METADATA = PROCESSED / "movie_metadata.csv"
SAMPLES = PROCESSED / "samples.csv"
TRAIN_INTERACTIONS = PROCESSED / "train_interactions.csv"
POPULARITY = PROCESSED / "movie_popularity_train.csv"
USERS_CSV = PROCESSED / "users.csv"
PREPROCESS_STATS = PROCESSED / "preprocess_stats.json"

EMB = OUT / "movie_text_embeddings.npy"
MOVIE_ORDER = OUT / "movie_id_order.npy"
MOVIE_TEXT = OUT / "movie_text.csv"

METHOD_FILES = {
    "faiss": "candidates_faiss_{split}.csv",
    "popularity": "candidates_popularity_{split}.csv",
    "random": "candidates_random_{split}.csv",
    "item_item": "candidates_item_item_{split}.csv",
    "hybrid": "candidates_hybrid_{split}.csv",
}


def ensure_dirs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LLM_DATA.mkdir(parents=True, exist_ok=True)


def parse_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    return [int(x) for x in text.split() if x] if text else []


def ids_to_text(values: Iterable[int]) -> str:
    return " ".join(str(int(x)) for x in values)


def genres(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {x.strip() for x in text.replace(",", "|").split("|") if x.strip()}


def recency_mean(vectors: np.ndarray, decay: float = 0.85) -> np.ndarray:
    n = len(vectors)
    if n == 0:
        raise ValueError("empty history")
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
    weights /= weights.sum()
    out = (vectors * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(out)
    return (out / norm).astype(np.float32) if norm > 0 else out.astype(np.float32)


def movie_pos(movie_order: Sequence[int]) -> dict[int, int]:
    return {int(mid): i for i, mid in enumerate(movie_order)}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
