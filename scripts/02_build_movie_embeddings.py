from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from common import EMB, MOVIE_METADATA, MOVIE_ORDER, MOVIE_TEXT, ensure_dirs


FIELDS = ["Plot", "Director", "Actors", "Language", "Country", "Runtime", "imdbRating", "Metascore", "Production"]


def clean(x: object) -> str:
    return "" if pd.isna(x) or str(x).strip() in {"", "N/A", "nan"} else str(x).strip()


def text(row: pd.Series) -> str:
    parts = [clean(row.title), f"Genres: {clean(row.genres).replace('|', ', ')}"]
    for field in FIELDS:
        value = clean(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    return ". ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    ensure_dirs()
    movies = pd.read_csv(MOVIE_METADATA).sort_values("movie_id").reset_index(drop=True)
    movies["movie_text"] = movies.apply(text, axis=1)
    model = SentenceTransformer(args.model_name)
    emb = model.encode(movies["movie_text"].tolist(), batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    np.save(EMB, emb)
    np.save(MOVIE_ORDER, movies["movie_id"].astype(np.int32).to_numpy())
    movies[["movie_id", "movie_text"]].to_csv(MOVIE_TEXT, index=False)
    print(f"embeddings={EMB} shape={emb.shape}")


if __name__ == "__main__":
    main()

