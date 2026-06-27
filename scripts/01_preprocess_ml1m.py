from __future__ import annotations

import argparse
import pickle

import pandas as pd

from common import (
    METADATA,
    MOVIE_METADATA,
    MOVIES,
    POPULARITY,
    PREPROCESS_STATS,
    RATINGS,
    SAMPLES,
    TRAIN_INTERACTIONS,
    USERS,
    USERS_CSV,
    ensure_dirs,
    ids_to_text,
)

META_FIELDS = ["Plot", "Director", "Actors", "Language", "Country", "Runtime", "imdbRating", "Metascore", "Production"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_history", type=int, default=5)
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--max_history", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 keeps all train samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()

    ratings = pd.read_csv(RATINGS, sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    movies = pd.read_csv(MOVIES, sep="::", engine="python", encoding="latin-1", names=["movie_id", "title", "genres"])
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    users = pd.read_csv(USERS, sep="::", engine="python", names=["user_id", "gender", "age", "occupation", "zip_code"])
    users = users.drop(columns=["zip_code"])

    with open(METADATA, "rb") as f:
        raw_meta = pickle.load(f)
    rows = []
    for key, meta in raw_meta:
        movie_id, metadata_title, imdb_id = key
        row = {"movie_id": int(movie_id), "metadata_title": metadata_title, "imdbId": imdb_id}
        for field in META_FIELDS:
            value = meta.get(field, "") if isinstance(meta, dict) else ""
            row[field] = "" if value == "N/A" else value
        rows.append(row)
    metadata = pd.DataFrame(rows).drop_duplicates("movie_id")
    movie_metadata = movies.merge(metadata, on="movie_id", how="left", validate="one_to_one")
    movie_metadata.to_csv(MOVIE_METADATA, index=False)
    users.to_csv(USERS_CSV, index=False)

    raw_groups = {
        int(user_id): group.sort_values(["timestamp", "movie_id"]).copy()
        for user_id, group in ratings.groupby("user_id", sort=True)
    }
    positive = ratings[ratings["rating"] >= args.min_rating].copy()
    positive = positive.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=False)
    sample_rows = []
    train_parts = []
    for user_id, pos_group in positive.groupby("user_id", sort=True):
        pos_group = pos_group.sort_values(["timestamp", "movie_id"]).reset_index(drop=True)
        raw_group = raw_groups[int(user_id)]
        raw_movie_ids = raw_group["movie_id"].astype(int).tolist()
        raw_index_to_pos = {int(raw_index): pos for pos, raw_index in enumerate(raw_group.index.tolist())}
        if len(pos_group) < args.min_history + 3:
            continue

        train_parts.append(pos_group.iloc[: len(pos_group) - 2][["user_id", "movie_id", "rating", "timestamp"]])
        split_positions = [(p, "train") for p in range(args.min_history, len(pos_group) - 2)]
        split_positions += [(len(pos_group) - 2, "val"), (len(pos_group) - 1, "test")]
        for pos, split in split_positions:
            target = pos_group.iloc[pos]
            full_history = pos_group.iloc[:pos]["movie_id"].astype(int).tolist()
            model_history = full_history[-args.max_history:] if args.max_history > 0 else full_history
            raw_pos = raw_index_to_pos[int(target["index"])]
            seen_before = raw_movie_ids[:raw_pos]
            sample_rows.append(
                {
                    "user_id": int(user_id),
                    "split": split,
                    "target_movie_id": int(target.movie_id),
                    "target_rating": float(target.rating),
                    "target_timestamp": int(target.timestamp),
                    "model_history_movie_ids": ids_to_text(model_history),
                    "full_history_movie_ids": ids_to_text(full_history),
                    "seen_movie_ids": ids_to_text(seen_before),
                    "history_len": len(model_history),
                    "full_history_len": len(full_history),
                    "seen_len": len(seen_before),
                }
            )

    samples = pd.DataFrame(sample_rows).sort_values(["split", "user_id"]).reset_index(drop=True)
    train_mask = samples["split"].eq("train")
    if args.max_train_samples > 0 and train_mask.sum() > args.max_train_samples:
        train = samples[train_mask].sample(args.max_train_samples, random_state=args.seed)
        samples = pd.concat([train, samples[~train_mask]], ignore_index=True).sort_values(["split", "user_id"]).reset_index(drop=True)

    leaked = samples.apply(lambda r: str(int(r.target_movie_id)) in set(str(r.full_history_movie_ids).split()), axis=1).sum()
    if leaked:
        raise AssertionError(f"target leakage in {leaked} rows")

    train_interactions = pd.concat(train_parts, ignore_index=True)
    train_interactions.to_csv(TRAIN_INTERACTIONS, index=False)
    popularity = train_interactions.groupby("movie_id").size().rename("train_popularity").reset_index()
    popularity.to_csv(POPULARITY, index=False)
    samples.to_csv(SAMPLES, index=False)
    stats = {
        "num_train": int(samples["split"].eq("train").sum()),
        "num_val": int(samples["split"].eq("val").sum()),
        "num_test": int(samples["split"].eq("test").sum()),
        "avg_history_len": float(samples["history_len"].mean()),
        "avg_full_history_len": float(samples["full_history_len"].mean()),
        "rating_filter": f"rating >= {args.min_rating}",
        "min_rating": float(args.min_rating),
        "min_history": int(args.min_history),
        "max_history": int(args.max_history),
        "max_train_samples": int(args.max_train_samples),
        "num_users": int(samples["user_id"].nunique()),
    }
    from common import write_json

    write_json(PREPROCESS_STATS, stats)
    print(f"movie_metadata={MOVIE_METADATA} rows={len(movie_metadata)}")
    print(f"samples={SAMPLES} counts={samples['split'].value_counts().to_dict()}")
    print(f"popularity={POPULARITY} rows={len(popularity)}")
    print(f"stats={PREPROCESS_STATS} {stats}")


if __name__ == "__main__":
    main()
