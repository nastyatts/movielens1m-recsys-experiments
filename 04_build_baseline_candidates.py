from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import LLM_DATA, OUT, POPULARITY, SAMPLES, TRAIN_INTERACTIONS, ensure_dirs, parse_ids


def top_popularity(popularity: pd.DataFrame) -> list[tuple[int, float]]:
    pop = popularity.sort_values(["train_popularity", "movie_id"], ascending=[False, True])
    return [(int(r.movie_id), float(r.train_popularity)) for r in pop.itertuples(index=False)]


def build_item_neighbors(train: pd.DataFrame, window: int) -> dict[int, list[tuple[int, float]]]:
    cooc: dict[int, Counter[int]] = defaultdict(Counter)
    for _, group in train.sort_values(["user_id", "timestamp", "movie_id"]).groupby("user_id"):
        seq = group["movie_id"].astype(int).tolist()
        for i, src in enumerate(seq):
            start = max(0, i - window)
            end = min(len(seq), i + window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                dst = seq[j]
                distance = abs(i - j)
                cooc[src][dst] += 1.0 / max(distance, 1)
    return {src: counter.most_common() for src, counter in cooc.items()}


def popularity_candidates(sample: pd.Series, pop_ranked: list[tuple[int, float]], top_k: int) -> list[dict]:
    seen = set(parse_ids(sample.seen_movie_ids))
    rows = []
    rank = 1
    for movie_id, score in pop_ranked:
        if movie_id in seen:
            continue
        rows.append({"candidate_movie_id": movie_id, "retrieval_rank": rank, "retrieval_score": score})
        rank += 1
        if rank > top_k:
            break
    return rows


def random_candidates(sample: pd.Series, movie_ids: list[int], top_k: int, seed: int) -> list[dict]:
    seen = set(parse_ids(sample.seen_movie_ids))
    rng = np.random.default_rng(seed + int(sample.name))
    shuffled = np.array(movie_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    rows = []
    rank = 1
    for movie_id in shuffled.tolist():
        movie_id = int(movie_id)
        if movie_id in seen:
            continue
        rows.append({"candidate_movie_id": movie_id, "retrieval_rank": rank, "retrieval_score": 1.0 / rank})
        rank += 1
        if rank > top_k:
            break
    return rows


def item_item_candidates(
    sample: pd.Series,
    neighbors: dict[int, list[tuple[int, float]]],
    pop_ranked: list[tuple[int, float]],
    top_k: int,
    last_n: int,
    recency_decay: float,
) -> list[dict]:
    seen = set(parse_ids(sample.seen_movie_ids))
    history = parse_ids(sample.model_history_movie_ids)[-last_n:]
    scores: Counter[int] = Counter()
    n = len(history)
    for i, source_movie in enumerate(history):
        weight = recency_decay ** (n - 1 - i)
        for dst, score in neighbors.get(source_movie, [])[:200]:
            if dst not in seen:
                scores[int(dst)] += float(score) * weight
    ranked = scores.most_common()
    rows = []
    used = set()
    rank = 1
    for movie_id, score in ranked:
        rows.append({"candidate_movie_id": int(movie_id), "retrieval_rank": rank, "retrieval_score": float(score)})
        used.add(int(movie_id))
        rank += 1
        if rank > top_k:
            return rows
    for movie_id, score in pop_ranked:
        if movie_id in seen or movie_id in used:
            continue
        rows.append({"candidate_movie_id": movie_id, "retrieval_rank": rank, "retrieval_score": 0.001 * score})
        rank += 1
        if rank > top_k:
            break
    return rows


def emit_rows(samples: pd.DataFrame, split: str, method: str, per_sample_rows: list[tuple[int, pd.Series, list[dict]]]) -> None:
    rows = []
    for sample_id, sample, candidates in per_sample_rows:
        for row in candidates:
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "user_id": int(sample.user_id),
                    "split": split,
                    "target_movie_id": int(sample.target_movie_id),
                    "candidate_movie_id": int(row["candidate_movie_id"]),
                    "retrieval_rank": int(row["retrieval_rank"]),
                    "retrieval_score": float(row["retrieval_score"]),
                    "method": method,
                }
            )
    out = OUT / f"candidates_{method}_{split}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{method} {split}: {out} rows={len(rows)} samples={len(samples)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--methods", nargs="+", default=["popularity", "random", "item_item"], choices=["popularity", "random", "item_item"])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--train_sample_size", type=int, default=0, help="0 means all train samples.")
    parser.add_argument("--train_sample_ids_out", default="", help="Optional CSV path for deterministic train sample ids.")
    parser.add_argument("--item_window", type=int, default=10)
    parser.add_argument("--last_n", type=int, default=10)
    parser.add_argument("--recency_decay", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()

    samples = pd.read_csv(SAMPLES)
    popularity = pd.read_csv(POPULARITY)
    train = pd.read_csv(TRAIN_INTERACTIONS)
    pop_ranked = top_popularity(popularity)
    movie_ids = [movie_id for movie_id, _ in pop_ranked]
    neighbors = build_item_neighbors(train, args.item_window)

    for split in args.splits:
        split_samples = samples[samples.split == split]
        if split == "train" and args.train_sample_size > 0 and len(split_samples) > args.train_sample_size:
            split_samples = split_samples.sample(args.train_sample_size, random_state=args.seed).sort_index()
            ids_path = (
                args.train_sample_ids_out
                if args.train_sample_ids_out
                else str(LLM_DATA / f"train_sample_ids_m{args.top_k}_n{args.train_sample_size}_seed{args.seed}.csv")
            )
            pd.DataFrame({"sample_id": split_samples.index.astype(int)}).to_csv(ids_path, index=False)
            print(f"saved train sample ids: {ids_path} rows={len(split_samples)}")

        pop_rows = [] if "popularity" in args.methods else None
        random_rows = [] if "random" in args.methods else None
        item_rows = [] if "item_item" in args.methods else None
        for sample_id, sample in split_samples.iterrows():
            if pop_rows is not None:
                pop_rows.append((sample_id, sample, popularity_candidates(sample, pop_ranked, args.top_k)))
            if random_rows is not None:
                random_rows.append((sample_id, sample, random_candidates(sample, movie_ids, args.top_k, args.seed)))
            if item_rows is not None:
                item_rows.append(
                    (
                        sample_id,
                        sample,
                        item_item_candidates(sample, neighbors, pop_ranked, args.top_k, args.last_n, args.recency_decay),
                    )
                )
        if pop_rows is not None:
            emit_rows(split_samples, split, "popularity", pop_rows)
        if random_rows is not None:
            emit_rows(split_samples, split, "random", random_rows)
        if item_rows is not None:
            emit_rows(split_samples, split, "item_item", item_rows)


if __name__ == "__main__":
    main()
