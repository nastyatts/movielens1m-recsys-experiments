from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import EMB, MOVIE_METADATA, MOVIE_ORDER, OUT, SAMPLES, parse_ids, recency_mean, write_json
from stage_a_experiment_utils import ensure_dir, eval_candidates, top_popularity


V6_OUT = OUT / "v6_rag_ablation"
TOKEN_RE = re.compile(r"[a-z0-9]+")
TEXT_FIELDS = ["Plot", "Director", "Actors", "Language", "Country", "Runtime", "imdbRating", "Metascore", "Production"]


def clean(value: object) -> str:
    return "" if pd.isna(value) or str(value).strip() in {"", "N/A", "nan"} else str(value).strip()


def item_doc(row: pd.Series) -> str:
    parts = [clean(row.title), f"Genres: {clean(row.genres).replace('|', ', ')}"]
    year = clean(row.get("year"))
    if year:
        parts.append(f"Year: {year}")
    for field in TEXT_FIELDS:
        value = clean(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    return ". ".join(part for part in parts if part)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def load_docs() -> pd.DataFrame:
    movies = pd.read_csv(MOVIE_METADATA).sort_values("movie_id").reset_index(drop=True)
    movies["doc"] = movies.apply(item_doc, axis=1)
    return movies[["movie_id", "doc"]]


def load_dense_embeddings(docs: pd.DataFrame, model_name: str, batch_size: int) -> tuple[np.ndarray, list[int]]:
    if EMB.exists() and MOVIE_ORDER.exists():
        emb = np.load(EMB).astype(np.float32)
        order = np.load(MOVIE_ORDER).astype(int).tolist()
        order_set = set(order)
        if all(int(mid) in order_set for mid in docs.movie_id.astype(int).tolist()):
            pos = {int(mid): i for i, mid in enumerate(order)}
            idx = [pos[int(mid)] for mid in docs.movie_id.astype(int).tolist()]
            return emb[idx], docs.movie_id.astype(int).tolist()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(docs["doc"].tolist(), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return emb.astype(np.float32), docs.movie_id.astype(int).tolist()


class BM25Index:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_lens = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        df = Counter()
        tf_by_term: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            for token, count in counts.items():
                df[token] += 1
                tf_by_term[token].append((doc_idx, int(count)))
        self.postings = {}
        self.idf = {}
        for token, postings in tf_by_term.items():
            docs = np.array([p[0] for p in postings], dtype=np.int32)
            tfs = np.array([p[1] for p in postings], dtype=np.float32)
            self.postings[token] = (docs, tfs)
            self.idf[token] = math.log(1.0 + (self.n_docs - df[token] + 0.5) / (df[token] + 0.5))

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for token, qtf in Counter(query_tokens).items():
            if token not in self.postings:
                continue
            docs, tfs = self.postings[token]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * self.doc_lens[docs] / max(self.avgdl, 1e-6))
            scores[docs] += float(qtf) * self.idf[token] * (tfs * (self.k1 + 1.0) / denom)
        return scores


def minmax_nonseen(scores: np.ndarray, seen_pos: set[int]) -> np.ndarray:
    out = scores.astype(np.float32).copy()
    valid = np.ones(len(out), dtype=bool)
    if seen_pos:
        valid[list(seen_pos)] = False
    valid_scores = out[valid]
    if len(valid_scores) == 0:
        return np.zeros_like(out)
    lo = float(valid_scores.min())
    hi = float(valid_scores.max())
    if hi - lo < 1e-8:
        out[valid] = 0.0
    else:
        out[valid] = (out[valid] - lo) / (hi - lo)
    if seen_pos:
        out[list(seen_pos)] = -1e9
    return out


def rank_from_scores(
    scores: np.ndarray,
    sample: pd.Series,
    split: str,
    sample_id: int,
    movie_ids: list[int],
    movie_pos: dict[int, int],
    pop_ranked: list[tuple[int, float]],
    top_k: int,
    method: str,
) -> list[dict]:
    seen = set(parse_ids(sample.seen_movie_ids))
    masked = scores.copy()
    for movie_id in seen:
        pos = movie_pos.get(movie_id)
        if pos is not None:
            masked[pos] = -1e9
    take = min(len(masked), max(top_k * 4, top_k + len(seen) + 50))
    top_idx = np.argpartition(-masked, take - 1)[:take]
    top_idx = top_idx[np.argsort(-masked[top_idx])]
    rows = []
    used = set()
    for idx in top_idx.tolist():
        movie_id = int(movie_ids[idx])
        if movie_id in seen or movie_id in used:
            continue
        rows.append(
            {
                "sample_id": int(sample_id),
                "user_id": int(sample.user_id),
                "split": split,
                "target_movie_id": int(sample.target_movie_id),
                "candidate_movie_id": movie_id,
                "retrieval_rank": len(rows) + 1,
                "retrieval_score": float(masked[idx]),
                "method": method,
            }
        )
        used.add(movie_id)
        if len(rows) >= top_k:
            break
    for movie_id, pop_score in pop_ranked:
        if len(rows) >= top_k:
            break
        if movie_id in seen or movie_id in used:
            continue
        rows.append(
            {
                "sample_id": int(sample_id),
                "user_id": int(sample.user_id),
                "split": split,
                "target_movie_id": int(sample.target_movie_id),
                "candidate_movie_id": int(movie_id),
                "retrieval_rank": len(rows) + 1,
                "retrieval_score": float(-1e6 + pop_score),
                "method": method,
            }
        )
        used.add(int(movie_id))
    return rows


def query_from_history(sample: pd.Series, docs_by_id: dict[int, str], max_history_len: int) -> str:
    history = parse_ids(sample.model_history_movie_ids)[-max_history_len:]
    return " ".join(docs_by_id.get(movie_id, "") for movie_id in history).strip()


def build_split(
    split: str,
    split_samples: pd.DataFrame,
    docs_by_id: dict[int, str],
    movie_ids: list[int],
    movie_pos: dict[int, int],
    emb: np.ndarray,
    bm25: BM25Index,
    pop_ranked: list[tuple[int, float]],
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    dense_rows: list[dict] = []
    bm25_rows: list[dict] = []
    hybrid_rows: dict[float, list[dict]] = {alpha: [] for alpha in args.alphas}
    for n, (sample_id, sample) in enumerate(split_samples.iterrows(), start=1):
        history = parse_ids(sample.model_history_movie_ids)[-args.max_history_len :]
        hist_pos = [movie_pos[mid] for mid in history if mid in movie_pos]
        if hist_pos:
            dense_scores = emb @ recency_mean(emb[hist_pos]).reshape(-1)
        else:
            dense_scores = np.zeros(len(movie_ids), dtype=np.float32)
        query = query_from_history(sample, docs_by_id, args.max_history_len)
        bm25_scores = bm25.score(tokenize(query))
        seen_pos = {movie_pos[mid] for mid in parse_ids(sample.seen_movie_ids) if mid in movie_pos}
        dense_norm = minmax_nonseen(dense_scores, seen_pos)
        bm25_norm = minmax_nonseen(bm25_scores, seen_pos)
        dense_rows.extend(rank_from_scores(dense_scores, sample, split, int(sample_id), movie_ids, movie_pos, pop_ranked, args.top_k, "dense"))
        bm25_rows.extend(rank_from_scores(bm25_scores, sample, split, int(sample_id), movie_ids, movie_pos, pop_ranked, args.top_k, "bm25"))
        for alpha in args.alphas:
            scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm
            method = f"hybrid_alpha{str(alpha).replace('.', 'p')}"
            hybrid_rows[alpha].extend(rank_from_scores(scores, sample, split, int(sample_id), movie_ids, movie_pos, pop_ranked, args.top_k, method))
        if n % 1000 == 0:
            print(f"rag {split}: {n}/{len(split_samples)}")
    frames = {
        "dense": pd.DataFrame(dense_rows),
        "bm25": pd.DataFrame(bm25_rows),
    }
    for alpha, rows in hybrid_rows.items():
        frames[f"hybrid_alpha{str(alpha).replace('.', 'p')}"] = pd.DataFrame(rows)
    metrics = {name: eval_candidates(frame, split_samples) for name, frame in frames.items()}
    return frames, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--max_history_len", type=int, default=5)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--top_n_test", type=int, default=5)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(V6_OUT)
    args.alphas = [float(x) for x in str(args.alphas).split(",") if x]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    docs = load_docs()
    docs_by_id = dict(zip(docs.movie_id.astype(int), docs.doc.astype(str)))
    movie_ids = docs.movie_id.astype(int).tolist()
    movie_pos = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    emb, emb_movie_ids = load_dense_embeddings(docs, args.model_name, args.batch_size)
    if emb_movie_ids != movie_ids:
        pos = {movie_id: i for i, movie_id in enumerate(emb_movie_ids)}
        emb = emb[[pos[mid] for mid in movie_ids]]
    tokenized_docs = [tokenize(doc) for doc in docs.doc.astype(str).tolist()]
    bm25 = BM25Index(tokenized_docs)
    pop_ranked = top_popularity()
    samples = pd.read_csv(SAMPLES)

    print(f"rag docs={len(docs)} splits={splits} top_k={args.top_k} max_history_len={args.max_history_len} alphas={args.alphas}")
    all_metrics = {}
    selected = set()
    val_rows = []
    test_rows = []

    for split in splits:
        split_samples = samples[samples.split == split]
        if args.sample_limit > 0:
            split_samples = split_samples.head(args.sample_limit)
        frames, metrics = build_split(split, split_samples, docs_by_id, movie_ids, movie_pos, emb, bm25, pop_ranked, args)
        all_metrics[split] = metrics
        if split == "val":
            for name, frame in frames.items():
                out = V6_OUT / f"candidates_{name}_val.csv"
                frame.to_csv(out, index=False)
                row = {"version": "v6_rag_ablation", "stage": "rag_ablation", "split": "val", "config": name, **metrics[name]}
                val_rows.append(row)
                print(f"saved {out} HR@1={metrics[name]['HR@1']:.6f} NDCG@30={metrics[name]['NDCG@30']:.6f}")
            val_df = pd.DataFrame(val_rows)
            for metric in ["HR@1", "NDCG@30", "TargetIn@30", "TargetIn@50"]:
                top = val_df.sort_values([metric, "HR@1", "NDCG@30"], ascending=False).head(args.top_n_test)
                selected.update(top["config"].tolist())
            selected.update(["dense", "bm25"])
        elif split == "test":
            for name, frame in frames.items():
                if name not in selected and name.startswith("hybrid_"):
                    continue
                out = V6_OUT / f"candidates_{name}_test.csv"
                frame.to_csv(out, index=False)
                row = {"version": "v6_rag_ablation", "stage": "rag_ablation", "split": "test", "config": name, **metrics[name]}
                test_rows.append(row)
                print(f"saved {out} HR@1={metrics[name]['HR@1']:.6f} NDCG@30={metrics[name]['NDCG@30']:.6f}")

    pd.DataFrame(val_rows).to_csv(V6_OUT / "rag_val_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(V6_OUT / "rag_test_metrics.csv", index=False)
    serializable_metrics = json.loads(json.dumps(all_metrics, default=float))
    write_json(
        V6_OUT / "best_rag_configs.json",
        {
            "selection_split": "val",
            "selected_for_test": sorted(selected),
            "top_n_test": args.top_n_test,
            "alphas": args.alphas,
            "metrics": serializable_metrics,
        },
    )
    (V6_OUT / "run_commands.txt").write_text(
        "SMOKE:\n"
        "python scripts/28_rag_ablation.py --splits val,test --top_k 200 --max_history_len 5 --sample_limit 500 --alphas 0,0.5,1 --top_n_test 3\n\n"
        "python scripts/19_collect_experiments_summary.py\n\n"
        "NORMAL:\n"
        "python scripts/28_rag_ablation.py --splits val,test --top_k 200 --max_history_len 5 --alphas 0,0.25,0.5,0.75,1 --top_n_test 5\n\n"
        "python scripts/19_collect_experiments_summary.py\n",
        encoding="utf-8",
    )
    print(f"wrote metrics: {V6_OUT / 'rag_val_metrics.csv'} {V6_OUT / 'rag_test_metrics.csv'}")
    print(f"selected_for_test={sorted(selected)}")


if __name__ == "__main__":
    main()
