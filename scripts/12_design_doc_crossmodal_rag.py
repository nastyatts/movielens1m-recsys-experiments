from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import EMB, MOVIE_METADATA, MOVIE_ORDER, OUT, POPULARITY, SAMPLES, TRAIN_INTERACTIONS, USERS_CSV, genres, parse_ids, write_json
from stage_a_experiment_utils import ensure_dir, eval_candidates, top_popularity


V10_OUT = OUT / "v10_sasrec_retrieval"
V14_OUT = OUT / "v14_design_doc_pipeline"


def split_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def beta_token(value: float) -> str:
    return str(value).replace(".", "p")


def load_text_embeddings() -> tuple[list[int], np.ndarray]:
    if not EMB.exists() or not MOVIE_ORDER.exists():
        raise FileNotFoundError(f"Missing frozen text embeddings: {EMB} and/or {MOVIE_ORDER}")
    emb = np.load(EMB).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    movie_ids = np.load(MOVIE_ORDER).astype(int).tolist()
    return movie_ids, emb


def build_maps(samples: pd.DataFrame, movie_ids: list[int]) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    item_ids = set(int(x) for x in movie_ids)
    item_ids.update(samples.target_movie_id.astype(int).tolist())
    for value in samples.model_history_movie_ids.tolist():
        item_ids.update(parse_ids(value))
    item_to_idx = {movie_id: idx + 1 for idx, movie_id in enumerate(sorted(item_ids))}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    user_ids = sorted(samples.user_id.astype(int).unique().tolist())
    user_to_idx = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
    return item_to_idx, idx_to_item, user_to_idx


def feature_matrices(item_to_idx: dict[int, int], movie_ids: list[int], text_emb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, dict[int, set[str]], int]:
    text_dim = int(text_emb.shape[1])
    text_matrix = np.zeros((len(item_to_idx) + 1, text_dim), dtype=np.float32)
    movie_pos = {int(movie_id): i for i, movie_id in enumerate(movie_ids)}
    for movie_id, idx in item_to_idx.items():
        pos = movie_pos.get(int(movie_id))
        if pos is not None:
            text_matrix[idx] = text_emb[pos]

    meta = pd.read_csv(MOVIE_METADATA)
    all_genres = sorted({genre for value in meta.genres.tolist() for genre in genres(value)})
    genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
    genre_dim = max(len(genre_to_idx), 1)
    genre_matrix = np.zeros((len(item_to_idx) + 1, genre_dim), dtype=np.float32)
    genre_map: dict[int, set[str]] = {}
    for row in meta.itertuples(index=False):
        movie_id = int(row.movie_id)
        gs = genres(row.genres)
        genre_map[movie_id] = gs
        item_idx = item_to_idx.get(movie_id)
        if item_idx is not None:
            for genre in gs:
                genre_matrix[item_idx, genre_to_idx[genre]] = 1.0
    return torch.tensor(text_matrix), torch.tensor(genre_matrix), genre_map, genre_dim


def user_feature_maps() -> tuple[pd.DataFrame, dict[str, int], dict[int, int], dict[int, int]]:
    users = pd.read_csv(USERS_CSV)
    gender_to_idx = {value: idx + 1 for idx, value in enumerate(sorted(users.gender.astype(str).unique()))}
    age_to_idx = {int(value): idx + 1 for idx, value in enumerate(sorted(users.age.astype(int).unique()))}
    occupation_to_idx = {int(value): idx + 1 for idx, value in enumerate(sorted(users.occupation.astype(int).unique()))}
    return users.set_index("user_id"), gender_to_idx, age_to_idx, occupation_to_idx


def popularity_maps() -> tuple[dict[int, float], list[tuple[int, float]]]:
    pop = pd.read_csv(POPULARITY)
    max_pop = max(float(pop.train_popularity.max()), 1.0)
    pop_norm = {int(row.movie_id): float(np.log1p(row.train_popularity) / np.log1p(max_pop)) for row in pop.itertuples(index=False)}
    return pop_norm, top_popularity()


def history_stats(sample: pd.Series, pop_norm: dict[int, float]) -> np.ndarray:
    hist = parse_ids(sample.model_history_movie_ids)
    if not hist:
        return np.array([0.0, 0.0], dtype=np.float32)
    pops = [pop_norm.get(movie_id, 0.0) for movie_id in hist]
    return np.array([min(len(hist), 200) / 200.0, float(np.mean(pops))], dtype=np.float32)


class QueryDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        item_to_idx: dict[int, int],
        user_to_idx: dict[int, int],
        users: pd.DataFrame,
        gender_to_idx: dict[str, int],
        age_to_idx: dict[int, int],
        occupation_to_idx: dict[int, int],
        pop_norm: dict[int, float],
        max_history_len: int,
    ):
        self.rows = []
        for sample_id, sample in samples.iterrows():
            user_id = int(sample.user_id)
            user_row = users.loc[user_id] if user_id in users.index else None
            history = [item_to_idx[movie_id] for movie_id in parse_ids(sample.model_history_movie_ids)[-max_history_len:] if movie_id in item_to_idx]
            hist_arr = np.zeros(max_history_len, dtype=np.int64)
            hist_mask = np.zeros(max_history_len, dtype=np.bool_)
            if history:
                hist_arr[: len(history)] = np.array(history, dtype=np.int64)
                hist_mask[: len(history)] = True
            self.rows.append(
                {
                    "sample_id": int(sample_id),
                    "user_id_raw": user_id,
                    "target_movie_id": int(sample.target_movie_id),
                    "seen": set(parse_ids(sample.seen_movie_ids)),
                    "user_idx": user_to_idx.get(user_id, 0),
                    "gender_idx": gender_to_idx.get(str(user_row.gender), 0) if user_row is not None else 0,
                    "age_idx": age_to_idx.get(int(user_row.age), 0) if user_row is not None else 0,
                    "occupation_idx": occupation_to_idx.get(int(user_row.occupation), 0) if user_row is not None else 0,
                    "history_ids": hist_arr,
                    "history_mask": hist_mask,
                    "numeric": history_stats(sample, pop_norm),
                    "sample": sample,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "sample_id": row["sample_id"],
            "user_idx": torch.tensor(row["user_idx"], dtype=torch.long),
            "gender_idx": torch.tensor(row["gender_idx"], dtype=torch.long),
            "age_idx": torch.tensor(row["age_idx"], dtype=torch.long),
            "occupation_idx": torch.tensor(row["occupation_idx"], dtype=torch.long),
            "history_ids": torch.tensor(row["history_ids"], dtype=torch.long),
            "history_mask": torch.tensor(row["history_mask"], dtype=torch.bool),
            "numeric": torch.tensor(row["numeric"], dtype=torch.float32),
        }


class TrainCandidateDataset(Dataset):
    def __init__(
        self,
        query_ds: QueryDataset,
        candidates: dict[int, list[int]],
        item_to_idx: dict[int, int],
        pop_norm: dict[int, float],
        max_candidates: int,
    ):
        self.query_ds = query_ds
        self.rows = []
        for idx, row in enumerate(query_ds.rows):
            cand = candidates.get(row["sample_id"], [])[:max_candidates]
            target = row["target_movie_id"]
            if target not in cand:
                continue
            cand_idx = [item_to_idx.get(movie_id, 0) for movie_id in cand]
            arr = np.zeros(max_candidates, dtype=np.int64)
            mask = np.zeros(max_candidates, dtype=np.bool_)
            arr[: len(cand_idx)] = np.array(cand_idx, dtype=np.int64)
            mask[: len(cand_idx)] = True
            self.rows.append((idx, arr, mask, cand.index(target), pop_norm.get(int(target), 0.0)))
        if not self.rows:
            raise ValueError("No target-present train candidates for cross-modal design doc pipeline.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        query_idx, cand_arr, cand_mask, label, target_pop = self.rows[idx]
        base = self.query_ds[query_idx]
        base.update(
            {
                "candidate_ids": torch.tensor(cand_arr, dtype=torch.long),
                "candidate_mask": torch.tensor(cand_mask, dtype=torch.bool),
                "label": torch.tensor(label, dtype=torch.long),
                "target_popularity": torch.tensor(target_pop, dtype=torch.float32),
            }
        )
        return base


def collate(batch: list[dict]) -> dict:
    keys = ["user_idx", "gender_idx", "age_idx", "occupation_idx", "history_ids", "history_mask", "numeric"]
    out = {"sample_id": [row["sample_id"] for row in batch]}
    for key in keys:
        out[key] = torch.stack([row[key] for row in batch])
    if "candidate_ids" in batch[0]:
        out["candidate_ids"] = torch.stack([row["candidate_ids"] for row in batch])
        out["candidate_mask"] = torch.stack([row["candidate_mask"] for row in batch])
        out["label"] = torch.stack([row["label"] for row in batch])
        out["target_popularity"] = torch.stack([row["target_popularity"] for row in batch])
    return out


class DesignDocCrossModalModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_users: int,
        num_genders: int,
        num_ages: int,
        num_occupations: int,
        text_matrix: torch.Tensor,
        genre_matrix: torch.Tensor,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(num_genders + 1, embedding_dim, padding_idx=0)
        self.age_emb = nn.Embedding(num_ages + 1, embedding_dim, padding_idx=0)
        self.occupation_emb = nn.Embedding(num_occupations + 1, embedding_dim, padding_idx=0)
        self.text_emb = nn.Embedding.from_pretrained(text_matrix.float(), freeze=True, padding_idx=0)
        self.genre_emb = nn.Embedding.from_pretrained(genre_matrix.float(), freeze=True, padding_idx=0)
        self.text_to_model = nn.Linear(text_matrix.shape[1], embedding_dim)
        self.genre_to_model = nn.Linear(genre_matrix.shape[1], embedding_dim)
        self.numeric_mlp = nn.Sequential(nn.Linear(2, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.to_text = nn.Linear(embedding_dim, text_matrix.shape[1])
        self.to_genre = nn.Linear(embedding_dim, genre_matrix.shape[1])
        self.empty_history_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.empty_history_token, std=0.02)

    def encode_query(self, batch: dict) -> torch.Tensor:
        hist_ids = batch["history_ids"]
        hist_mask = batch["history_mask"]
        item_tokens = self.item_emb(hist_ids) + self.text_to_model(self.text_emb(hist_ids)) + self.genre_to_model(self.genre_emb(hist_ids))
        n = hist_ids.shape[1]
        weights = torch.linspace(0.25, 1.0, n, device=hist_ids.device).view(1, n, 1)
        masked = item_tokens * hist_mask.unsqueeze(-1) * weights
        pooled = masked.sum(dim=1) / torch.clamp((hist_mask.unsqueeze(-1) * weights).sum(dim=1), min=1e-6)
        user = (
            self.user_emb(batch["user_idx"])
            + self.gender_emb(batch["gender_idx"])
            + self.age_emb(batch["age_idx"])
            + self.occupation_emb(batch["occupation_idx"])
            + self.numeric_mlp(batch["numeric"])
            + pooled
        )
        effective_mask = hist_mask.clone()
        attn_tokens = item_tokens.clone()
        empty = ~effective_mask.any(dim=1)
        if empty.any():
            attn_tokens[empty] = 0.0
            attn_tokens[empty, 0:1, :] = self.empty_history_token
            effective_mask[empty, 0] = True
        attended, _ = self.attn(user.unsqueeze(1), attn_tokens, attn_tokens, key_padding_mask=~effective_mask)
        fused = self.norm(user + self.dropout(attended.squeeze(1)))
        q = self.to_text(fused)
        q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-8)
        return q

    def encode_fused(self, batch: dict) -> torch.Tensor:
        hist_ids = batch["history_ids"]
        hist_mask = batch["history_mask"]
        item_tokens = self.item_emb(hist_ids) + self.text_to_model(self.text_emb(hist_ids)) + self.genre_to_model(self.genre_emb(hist_ids))
        n = hist_ids.shape[1]
        weights = torch.linspace(0.25, 1.0, n, device=hist_ids.device).view(1, n, 1)
        masked = item_tokens * hist_mask.unsqueeze(-1) * weights
        pooled = masked.sum(dim=1) / torch.clamp((hist_mask.unsqueeze(-1) * weights).sum(dim=1), min=1e-6)
        user = (
            self.user_emb(batch["user_idx"])
            + self.gender_emb(batch["gender_idx"])
            + self.age_emb(batch["age_idx"])
            + self.occupation_emb(batch["occupation_idx"])
            + self.numeric_mlp(batch["numeric"])
            + pooled
        )
        effective_mask = hist_mask.clone()
        attn_tokens = item_tokens.clone()
        empty = ~effective_mask.any(dim=1)
        if empty.any():
            attn_tokens[empty] = 0.0
            attn_tokens[empty, 0:1, :] = self.empty_history_token
            effective_mask[empty, 0] = True
        attended, _ = self.attn(user.unsqueeze(1), attn_tokens, attn_tokens, key_padding_mask=~effective_mask)
        return self.norm(user + self.dropout(attended.squeeze(1)))

    def score_candidates(self, batch: dict, lambda_genre_train: float = 0.1) -> torch.Tensor:
        fused = self.encode_fused(batch)
        q = self.to_text(fused)
        q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-8)
        cand_text = self.text_emb(batch["candidate_ids"])
        text_scores = torch.einsum("bd,bkd->bk", q, cand_text)
        q_genre = self.to_genre(fused)
        cand_genre = self.genre_emb(batch["candidate_ids"])
        genre_scores = torch.einsum("bd,bkd->bk", q_genre, cand_genre)
        scores = text_scores + lambda_genre_train * genre_scores
        scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
        return scores


def initial_text_candidates(
    q_text: np.ndarray,
    sample: pd.Series,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    text_matrix: np.ndarray,
    pop_ranked: list[tuple[int, float]],
    top_k: int,
    inject_target: bool,
) -> list[int]:
    scores = text_matrix @ q_text
    scores[0] = -1e9
    seen = set(parse_ids(sample.seen_movie_ids))
    for movie_id in seen:
        idx = item_to_idx.get(movie_id)
        if idx is not None:
            scores[idx] = -1e9
    take = min(len(scores) - 1, max(top_k * 4, top_k + len(seen) + 50))
    idxs = np.argpartition(-scores, take)[:take]
    idxs = idxs[np.argsort(-scores[idxs])]
    out = []
    used = set()
    for idx in idxs.tolist():
        movie_id = idx_to_item.get(int(idx))
        if movie_id is None or movie_id in seen or movie_id in used:
            continue
        out.append(movie_id)
        used.add(movie_id)
        if len(out) >= top_k:
            break
    for movie_id, _ in pop_ranked:
        if len(out) >= top_k:
            break
        if movie_id not in seen and movie_id not in used:
            out.append(int(movie_id))
            used.add(int(movie_id))
    target = int(sample.target_movie_id)
    if inject_target and target not in out:
        if len(out) >= top_k:
            out = out[: top_k - 1]
        out.append(target)
    return out[:top_k]


def history_text_query(sample: pd.Series, item_to_idx: dict[int, int], text_matrix: np.ndarray) -> np.ndarray:
    hist = [item_to_idx[mid] for mid in parse_ids(sample.model_history_movie_ids) if mid in item_to_idx]
    if not hist:
        return np.zeros(text_matrix.shape[1], dtype=np.float32)
    hist = hist[-50:]
    weights = np.array([0.85 ** (len(hist) - 1 - i) for i in range(len(hist))], dtype=np.float32)
    weights /= max(float(weights.sum()), 1e-8)
    q = (text_matrix[hist] * weights[:, None]).sum(axis=0)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q.astype(np.float32)


def build_train_candidate_dict(samples: pd.DataFrame, item_to_idx: dict[int, int], idx_to_item: dict[int, int], text_matrix: np.ndarray, top_k: int) -> dict[int, list[int]]:
    pop_ranked = top_popularity()
    out = {}
    for n, (sample_id, sample) in enumerate(samples.iterrows(), start=1):
        q = history_text_query(sample, item_to_idx, text_matrix)
        out[int(sample_id)] = initial_text_candidates(q, sample, item_to_idx, idx_to_item, text_matrix, pop_ranked, top_k, inject_target=True)
        if n % 10000 == 0:
            print(f"train initial candidates {n}/{len(samples)}", flush=True)
    return out


def genre_overlap(movie_id: int, sample: pd.Series, genre_map: dict[int, set[str]]) -> float:
    hist_genres = set()
    for mid in parse_ids(sample.model_history_movie_ids):
        hist_genres.update(genre_map.get(mid, set()))
    cand = genre_map.get(int(movie_id), set())
    if not hist_genres and not cand:
        return 0.0
    return float(len(hist_genres & cand) / max(len(hist_genres | cand), 1))


def sasrec_lookup(run_name: str, split: str) -> dict[int, dict[int, tuple[float, int]]]:
    path = V10_OUT / "runs" / run_name / f"candidates_sasrec_{split}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    lookup: dict[int, dict[int, tuple[float, int]]] = {}
    for row in df.itertuples(index=False):
        lookup.setdefault(int(row.sample_id), {})[int(row.candidate_movie_id)] = (float(row.retrieval_score), int(row.retrieval_rank))
    return lookup


TOKEN_RE = re.compile(r"[a-z0-9]+")
BM25F_FIELDS = {
    "title": 3.0,
    "genres": 2.5,
    "Plot": 1.0,
    "Director": 1.8,
    "Actors": 1.5,
    "Language": 0.4,
    "Country": 0.4,
    "Runtime": 0.2,
    "imdbRating": 0.3,
    "Metascore": 0.3,
    "Production": 0.8,
}
SOURCE_WEIGHTS = {
    "dense_user": 1.0,
    "dense_history": 0.8,
    "bm25f_metadata": 0.9,
    "structured_metadata": 0.7,
    "cowatch_context": 1.1,
    "popularity_fallback": 0.15,
}


def tokenize(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return TOKEN_RE.findall(str(value).lower())


def first_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


class StrongRagIndex:
    def __init__(
        self,
        item_to_idx: dict[int, int],
        idx_to_item: dict[int, int],
        text_matrix: np.ndarray,
        genre_matrix: np.ndarray,
        pop_norm: dict[int, float],
    ):
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.text_matrix = text_matrix
        self.genre_matrix = genre_matrix
        self.pop_norm = pop_norm
        self.meta = pd.read_csv(MOVIE_METADATA).set_index("movie_id", drop=False)
        self.movie_ids = [movie_id for movie_id in item_to_idx if movie_id in self.meta.index]
        self.movie_id_set = set(self.movie_ids)
        self.pop_ranked = top_popularity()
        self.field_counters: dict[int, dict[str, Counter[str]]] = {}
        self.field_lengths: dict[str, list[int]] = defaultdict(list)
        self.postings: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self._build_bm25f()
        self.cowatch = self._build_cowatch_neighbors()

    def _build_bm25f(self) -> None:
        doc_freq: Counter[str] = Counter()
        for row in self.meta.itertuples(index=False):
            movie_id = int(row.movie_id)
            field_map: dict[str, Counter[str]] = {}
            doc_terms = set()
            for field in BM25F_FIELDS:
                tokens = tokenize(getattr(row, field, ""))
                counter = Counter(tokens)
                field_map[field] = counter
                self.field_lengths[field].append(sum(counter.values()))
                doc_terms.update(counter.keys())
            self.field_counters[movie_id] = field_map
            for token in doc_terms:
                doc_freq[token] += 1
        num_docs = max(len(self.field_counters), 1)
        self.idf = {token: math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5)) for token, df in doc_freq.items()}
        avg_len = {field: max(float(np.mean(lengths)), 1.0) for field, lengths in self.field_lengths.items()}
        for movie_id, field_map in self.field_counters.items():
            for field, counter in field_map.items():
                length = max(sum(counter.values()), 1)
                for token, tf in counter.items():
                    self.postings[token].append((movie_id, field, int(tf), int(length)))
        self.avg_field_len = avg_len

    def _build_cowatch_neighbors(self, window: int = 50, max_neighbors: int = 500) -> dict[int, list[tuple[int, float]]]:
        if not TRAIN_INTERACTIONS.exists():
            return {}
        train = pd.read_csv(TRAIN_INTERACTIONS)
        cooc: dict[int, Counter[int]] = defaultdict(Counter)
        for _, group in train.sort_values(["user_id", "timestamp", "movie_id"]).groupby("user_id", sort=False):
            seq = group.movie_id.astype(int).tolist()
            for i, src in enumerate(seq):
                lo = max(0, i - window)
                hi = min(len(seq), i + window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    dst = int(seq[j])
                    cooc[int(src)][dst] += 1.0 / (1.0 + abs(i - j))
        return {movie_id: counter.most_common(max_neighbors) for movie_id, counter in cooc.items()}

    def movie_tokens(self, movie_id: int) -> Counter[str]:
        out: Counter[str] = Counter()
        for counter in self.field_counters.get(int(movie_id), {}).values():
            out.update(counter)
        return out

    def bm25f_scores(self, query_terms: Counter[str]) -> Counter[int]:
        scores: Counter[int] = Counter()
        k1 = 1.5
        b = 0.75
        for token, qtf in query_terms.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            for movie_id, field, tf, length in self.postings.get(token, []):
                weight = BM25F_FIELDS[field]
                norm = 1.0 - b + b * (length / self.avg_field_len[field])
                part = idf * weight * ((tf * (k1 + 1.0)) / (tf + k1 * norm))
                scores[int(movie_id)] += float(qtf) * float(part)
        return scores


def top_ranked(scores: dict[int, float] | Counter[int], seen: set[int], top_k: int) -> list[tuple[int, float]]:
    ranked = sorted(
        ((int(movie_id), float(score)) for movie_id, score in scores.items() if int(movie_id) not in seen),
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[:top_k]


def dense_history_scores(sample: pd.Series, index: StrongRagIndex, last_n: int = 10, recency_decay: float = 0.85) -> Counter[int]:
    hist = [movie_id for movie_id in parse_ids(sample.model_history_movie_ids)[-last_n:] if movie_id in index.item_to_idx]
    scores: Counter[int] = Counter()
    n = len(hist)
    if not hist:
        return scores
    all_vecs = index.text_matrix
    for i, movie_id in enumerate(hist):
        item_idx = index.item_to_idx[movie_id]
        weight = recency_decay ** (n - 1 - i)
        sims = all_vecs @ all_vecs[item_idx]
        take = min(len(sims) - 1, 500)
        idxs = np.argpartition(-sims, take)[:take]
        for idx in idxs.tolist():
            cand = index.idx_to_item.get(int(idx))
            if cand is not None and cand != movie_id:
                scores[int(cand)] += float(weight * sims[idx])
    return scores


def bm25_query_terms(sample: pd.Series, index: StrongRagIndex, last_n: int = 5) -> Counter[str]:
    terms: Counter[str] = Counter()
    hist = parse_ids(sample.model_history_movie_ids)[-last_n:]
    n = len(hist)
    for i, movie_id in enumerate(hist):
        weight = max(1, int(round(3 * (0.85 ** (n - 1 - i)))))
        for token, count in index.movie_tokens(movie_id).items():
            terms[token] += int(count) * weight
    return terms


def structured_score(movie_id: int, sample: pd.Series, index: StrongRagIndex) -> float:
    if movie_id not in index.meta.index:
        return 0.0
    cand = index.meta.loc[int(movie_id)]
    hist_ids = [mid for mid in parse_ids(sample.model_history_movie_ids)[-20:] if mid in index.meta.index]
    if not hist_ids:
        return 0.0
    hist = index.meta.loc[hist_ids]
    hist_genres = set()
    for value in hist.genres.tolist():
        hist_genres.update(genres(value))
    cand_genres = genres(cand.genres)
    genre = len(hist_genres & cand_genres) / max(len(hist_genres | cand_genres), 1) if hist_genres or cand_genres else 0.0
    directors = {str(x).strip().lower() for x in hist.Director.fillna("").tolist() if str(x).strip()}
    director = 1.0 if str(cand.Director).strip().lower() in directors and str(cand.Director).strip() else 0.0
    hist_actors = set()
    for value in hist.Actors.fillna("").tolist():
        hist_actors.update(x.strip().lower() for x in str(value).split(",") if x.strip())
    cand_actors = {x.strip().lower() for x in str(cand.Actors).split(",") if x.strip()}
    actor = len(hist_actors & cand_actors) / max(len(cand_actors), 1) if cand_actors else 0.0
    languages = {str(x).strip().lower() for x in hist.Language.fillna("").tolist() if str(x).strip()}
    countries = {str(x).strip().lower() for x in hist.Country.fillna("").tolist() if str(x).strip()}
    language = 1.0 if str(cand.Language).strip().lower() in languages and str(cand.Language).strip() else 0.0
    country = 1.0 if str(cand.Country).strip().lower() in countries and str(cand.Country).strip() else 0.0
    years = [float(x) for x in hist.year.dropna().tolist()]
    year = 0.0
    if years and not pd.isna(cand.year):
        year = max(0.0, 1.0 - abs(float(cand.year) - float(np.mean(years))) / 30.0)
    rating_vals = [first_float(x) for x in hist.imdbRating.tolist()]
    rating_vals = [x for x in rating_vals if x is not None]
    cand_rating = first_float(cand.imdbRating)
    rating = max(0.0, 1.0 - abs(float(cand_rating) - float(np.mean(rating_vals))) / 5.0) if rating_vals and cand_rating is not None else 0.0
    meta_vals = [first_float(x) for x in hist.Metascore.tolist()]
    meta_vals = [x for x in meta_vals if x is not None]
    cand_meta = first_float(cand.Metascore)
    metascore = max(0.0, 1.0 - abs(float(cand_meta) - float(np.mean(meta_vals))) / 50.0) if meta_vals and cand_meta is not None else 0.0
    return float(0.30 * genre + 0.15 * director + 0.15 * actor + 0.08 * language + 0.08 * country + 0.10 * year + 0.07 * rating + 0.07 * metascore)


def cowatch_scores(sample: pd.Series, index: StrongRagIndex, last_n: int = 10, recency_decay: float = 0.85) -> Counter[int]:
    hist = parse_ids(sample.model_history_movie_ids)[-last_n:]
    scores: Counter[int] = Counter()
    n = len(hist)
    for i, source in enumerate(hist):
        weight = recency_decay ** (n - 1 - i)
        for dst, score in index.cowatch.get(int(source), []):
            scores[int(dst)] += float(weight * score)
    return scores


def normalize_feature(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return {movie_id: 0.0 for movie_id in values}
    return {movie_id: float((score - lo) / (hi - lo)) for movie_id, score in values.items()}


def build_strong_pool(
    sample_id: int,
    sample: pd.Series,
    q_text: np.ndarray,
    q_genre: np.ndarray,
    index: StrongRagIndex,
    args: argparse.Namespace,
    genre_matrix_np: np.ndarray,
    split: str,
) -> list[dict]:
    seen = set(parse_ids(sample.seen_movie_ids))
    source_rankings: dict[str, list[tuple[int, float]]] = {}
    dense_scores = {movie_id: float(np.dot(q_text, index.text_matrix[item_idx])) for movie_id, item_idx in index.item_to_idx.items()}
    source_rankings["dense_user"] = top_ranked(dense_scores, seen, args.max_candidates)
    source_rankings["dense_history"] = top_ranked(dense_history_scores(sample, index), seen, args.max_candidates)
    source_rankings["bm25f_metadata"] = top_ranked(index.bm25f_scores(bm25_query_terms(sample, index)), seen, args.max_candidates)
    cowatch_score_map = cowatch_scores(sample, index)
    structured_pool = {
        int(movie_id)
        for source, ranking in source_rankings.items()
        for movie_id, _ in ranking
        if source in {"dense_user", "dense_history", "bm25f_metadata"}
    }
    structured_pool.update(int(movie_id) for movie_id, _ in top_ranked(cowatch_score_map, seen, args.max_candidates))
    structured_pool.update(int(movie_id) for movie_id, _ in index.pop_ranked[: args.max_candidates])
    structured_scores = {movie_id: structured_score(movie_id, sample, index) for movie_id in structured_pool if movie_id not in seen}
    source_rankings["structured_metadata"] = top_ranked(structured_scores, seen, args.max_candidates)
    source_rankings["cowatch_context"] = top_ranked(cowatch_score_map, seen, args.max_candidates)
    source_rankings["popularity_fallback"] = [(movie_id, score) for movie_id, score in index.pop_ranked if movie_id not in seen][: args.max_candidates]

    rrf_scores: Counter[int] = Counter()
    raw_features: dict[str, dict[int, float]] = {source: {} for source in SOURCE_WEIGHTS}
    source_masks: dict[int, set[str]] = defaultdict(set)
    for source, ranking in source_rankings.items():
        for rank, (movie_id, score) in enumerate(ranking, start=1):
            rrf_scores[int(movie_id)] += SOURCE_WEIGHTS[source] / (60.0 + rank)
            raw_features[source][int(movie_id)] = float(score)
            source_masks[int(movie_id)].add(source)
    for movie_id, score in index.pop_ranked:
        if len(rrf_scores) >= args.max_candidates * 3:
            break
        if movie_id not in seen and movie_id not in rrf_scores:
            rrf_scores[int(movie_id)] += SOURCE_WEIGHTS["popularity_fallback"] / (60.0 + len(rrf_scores) + 1)
            raw_features["popularity_fallback"][int(movie_id)] = float(score)
            source_masks[int(movie_id)].add("popularity_fallback")

    candidate_ids = list(rrf_scores.keys())
    feature_values = {
        "dense_user_score": {mid: raw_features["dense_user"].get(mid, 0.0) for mid in candidate_ids},
        "dense_history_score": {mid: raw_features["dense_history"].get(mid, 0.0) for mid in candidate_ids},
        "bm25f_score": {mid: raw_features["bm25f_metadata"].get(mid, 0.0) for mid in candidate_ids},
        "structured_score": {mid: raw_features["structured_metadata"].get(mid, structured_scores.get(mid, 0.0)) for mid in candidate_ids},
        "cowatch_score": {mid: raw_features["cowatch_context"].get(mid, 0.0) for mid in candidate_ids},
        "popularity_score": {mid: index.pop_norm.get(mid, 0.0) for mid in candidate_ids},
        "rrf_score": {mid: float(rrf_scores[mid]) for mid in candidate_ids},
        "crossmodal_genre_score": {mid: float(np.dot(q_genre, genre_matrix_np[index.item_to_idx.get(mid, 0)])) for mid in candidate_ids},
    }
    normalized = {name: normalize_feature(values) for name, values in feature_values.items()}
    rows = []
    for movie_id in candidate_ids:
        relevance = (
            0.28 * normalized["rrf_score"].get(movie_id, 0.0)
            + 0.18 * normalized["dense_user_score"].get(movie_id, 0.0)
            + 0.12 * normalized["dense_history_score"].get(movie_id, 0.0)
            + 0.15 * normalized["bm25f_score"].get(movie_id, 0.0)
            + 0.12 * normalized["structured_score"].get(movie_id, 0.0)
            + 0.10 * normalized["cowatch_score"].get(movie_id, 0.0)
            + 0.05 * normalized["crossmodal_genre_score"].get(movie_id, 0.0)
        )
        pop_norm_score = normalized["popularity_score"].get(movie_id, 0.0)
        rows.append(
            {
                "sample_id": int(sample_id),
                "user_id": int(sample.user_id),
                "split": split,
                "target_movie_id": int(sample.target_movie_id),
                "candidate_movie_id": int(movie_id),
                "dense_user_score": feature_values["dense_user_score"].get(movie_id, 0.0),
                "dense_history_score": feature_values["dense_history_score"].get(movie_id, 0.0),
                "bm25f_score": feature_values["bm25f_score"].get(movie_id, 0.0),
                "structured_score": feature_values["structured_score"].get(movie_id, 0.0),
                "cowatch_score": feature_values["cowatch_score"].get(movie_id, 0.0),
                "popularity_score": feature_values["popularity_score"].get(movie_id, 0.0),
                "rrf_score": feature_values["rrf_score"].get(movie_id, 0.0),
                "crossmodal_genre_score": feature_values["crossmodal_genre_score"].get(movie_id, 0.0),
                "relevance_score": float(relevance),
                "debias_score": float(relevance - args.lambda_pop * pop_norm_score),
                "source_mask": "|".join(sorted(source_masks.get(movie_id, set()))),
            }
        )
    return rows


def frame_from_pool_rows(rows: list[dict], method: str, score_col: str, top_k: int) -> pd.DataFrame:
    cols = [
        "sample_id",
        "user_id",
        "split",
        "target_movie_id",
        "candidate_movie_id",
        "retrieval_rank",
        "retrieval_score",
        "method",
        "dense_user_score",
        "dense_history_score",
        "bm25f_score",
        "structured_score",
        "cowatch_score",
        "popularity_score",
        "rrf_score",
        "crossmodal_genre_score",
        "relevance_score",
        "debias_score",
        "source_mask",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = []
    for _, group in pd.DataFrame(rows).groupby("sample_id", sort=False):
        ranked = group.sort_values([score_col, "candidate_movie_id"], ascending=[False, True]).head(top_k)
        for rank, row in enumerate(ranked.to_dict("records"), start=1):
            row["retrieval_rank"] = rank
            row["retrieval_score"] = float(row[score_col])
            row["method"] = method
            out.append(row)
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=cols)


def write_strong_candidate_files_streaming(
    model: DesignDocCrossModalModel,
    query_ds: QueryDataset,
    samples: pd.DataFrame,
    split: str,
    index: StrongRagIndex,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Path], dict]:
    loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    sample_by_id = samples
    genre_matrix_np = model.genre_emb.weight.detach().cpu().numpy()
    chunk_rows = []
    chunk_samples = 0
    output_specs = {
        "rag_hybrid_strong": ("rrf_score", out_dir / f"candidates_rag_hybrid_strong_{split}.csv"),
        "rag_hybrid_strong_relevance": ("relevance_score", out_dir / f"candidates_rag_hybrid_strong_relevance_{split}.csv"),
        "rag_hybrid_strong_relevance_debias": ("debias_score", out_dir / f"candidates_rag_hybrid_strong_relevance_debias_{split}.csv"),
    }
    wrote_header = {name: False for name in output_specs}
    for _, path in output_specs.values():
        if path.exists():
            path.unlink()
    source_counts: Counter[str] = Counter()
    pool_sizes = []

    def flush_chunk() -> None:
        nonlocal chunk_rows, chunk_samples
        if not chunk_rows:
            return
        for name, (score_col, path) in output_specs.items():
            frame = frame_from_pool_rows(chunk_rows, name, score_col, args.max_candidates)
            frame.to_csv(path, mode="a", header=not wrote_header[name], index=False)
            wrote_header[name] = True
        chunk_rows = []
        chunk_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            fused = model.encode_fused(batch_dev)
            q_text = model.to_text(fused)
            q_text = q_text / torch.clamp(torch.linalg.norm(q_text, dim=1, keepdim=True), min=1e-8)
            q_genre = model.to_genre(fused)
            q = q_text.detach().cpu().numpy()
            qg = q_genre.detach().cpu().numpy()
            for i, sample_id in enumerate(batch["sample_id"]):
                sample = sample_by_id.loc[int(sample_id)]
                rows = build_strong_pool(int(sample_id), sample, q[i], qg[i], index, args, genre_matrix_np, split)
                chunk_rows.extend(rows)
                chunk_samples += 1
                pool_sizes.append(len({int(row["candidate_movie_id"]) for row in rows}))
                for row in rows:
                    for source in str(row.get("source_mask", "")).split("|"):
                        if source:
                            source_counts[source] += 1
                if args.strong_rag_chunk_size > 0 and chunk_samples >= args.strong_rag_chunk_size:
                    flush_chunk()
    flush_chunk()
    for name, (_, path) in output_specs.items():
        if not wrote_header[name]:
            frame_from_pool_rows([], name, "rrf_score", args.max_candidates).to_csv(path, index=False)
    diagnostics = {
        "split": split,
        "average_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "source_coverage": {source: int(count) for source, count in source_counts.items()},
        "average_candidates_per_source": {source: float(count / max(len(query_ds), 1)) for source, count in source_counts.items()},
        "strong_rag_chunk_size": int(args.strong_rag_chunk_size),
    }
    return {name: path for name, (_, path) in output_specs.items()}, diagnostics


def make_candidate_rows(
    model: DesignDocCrossModalModel,
    query_ds: QueryDataset,
    samples: pd.DataFrame,
    split: str,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    text_matrix_np: np.ndarray,
    genre_map: dict[int, set[str]],
    pop_norm: dict[int, float],
    sasrec_scores: dict[int, dict[int, tuple[float, int]]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, pd.DataFrame]:
    loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    sample_by_id = samples
    raw_rows = []
    rel_rows = []
    debias_rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            fused = model.encode_fused(batch_dev)
            q_text = model.to_text(fused)
            q_text = q_text / torch.clamp(torch.linalg.norm(q_text, dim=1, keepdim=True), min=1e-8)
            q_genre = model.to_genre(fused)
            q = q_text.detach().cpu().numpy()
            qg = q_genre.detach().cpu().numpy()
            genre_matrix_np = model.genre_emb.weight.detach().cpu().numpy()
            for i, sample_id in enumerate(batch["sample_id"]):
                sample = sample_by_id.loc[int(sample_id)]
                q_i = q[i]
                qg_i = qg[i]
                candidates = initial_text_candidates(q_i, sample, item_to_idx, idx_to_item, text_matrix_np, top_popularity(), args.max_candidates, inject_target=False)
                scored = []
                sas_for_sample = sasrec_scores.get(int(sample_id), {})
                sas_values = [sas_for_sample.get(mid, (0.0, 9999))[0] for mid in candidates]
                sas_mean = float(np.mean(sas_values)) if sas_values else 0.0
                sas_std = max(float(np.std(sas_values)), 1e-6)
                for movie_id in candidates:
                    item_idx = item_to_idx.get(movie_id, 0)
                    cos = float(np.dot(q_i, text_matrix_np[item_idx]))
                    gen_overlap = genre_overlap(movie_id, sample, genre_map)
                    gen_model = float(np.dot(qg_i, genre_matrix_np[item_idx]))
                    gen = args.lambda_genre_train * gen_model + gen_overlap
                    pop = pop_norm.get(movie_id, 0.0)
                    sas_raw, _ = sas_for_sample.get(movie_id, (0.0, 9999))
                    sas_score = (sas_raw - sas_mean) / sas_std
                    relevance = args.lambda_cos * cos + args.lambda_genre * gen + args.lambda_sasrec * sas_score
                    debias = relevance - args.lambda_pop * pop
                    scored.append((movie_id, cos, relevance, debias))
                for rows, score_pos, method in [
                    (raw_rows, 1, "crossmodal_rag"),
                    (rel_rows, 2, "crossmodal_rag_relevance"),
                    (debias_rows, 3, "crossmodal_rag_relevance_debias"),
                ]:
                    ranked = sorted(scored, key=lambda x: (-x[score_pos], x[0]))
                    for rank, item in enumerate(ranked, start=1):
                        rows.append(
                            {
                                "sample_id": int(sample_id),
                                "user_id": int(sample.user_id),
                                "split": split,
                                "target_movie_id": int(sample.target_movie_id),
                                "candidate_movie_id": int(item[0]),
                                "retrieval_rank": rank,
                                "retrieval_score": float(item[score_pos]),
                                "method": method,
                            }
                        )
    return {
        "crossmodal_rag": pd.DataFrame(raw_rows),
        "crossmodal_rag_relevance": pd.DataFrame(rel_rows),
        "crossmodal_rag_relevance_debias": pd.DataFrame(debias_rows),
    }


def write_run_commands(out_dir: Path, args: argparse.Namespace) -> None:
    (out_dir / "run_commands.txt").write_text(
        "python scripts/36_design_doc_crossmodal_rag.py \\\n"
        f"  --run_name {args.run_name} \\\n"
        f"  --max_candidates {args.max_candidates} \\\n"
        f"  --train_candidates {args.train_candidates} \\\n"
        f"  --max_history_len {args.max_history_len} \\\n"
        f"  --embedding_dim {args.embedding_dim} \\\n"
        f"  --num_heads {args.num_heads} \\\n"
        f"  --dropout {args.dropout} \\\n"
        f"  --batch_size {args.batch_size} \\\n"
        f"  --max_epochs {args.max_epochs} \\\n"
        f"  --patience {args.patience} \\\n"
        f"  --learning_rate {args.learning_rate} \\\n"
        f"  --max_train_samples {args.max_train_samples} \\\n"
        f"  --selection_method {args.selection_method} \\\n"
        f"  --selection_eval_samples {args.selection_eval_samples} \\\n"
        f"  --strong_rag_chunk_size {args.strong_rag_chunk_size} \\\n"
        "  --save_train_candidates\n\n"
        "python scripts/19_collect_experiments_summary.py\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_run_name", default="len50_d128_l3_do0p2_lr0p001")
    parser.add_argument("--max_candidates", type=int, default=200)
    parser.add_argument("--train_candidates", type=int, default=50)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--lambda_cos", type=float, default=1.0)
    parser.add_argument("--lambda_genre", type=float, default=0.1)
    parser.add_argument("--lambda_sasrec", type=float, default=0.0)
    parser.add_argument("--lambda_pop", type=float, default=0.1)
    parser.add_argument("--lambda_genre_train", type=float, default=0.1)
    parser.add_argument("--use_ips_loss", action="store_true")
    parser.add_argument("--max_ips_weight", type=float, default=5.0)
    parser.add_argument("--min_propensity", type=float, default=0.05)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--save_train_candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_candidate_injection", choices=["none", "append_last"], default="none")
    parser.add_argument("--strong_rag_chunk_size", type=int, default=2000)
    parser.add_argument("--selection_method", choices=["crossmodal_rag_relevance_debias", "rag_hybrid_strong_relevance_debias"], default="rag_hybrid_strong_relevance_debias")
    parser.add_argument("--selection_eval_samples", type=int, default=1000, help="0 means full validation for checkpoint selection.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_candidate_injection != "none":
        raise ValueError("Strong RAG train target injection is disabled. Use --save_candidate_injection none.")
    ensure_dir(V14_OUT)
    args.run_name = args.run_name.strip() or ("genre_ips" if args.use_ips_loss else "base")
    out_dir = V14_OUT / "runs" / args.run_name
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {out_dir}. Use --overwrite to replace this run.")
    ensure_dir(out_dir)
    write_run_commands(out_dir, args)
    write_json(out_dir / "run_config.json", vars(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = pd.read_csv(SAMPLES)
    train_samples = samples[samples.split == "train"]
    if args.max_train_samples > 0 and len(train_samples) > args.max_train_samples:
        train_samples = train_samples.sample(args.max_train_samples, random_state=args.seed).sort_index()
    val_samples = samples[samples.split == "val"]
    test_samples = samples[samples.split == "test"]
    movie_ids, text_emb = load_text_embeddings()
    item_to_idx, idx_to_item, user_to_idx = build_maps(samples, movie_ids)
    text_matrix, genre_matrix, genre_map, _ = feature_matrices(item_to_idx, movie_ids, text_emb)
    text_np = text_matrix.numpy()
    users, gender_to_idx, age_to_idx, occupation_to_idx = user_feature_maps()
    pop_norm, _ = popularity_maps()
    strong_index = StrongRagIndex(item_to_idx, idx_to_item, text_np, genre_matrix.numpy(), pop_norm)

    train_q = QueryDataset(train_samples, item_to_idx, user_to_idx, users, gender_to_idx, age_to_idx, occupation_to_idx, pop_norm, args.max_history_len)
    val_q = QueryDataset(val_samples, item_to_idx, user_to_idx, users, gender_to_idx, age_to_idx, occupation_to_idx, pop_norm, args.max_history_len)
    test_q = QueryDataset(test_samples, item_to_idx, user_to_idx, users, gender_to_idx, age_to_idx, occupation_to_idx, pop_norm, args.max_history_len)
    selection_val_samples = val_samples
    if args.selection_eval_samples > 0 and len(selection_val_samples) > args.selection_eval_samples:
        selection_val_samples = selection_val_samples.sample(args.selection_eval_samples, random_state=args.seed).sort_index()
    selection_val_q = QueryDataset(selection_val_samples, item_to_idx, user_to_idx, users, gender_to_idx, age_to_idx, occupation_to_idx, pop_norm, args.max_history_len)
    train_cands = build_train_candidate_dict(train_samples, item_to_idx, idx_to_item, text_np, args.train_candidates)
    train_ds = TrainCandidateDataset(train_q, train_cands, item_to_idx, pop_norm, args.train_candidates)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    model = DesignDocCrossModalModel(
        len(item_to_idx),
        len(user_to_idx),
        len(gender_to_idx),
        len(age_to_idx),
        len(occupation_to_idx),
        text_matrix,
        genre_matrix,
        args.embedding_dim,
        args.num_heads,
        args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(f"design_doc_crossmodal train={len(train_ds)} val={len(val_q)} test={len(test_q)} device={device}", flush=True)
    best_metric = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            scores = model.score_candidates(batch_dev, args.lambda_genre_train)
            labels = batch["label"].to(device)
            if args.use_ips_loss:
                loss_per_row = nn.functional.cross_entropy(scores, labels, reduction="none")
                propensity = torch.clamp(batch["target_popularity"].to(device), min=args.min_propensity)
                weights = torch.clamp(1.0 / torch.sqrt(propensity), max=args.max_ips_weight)
                loss = (loss_per_row * weights).mean()
            else:
                loss = loss_fn(scores, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * len(batch["label"])
            total += len(batch["label"])
        if args.selection_method == "crossmodal_rag_relevance_debias":
            val_frames = make_candidate_rows(model, selection_val_q, selection_val_samples, "val", item_to_idx, idx_to_item, text_np, genre_map, pop_norm, {}, args, device)
            val_metrics = eval_candidates(val_frames["crossmodal_rag_relevance_debias"], selection_val_samples)
        else:
            tmp_dir = out_dir / "tmp_selection"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            strong_paths, _ = write_strong_candidate_files_streaming(model, selection_val_q, selection_val_samples, "val", strong_index, args, device, tmp_dir)
            val_metrics = eval_candidates(pd.read_csv(strong_paths["rag_hybrid_strong_relevance_debias"]), selection_val_samples)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        metric = float(val_metrics["NDCG@30"])
        history.append({"epoch": epoch, "loss": total_loss / max(total, 1), "selection_method": args.selection_method, "selection_eval_samples": int(len(selection_val_samples)), **val_metrics})
        print(f"epoch={epoch} loss={history[-1]['loss']:.6f} selection_method={args.selection_method} val_NDCG@30={metric:.6f}", flush=True)
        if metric > best_metric:
            best_metric = metric
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history}, out_dir / "crossmodal_design_doc_model.pt")
    val_sas = sasrec_lookup(args.sasrec_run_name, "val") if args.lambda_sasrec > 0 else {}
    test_sas = sasrec_lookup(args.sasrec_run_name, "test") if args.lambda_sasrec > 0 else {}
    val_frames = make_candidate_rows(model, val_q, val_samples, "val", item_to_idx, idx_to_item, text_np, genre_map, pop_norm, val_sas, args, device)
    test_frames = make_candidate_rows(model, test_q, test_samples, "test", item_to_idx, idx_to_item, text_np, genre_map, pop_norm, test_sas, args, device)
    strong_diagnostics = {"args": vars(args), "splits": {}}
    strong_paths_by_split = {}
    strong_samples_by_split = {}
    if args.save_train_candidates:
        train_strong_paths, train_diag = write_strong_candidate_files_streaming(model, train_q, train_samples, "train", strong_index, args, device, out_dir)
        strong_paths_by_split["train"] = train_strong_paths
        strong_samples_by_split["train"] = train_samples
        strong_diagnostics["splits"]["train"] = train_diag
    val_strong_paths, val_diag = write_strong_candidate_files_streaming(model, val_q, val_samples, "val", strong_index, args, device, out_dir)
    test_strong_paths, test_diag = write_strong_candidate_files_streaming(model, test_q, test_samples, "test", strong_index, args, device, out_dir)
    strong_paths_by_split["val"] = val_strong_paths
    strong_paths_by_split["test"] = test_strong_paths
    strong_samples_by_split["val"] = val_samples
    strong_samples_by_split["test"] = test_samples
    strong_diagnostics["splits"]["val"] = val_diag
    strong_diagnostics["splits"]["test"] = test_diag
    strong_metric_rows_by_split: dict[str, list[dict]] = defaultdict(list)
    for split, paths in strong_paths_by_split.items():
        split_samples = strong_samples_by_split[split]
        for name, path in paths.items():
            frame = pd.read_csv(path)
            metrics = eval_candidates(frame, split_samples)
            metric_row = {
                "version": "v14_design_doc_pipeline",
                "stage": "rag_hybrid_strong",
                "split": split,
                "config": f"{args.run_name}/{name}",
                "run_name": args.run_name,
                **metrics,
            }
            strong_metric_rows_by_split[split].append(metric_row)
            strong_diagnostics["splits"][split].setdefault("target_in_by_method", {})[name] = {
                "TargetIn@30": metrics.get("TargetIn@30"),
                "TargetIn@50": metrics.get("TargetIn@50"),
                "TargetIn@200": metrics.get("TargetIn@200"),
            }
        pd.DataFrame(strong_metric_rows_by_split[split]).to_csv(out_dir / f"rag_hybrid_strong_{split}_metrics.csv", index=False)
    write_json(out_dir / "rag_hybrid_strong_diagnostics.json", strong_diagnostics)
    val_rows = []
    test_rows = []
    for name, frame in val_frames.items():
        metrics = eval_candidates(frame, val_samples)
        frame.to_csv(out_dir / f"candidates_{name}_val.csv", index=False)
        val_rows.append({"version": "v14_design_doc_pipeline", "stage": "design_doc_crossmodal_rag", "split": "val", "config": f"{args.run_name}/{name}", "run_name": args.run_name, **metrics})
    selected = set(pd.DataFrame(val_rows).sort_values(["NDCG@30", "HR@1"], ascending=False).head(3)["config"].tolist())
    selected_names = {config.split("/", 1)[1] if "/" in config else config for config in selected}
    for name, frame in test_frames.items():
        if name not in selected_names:
            continue
        frame.to_csv(out_dir / f"candidates_{name}_test.csv", index=False)
        metrics = eval_candidates(frame, test_samples)
        test_rows.append({"version": "v14_design_doc_pipeline", "stage": "design_doc_crossmodal_rag", "split": "test", "config": f"{args.run_name}/{name}", "run_name": args.run_name, **metrics})
    pd.DataFrame(val_rows).to_csv(out_dir / "crossmodal_val_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "crossmodal_test_metrics.csv", index=False)
    write_json(
        out_dir / "best_crossmodal_configs.json",
        {
            "selection_split": "val",
            "selected_for_test": sorted(selected),
            "selected_names_for_test": sorted(selected_names),
            "args": vars(args),
            "history": history,
        },
    )
    print(f"saved v14 design doc outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
