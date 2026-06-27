from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import EMB, MOVIE_METADATA, MOVIE_ORDER, OUT, POPULARITY, SAMPLES, genres, parse_ids, write_json
from stage_a_experiment_utils import ensure_dir, eval_candidates, top_popularity


V10_OUT = OUT / "v10_sasrec_retrieval"
V14_OUT = OUT / "v14_cross_attention_reranker"


def load_sasrec_module():
    path = Path(__file__).with_name("32_train_sasrec_candidates.py")
    spec = importlib.util.spec_from_file_location("sasrec_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SASREC_MOD = load_sasrec_module()


def split_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def beta_token(value: float) -> str:
    return str(value).replace(".", "p")


def sasrec_run_dir(run_name: str) -> Path:
    path = V10_OUT / "runs" / run_name
    if not path.exists():
        raise FileNotFoundError(f"Missing SASRec run dir: {path}")
    return path


def load_sasrec_candidates(run_dir: Path, split: str) -> pd.DataFrame | None:
    path = run_dir / f"candidates_sasrec_{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_train_candidates_from_sasrec(run_dir: Path, train_samples: pd.DataFrame, top_k: int, device: torch.device) -> pd.DataFrame:
    model_path = run_dir / "sasrec_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot build train SASRec candidates without checkpoint: {model_path}")
    try:
        payload = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location=device)
    config = payload.get("config", {})
    item_to_idx = {int(k): int(v) for k, v in payload["item_to_idx"].items()}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    max_seq_len = int(config.get("max_seq_len", 50))
    model = SASREC_MOD.SASRec(
        len(item_to_idx),
        max_seq_len,
        int(config.get("embedding_dim", 128)),
        int(config.get("num_heads", 4)),
        int(config.get("num_layers", 2)),
        float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    print(f"building train SASRec candidates from {model_path} rows={len(train_samples)}", flush=True)
    return SASREC_MOD.build_candidates(model, train_samples, "train", item_to_idx, idx_to_item, top_popularity(), top_k, max_seq_len, device)


def build_item_mappings(*frames: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    samples = pd.read_csv(SAMPLES)
    item_ids = set(pd.read_csv(POPULARITY).movie_id.astype(int).tolist())
    item_ids.update(samples.target_movie_id.astype(int).tolist())
    for value in samples.model_history_movie_ids.tolist():
        item_ids.update(parse_ids(value))
    for frame in frames:
        if frame is not None and not frame.empty:
            item_ids.update(frame.candidate_movie_id.astype(int).tolist())
            item_ids.update(frame.target_movie_id.astype(int).tolist())
    item_to_idx = {movie_id: idx + 1 for idx, movie_id in enumerate(sorted(item_ids))}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    return item_to_idx, idx_to_item


def feature_matrices(item_to_idx: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor, dict[int, float], int, int]:
    num_items = len(item_to_idx)
    if EMB.exists() and MOVIE_ORDER.exists():
        emb = np.load(EMB).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)
        order = np.load(MOVIE_ORDER).astype(int).tolist()
        pos = {int(movie_id): i for i, movie_id in enumerate(order)}
        text_dim = int(emb.shape[1])
        text_matrix = np.zeros((num_items + 1, text_dim), dtype=np.float32)
        for movie_id, idx in item_to_idx.items():
            if movie_id in pos:
                text_matrix[idx] = emb[pos[movie_id]]
    else:
        text_dim = 1
        text_matrix = np.zeros((num_items + 1, text_dim), dtype=np.float32)

    meta = pd.read_csv(MOVIE_METADATA) if MOVIE_METADATA.exists() else pd.DataFrame(columns=["movie_id", "genres"])
    genre_values = sorted({g for value in meta.get("genres", pd.Series(dtype=str)).tolist() for g in genres(value)})
    genre_to_idx = {genre: i for i, genre in enumerate(genre_values)}
    genre_dim = max(len(genre_to_idx), 1)
    genre_matrix = np.zeros((num_items + 1, genre_dim), dtype=np.float32)
    if not meta.empty and genre_to_idx:
        for row in meta.itertuples(index=False):
            movie_id = int(row.movie_id)
            item_idx = item_to_idx.get(movie_id)
            if item_idx is None:
                continue
            for genre in genres(row.genres):
                genre_matrix[item_idx, genre_to_idx[genre]] = 1.0

    pop = pd.read_csv(POPULARITY)
    max_pop = max(float(pop.train_popularity.max()), 1.0)
    pop_map = {int(row.movie_id): float(np.log1p(row.train_popularity) / np.log1p(max_pop)) for row in pop.itertuples(index=False)}
    return torch.tensor(text_matrix), torch.tensor(genre_matrix), pop_map, text_dim, genre_dim


class CrossAttentionDataset(Dataset):
    def __init__(
        self,
        candidates: pd.DataFrame,
        samples: pd.DataFrame,
        item_to_idx: dict[int, int],
        pop_map: dict[int, float],
        max_candidates: int,
        max_history_len: int,
        require_label: bool,
    ):
        self.rows = []
        sample_lookup = samples
        for sample_id, group in candidates.sort_values(["sample_id", "retrieval_rank"]).groupby("sample_id", sort=False):
            sample_id_int = int(sample_id)
            if sample_id_int not in sample_lookup.index:
                continue
            sample = sample_lookup.loc[sample_id_int]
            group = group.head(max_candidates).copy()
            if group.empty:
                continue
            target = int(sample.target_movie_id)
            candidate_ids = group.candidate_movie_id.astype(int).tolist()
            label = candidate_ids.index(target) if target in candidate_ids else -1
            if require_label and label < 0:
                continue
            history = [item_to_idx[mid] for mid in parse_ids(sample.model_history_movie_ids)[-max_history_len:] if mid in item_to_idx]
            hist_arr = np.zeros(max_history_len, dtype=np.int64)
            hist_mask = np.zeros(max_history_len, dtype=np.bool_)
            if history:
                hist_arr[: len(history)] = np.array(history, dtype=np.int64)
                hist_mask[: len(history)] = True

            cand_arr = np.zeros(max_candidates, dtype=np.int64)
            cand_mask = np.zeros(max_candidates, dtype=np.bool_)
            cand_idx = [item_to_idx.get(mid, 0) for mid in candidate_ids]
            cand_arr[: len(cand_idx)] = np.array(cand_idx, dtype=np.int64)
            cand_mask[: len(cand_idx)] = True

            ranks = group.retrieval_rank.astype(float).to_numpy()
            scores = group.retrieval_score.astype(float).to_numpy()
            if len(scores) > 1:
                scores = (scores - np.nanmean(scores)) / max(float(np.nanstd(scores)), 1e-6)
            else:
                scores = np.zeros_like(scores)
            numeric = np.zeros((max_candidates, 3), dtype=np.float32)
            numeric[: len(group), 0] = np.nan_to_num(scores, nan=0.0)
            numeric[: len(group), 1] = ranks / 200.0
            numeric[: len(group), 2] = np.array([pop_map.get(mid, 0.0) for mid in candidate_ids], dtype=np.float32)
            self.rows.append((sample_id_int, group, hist_arr, hist_mask, cand_arr, cand_mask, numeric, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        sample_id, group, hist_arr, hist_mask, cand_arr, cand_mask, numeric, label = self.rows[idx]
        return {
            "sample_id": sample_id,
            "history_ids": torch.tensor(hist_arr, dtype=torch.long),
            "history_mask": torch.tensor(hist_mask, dtype=torch.bool),
            "candidate_ids": torch.tensor(cand_arr, dtype=torch.long),
            "candidate_mask": torch.tensor(cand_mask, dtype=torch.bool),
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "sample_id": [row["sample_id"] for row in batch],
        "history_ids": torch.stack([row["history_ids"] for row in batch]),
        "history_mask": torch.stack([row["history_mask"] for row in batch]),
        "candidate_ids": torch.stack([row["candidate_ids"] for row in batch]),
        "candidate_mask": torch.stack([row["candidate_mask"] for row in batch]),
        "numeric": torch.stack([row["numeric"] for row in batch]),
        "label": torch.stack([row["label"] for row in batch]),
    }


class CrossAttentionReranker(nn.Module):
    def __init__(
        self,
        num_items: int,
        text_matrix: torch.Tensor,
        genre_matrix: torch.Tensor,
        max_history_len: int,
        embedding_dim: int,
        text_projection_dim: int,
        genre_projection_dim: int,
        num_heads: int,
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.text_emb = nn.Embedding.from_pretrained(text_matrix.float(), freeze=True, padding_idx=0)
        self.genre_emb = nn.Embedding.from_pretrained(genre_matrix.float(), freeze=True, padding_idx=0)
        self.text_proj = nn.Linear(text_matrix.shape[1], text_projection_dim)
        self.genre_proj = nn.Linear(genre_matrix.shape[1], genre_projection_dim)
        self.numeric_proj = nn.Linear(3, embedding_dim)
        self.history_fuse = nn.Linear(embedding_dim + text_projection_dim + genre_projection_dim, embedding_dim)
        self.candidate_fuse = nn.Linear(embedding_dim * 2 + text_projection_dim + genre_projection_dim, embedding_dim)
        self.pos_emb = nn.Embedding(max_history_len, embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_items(self, item_ids: torch.Tensor, numeric: torch.Tensor | None, is_candidate: bool) -> torch.Tensor:
        item = self.item_emb(item_ids)
        text = self.text_proj(self.text_emb(item_ids))
        genre = self.genre_proj(self.genre_emb(item_ids))
        if is_candidate:
            assert numeric is not None
            num = self.numeric_proj(numeric)
            fused = self.candidate_fuse(torch.cat([item, text, genre, num], dim=-1))
        else:
            fused = self.history_fuse(torch.cat([item, text, genre], dim=-1))
        return self.dropout(fused)

    def forward(self, history_ids: torch.Tensor, history_mask: torch.Tensor, candidate_ids: torch.Tensor, candidate_mask: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        history = self.encode_items(history_ids, None, is_candidate=False)
        pos = torch.arange(history_ids.shape[1], device=history_ids.device).unsqueeze(0)
        history = history + self.pos_emb(pos)
        candidate = self.encode_items(candidate_ids, numeric, is_candidate=True)
        effective_history_mask = history_mask.clone()
        empty = ~effective_history_mask.any(dim=1)
        if empty.any():
            effective_history_mask[empty, 0] = True
        attended, _ = self.attn(candidate, history, history, key_padding_mask=~effective_history_mask)
        x = torch.cat([candidate, attended, candidate * attended, numeric], dim=-1)
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask, -1e9)
        return scores


def rerank_candidates(
    model: CrossAttentionReranker,
    base: pd.DataFrame,
    ds: CrossAttentionDataset,
    device: torch.device,
    output_top_k: int = 200,
) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    scored: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            scores = model(
                batch["history_ids"].to(device),
                batch["history_mask"].to(device),
                batch["candidate_ids"].to(device),
                batch["candidate_mask"].to(device),
                batch["numeric"].to(device),
            ).detach().cpu().numpy()
            for sample_id, row_scores in zip(batch["sample_id"], scores):
                scored[int(sample_id)] = row_scores
    rows = []
    for sample_id, group in base.sort_values(["sample_id", "retrieval_rank"]).groupby("sample_id", sort=False):
        group = group.copy()
        row_scores = scored.get(int(sample_id))
        top = group.head(len(row_scores) if row_scores is not None else 0).copy() if row_scores is not None else pd.DataFrame()
        if row_scores is not None and not top.empty:
            top["crossattn_score"] = row_scores[: len(top)]
            top_ranked = top.sort_values(["crossattn_score", "candidate_movie_id"], ascending=[False, True])
            top_ids = set(top_ranked.candidate_movie_id.astype(int).tolist())
            tail = group[~group.candidate_movie_id.astype(int).isin(top_ids)].copy()
            tail["crossattn_score"] = np.nan
            combined = pd.concat([top_ranked, tail], ignore_index=True)
        else:
            combined = group.copy()
            combined["crossattn_score"] = np.nan
        for rank, item in enumerate(combined.head(output_top_k).itertuples(index=False), start=1):
            row = item._asdict()
            row["retrieval_rank"] = rank
            row["retrieval_score"] = float(row.get("crossattn_score")) if not pd.isna(row.get("crossattn_score", np.nan)) else -1e9
            row["method"] = "crossattn"
            rows.append(row)
    return pd.DataFrame(rows)


def norm_by_sample(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("sample_id")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[col] - mean) / std).fillna(0.0)


def blend_candidates(
    base: pd.DataFrame,
    neural: pd.DataFrame,
    mode: str,
    beta: float,
    config: str,
    output_top_k: int = 200,
) -> pd.DataFrame:
    base_df = base.rename(columns={"retrieval_rank": "base_rank", "retrieval_score": "base_score"})
    neural_df = neural[["sample_id", "candidate_movie_id", "retrieval_rank", "crossattn_score"]].rename(
        columns={"retrieval_rank": "neural_rank", "crossattn_score": "neural_score"}
    )
    merged = base_df.merge(neural_df, on=["sample_id", "candidate_movie_id"], how="left")
    merged["neural_rank"] = merged["neural_rank"].fillna(9999.0)
    low_neural = merged["neural_score"].min(skipna=True) - 1.0 if merged["neural_score"].notna().any() else -1e9
    merged["neural_score"] = merged["neural_score"].fillna(low_neural)
    if mode == "rrf":
        merged["final_score"] = 1.0 / (60.0 + merged["base_rank"]) + beta * (1.0 / (60.0 + merged["neural_rank"]))
    elif mode == "score":
        if beta == 0:
            merged["final_score"] = -merged["base_rank"].astype(float)
        else:
            merged["base_norm"] = norm_by_sample(merged, "base_score")
            merged["neural_norm"] = norm_by_sample(merged, "neural_score")
            merged["final_score"] = merged["base_norm"] + beta * merged["neural_norm"]
    else:
        raise ValueError(mode)
    rows = []
    for _, group in merged.groupby("sample_id", sort=False):
        ranked = group.sort_values(["final_score", "candidate_movie_id"], ascending=[False, True]).head(output_top_k)
        for rank, item in enumerate(ranked.itertuples(index=False), start=1):
            row = item._asdict()
            row["retrieval_rank"] = rank
            row["retrieval_score"] = float(row["final_score"])
            row["method"] = config
            rows.append(row)
    cols = ["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id", "retrieval_rank", "retrieval_score", "method"]
    return pd.DataFrame(rows)[cols]


def write_run_commands(out_dir: Path, args: argparse.Namespace) -> None:
    (out_dir / "run_commands.txt").write_text(
        "python scripts/36_train_cross_attention_reranker.py \\\n"
        f"  --sasrec_run_name {args.sasrec_run_name} \\\n"
        f"  --output_dir {out_dir.as_posix()} \\\n"
        f"  --max_candidates {args.max_candidates} \\\n"
        f"  --max_history_len {args.max_history_len} \\\n"
        f"  --embedding_dim {args.embedding_dim} \\\n"
        f"  --text_projection_dim {args.text_projection_dim} \\\n"
        f"  --genre_projection_dim {args.genre_projection_dim} \\\n"
        f"  --num_heads {args.num_heads} \\\n"
        f"  --dropout {args.dropout} \\\n"
        f"  --hidden_dim {args.hidden_dim} \\\n"
        f"  --batch_size {args.batch_size} \\\n"
        f"  --max_epochs {args.max_epochs} \\\n"
        f"  --patience {args.patience} \\\n"
        f"  --learning_rate {args.learning_rate} \\\n"
        f"  --weight_decay {args.weight_decay} \\\n"
        f"  --max_train_samples {args.max_train_samples} \\\n"
        f"  --betas {args.betas} \\\n"
        f"  --blend_modes {args.blend_modes} \\\n"
        f"  --top_n_test {args.top_n_test}\n\n"
        "Expected train outputs after this run:\n"
        f"  {out_dir.as_posix()}/candidates_crossattn_train.csv\n"
        f"  {out_dir.as_posix()}/candidates_blend_crossattn_score_beta0p5_train.csv\n\n"
        "Note: train outputs are saved only for top --max_candidates candidates per sample, intended for LLM-stage m=30.\n\n"
        "python scripts/19_collect_experiments_summary.py\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_run_name", default="len50_d128_l3_do0p2_lr0p001")
    parser.add_argument("--output_dir", default=str(V14_OUT))
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--text_projection_dim", type=int, default=128)
    parser.add_argument("--genre_projection_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--betas", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--blend_modes", default="score,rrf")
    parser.add_argument("--top_n_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir))
    write_run_commands(out_dir, args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = sasrec_run_dir(args.sasrec_run_name)
    samples = pd.read_csv(SAMPLES)
    train_samples = samples[samples.split == "train"]
    if args.max_train_samples > 0 and len(train_samples) > args.max_train_samples:
        train_samples = train_samples.sample(args.max_train_samples, random_state=args.seed).sort_index()
    val_samples = samples[samples.split == "val"]
    test_samples = samples[samples.split == "test"]

    train_candidates = load_sasrec_candidates(run_dir, "train")
    if train_candidates is None:
        train_candidates = build_train_candidates_from_sasrec(run_dir, train_samples, args.max_candidates, device)
        train_candidates.to_csv(out_dir / "candidates_sasrec_train.csv", index=False)
    else:
        train_candidates = train_candidates[train_candidates.sample_id.astype(int).isin(set(train_samples.index.astype(int)))].copy()
    val_candidates = load_sasrec_candidates(run_dir, "val")
    test_candidates = load_sasrec_candidates(run_dir, "test")
    if val_candidates is None or test_candidates is None:
        raise FileNotFoundError(f"SASRec val/test candidates are required in {run_dir}")

    item_to_idx, _ = build_item_mappings(train_candidates, val_candidates, test_candidates)
    text_matrix, genre_matrix, pop_map, text_dim, genre_dim = feature_matrices(item_to_idx)
    train_ds = CrossAttentionDataset(train_candidates, train_samples, item_to_idx, pop_map, args.max_candidates, args.max_history_len, require_label=True)
    val_ds = CrossAttentionDataset(val_candidates, val_samples, item_to_idx, pop_map, args.max_candidates, args.max_history_len, require_label=False)
    test_ds = CrossAttentionDataset(test_candidates, test_samples, item_to_idx, pop_map, args.max_candidates, args.max_history_len, require_label=False)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    model = CrossAttentionReranker(
        len(item_to_idx),
        text_matrix,
        genre_matrix,
        args.max_history_len,
        args.embedding_dim,
        args.text_projection_dim,
        args.genre_projection_dim,
        args.num_heads,
        args.dropout,
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(
        f"crossattn run={args.sasrec_run_name} train_target_present={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"items={len(item_to_idx)} text_dim={text_dim} genre_dim={genre_dim} device={device}",
        flush=True,
    )

    best_metric = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_idx, batch in enumerate(loader, start=1):
            scores = model(
                batch["history_ids"].to(device),
                batch["history_mask"].to(device),
                batch["candidate_ids"].to(device),
                batch["candidate_mask"].to(device),
                batch["numeric"].to(device),
            )
            loss = loss_fn(scores, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            total += len(batch["label"])
            if batch_idx % 100 == 0:
                print(f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={loss.item():.6f}", flush=True)
        val_ranked = rerank_candidates(model, val_candidates, val_ds, device)
        val_metrics = eval_candidates(val_ranked, val_samples)
        metric = float(val_metrics["NDCG@30"])
        history.append({"epoch": epoch, "loss": total_loss / max(total, 1), **val_metrics})
        print(f"epoch={epoch} loss={history[-1]['loss']:.6f} val_NDCG@30={metric:.6f}", flush=True)
        if metric > best_metric:
            best_metric = metric
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping epoch={epoch} best_NDCG@30={best_metric:.6f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_idx": item_to_idx,
            "args": vars(args),
            "history": history,
        },
        out_dir / "crossattn_model.pt",
    )

    train_candidates_output = (
        train_candidates.sort_values(["sample_id", "retrieval_rank"]).groupby("sample_id", sort=False).head(args.max_candidates).copy()
    )
    train_ds_output = CrossAttentionDataset(
        train_candidates_output,
        train_samples,
        item_to_idx,
        pop_map,
        args.max_candidates,
        args.max_history_len,
        require_label=False,
    )
    ranked_train = rerank_candidates(
        model,
        train_candidates_output,
        train_ds_output,
        device,
        output_top_k=args.max_candidates,
    )
    ranked_train.to_csv(out_dir / "candidates_crossattn_train.csv", index=False)
    try:
        train_metrics = eval_candidates(ranked_train, train_samples)
        pd.DataFrame(
            [
                {
                    "version": "v14_cross_attention_reranker",
                    "stage": "crossattn",
                    "split": "train",
                    "config": "crossattn",
                    "sasrec_run_name": args.sasrec_run_name,
                    **train_metrics,
                }
            ]
        ).to_csv(out_dir / "crossattn_train_metrics.csv", index=False)
        print(f"saved crossattn train HR@1={train_metrics['HR@1']:.6f} NDCG@30={train_metrics['NDCG@30']:.6f}", flush=True)
    except Exception as exc:
        print(f"warning: could not evaluate train crossattn candidates: {exc}", flush=True)
    train_blend_config = "blend_crossattn_score_beta0p5"
    train_blend = blend_candidates(
        train_candidates_output,
        ranked_train,
        "score",
        0.5,
        train_blend_config,
        output_top_k=args.max_candidates,
    )
    train_blend.to_csv(out_dir / f"candidates_{train_blend_config}_train.csv", index=False)
    print(f"saved train blend {train_blend_config} rows={len(train_blend)}", flush=True)

    cross_rows = []
    ranked_by_split = {}
    for split, base, ds, split_samples in [("val", val_candidates, val_ds, val_samples), ("test", test_candidates, test_ds, test_samples)]:
        ranked = rerank_candidates(model, base, ds, device)
        ranked_by_split[split] = ranked
        ranked.to_csv(out_dir / f"candidates_crossattn_{split}.csv", index=False)
        metrics = eval_candidates(ranked, split_samples)
        cross_rows.append(
            {
                "version": "v14_cross_attention_reranker",
                "stage": "crossattn",
                "split": split,
                "config": "crossattn",
                "sasrec_run_name": args.sasrec_run_name,
                **metrics,
            }
        )
        print(f"saved crossattn {split} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    cross_df = pd.DataFrame(cross_rows)
    cross_df[cross_df.split == "val"].to_csv(out_dir / "crossattn_val_metrics.csv", index=False)
    cross_df[cross_df.split == "test"].to_csv(out_dir / "crossattn_test_metrics.csv", index=False)

    betas = [float(x) for x in split_list(args.betas)]
    modes = split_list(args.blend_modes)
    val_blend_rows = []
    lookup = {}
    for mode in modes:
        for beta in betas:
            config = f"blend_crossattn_{mode}_beta{beta_token(beta)}"
            cand = blend_candidates(val_candidates, ranked_by_split["val"], mode, beta, config)
            metrics = eval_candidates(cand, val_samples)
            val_blend_rows.append(
                {
                    "version": "v14_cross_attention_reranker",
                    "stage": "crossattn_safe_blend",
                    "split": "val",
                    "config": config,
                    "sasrec_run_name": args.sasrec_run_name,
                    "mode": mode,
                    "beta": beta,
                    **metrics,
                }
            )
            lookup[config] = (mode, beta)
            print(f"val blend {config} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    val_blend_df = pd.DataFrame(val_blend_rows)
    val_blend_df.to_csv(out_dir / "blend_val_metrics.csv", index=False)
    selected = set()
    best_configs = {}
    for metric in ["HR@1", "NDCG@30", "MRR@30", "TargetIn@30"]:
        top = val_blend_df.sort_values([metric, "HR@1", "NDCG@30"], ascending=False).head(args.top_n_test)
        best_configs[metric] = top.to_dict("records")
        selected.update(top["config"].tolist())

    test_rows = []
    for config in sorted(selected):
        mode, beta = lookup[config]
        for split, base, split_samples in [("val", val_candidates, val_samples), ("test", test_candidates, test_samples)]:
            cand = blend_candidates(base, ranked_by_split[split], mode, beta, config)
            cand.to_csv(out_dir / f"candidates_{config}_{split}.csv", index=False)
            metrics = eval_candidates(cand, split_samples)
            if split == "test":
                test_rows.append(
                    {
                        "version": "v14_cross_attention_reranker",
                        "stage": "crossattn_safe_blend",
                        "split": "test",
                        "config": config,
                        "sasrec_run_name": args.sasrec_run_name,
                        "mode": mode,
                        "beta": beta,
                        **metrics,
                    }
                )
            print(f"saved blend {split} {config} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    pd.DataFrame(test_rows).to_csv(out_dir / "blend_test_metrics.csv", index=False)
    write_json(
        out_dir / "best_crossattn_configs.json",
        {
            "selection_split": "val",
            "sasrec_run_name": args.sasrec_run_name,
            "selected_for_test": sorted(selected),
            "best_configs": json.loads(json.dumps(best_configs, default=float)),
            "history": history,
        },
    )
    print(f"saved v14 outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
