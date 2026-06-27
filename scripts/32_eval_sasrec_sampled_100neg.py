from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import OUT, POPULARITY, SAMPLES, TRAIN_INTERACTIONS, parse_ids, write_json


V10_OUT = OUT / "v10_sasrec_retrieval"


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int, embedding_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1), persistent=False)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        padding_mask = input_ids.eq(0)
        encoded = self.encoder(x, mask=self.causal_mask[:seq_len, :seq_len], src_key_padding_mask=padding_mask)
        return self.norm(encoded)

    def score_last(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids)
        last_pos = lengths.clamp_min(1).sub(1)
        last_hidden = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last_pos]
        logits = last_hidden @ self.item_embedding.weight.T
        logits[:, 0] = -1e9
        return logits


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        return Path(args.checkpoint_path)
    return V10_OUT / "runs" / args.run_name / "sasrec_model.pt"


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return V10_OUT / "runs" / args.run_name


def load_checkpoint(args: argparse.Namespace) -> tuple[SASRec, dict[int, int], dict, Path]:
    path = checkpoint_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Missing SASRec checkpoint: {path}")
    payload = safe_torch_load(path)
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    item_to_idx_raw = payload.get("item_to_idx") if isinstance(payload, dict) else None
    if item_to_idx_raw is None:
        raise KeyError("SASRec checkpoint must contain item_to_idx.")
    item_to_idx = {int(k): int(v) for k, v in item_to_idx_raw.items()}
    config = dict(payload.get("config") or {})
    max_seq_len = int(config.get("max_seq_len", args.max_history_len))
    embedding_dim = int(config.get("embedding_dim", state["item_embedding.weight"].shape[1]))
    num_heads = int(config.get("num_heads", 4))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.0))
    model = SASRec(len(item_to_idx), max_seq_len, embedding_dim, num_heads, num_layers, dropout)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, item_to_idx, config, path


def all_movie_ids(samples: pd.DataFrame, item_to_idx: dict[int, int]) -> list[int]:
    movies = set(int(x) for x in item_to_idx)
    if POPULARITY.exists():
        movies.update(pd.read_csv(POPULARITY).movie_id.astype(int).tolist())
    if TRAIN_INTERACTIONS.exists():
        movies.update(pd.read_csv(TRAIN_INTERACTIONS).movie_id.astype(int).tolist())
    movies.update(samples.target_movie_id.astype(int).tolist())
    for value in samples.model_history_movie_ids.tolist():
        movies.update(parse_ids(value))
    for value in samples.seen_movie_ids.tolist():
        movies.update(parse_ids(value))
    return sorted(int(x) for x in movies if int(x) in item_to_idx)


def make_input(sample: pd.Series, item_to_idx: dict[int, int], max_history_len: int, model_seq_len: int, device: torch.device):
    history = [item_to_idx[mid] for mid in parse_ids(sample.model_history_movie_ids) if mid in item_to_idx][-max_history_len:]
    history = history[-model_seq_len:]
    if not history:
        return None
    arr = np.zeros(model_seq_len, dtype=np.int64)
    arr[: len(history)] = np.array(history, dtype=np.int64)
    return (
        torch.tensor(arr, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor([len(history)], dtype=torch.long, device=device),
    )


def sample_negatives(
    rng: np.random.Generator,
    candidate_pool: np.ndarray,
    forbidden: set[int],
    num_negatives: int,
) -> list[int] | None:
    allowed = [int(movie_id) for movie_id in candidate_pool.tolist() if int(movie_id) not in forbidden]
    if len(allowed) < num_negatives:
        return None
    return [int(x) for x in rng.choice(np.array(allowed, dtype=np.int64), size=num_negatives, replace=False).tolist()]


def dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history_len", type=int, default=200)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(save_samples=True)
    parser.add_argument("--save_samples", dest="save_samples", action="store_true")
    parser.add_argument("--no_save_samples", dest="save_samples", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model, item_to_idx, config, ckpt_path = load_checkpoint(args)
    device = torch.device(args.device)
    model.to(device)
    samples = pd.read_csv(SAMPLES)
    split_samples = samples[samples.split == args.split].copy()
    movie_pool = np.array(all_movie_ids(samples, item_to_idx), dtype=np.int64)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"sampled_{args.num_negatives}neg_{args.split}_items_seed{args.seed}.jsonl"
    metrics_path = out_dir / f"sampled_{args.num_negatives}neg_{args.split}_seed{args.seed}_metrics.json"
    sample_file = samples_path.open("w", encoding="utf-8") if args.save_samples else None

    hits10 = 0
    ndcg10 = 0.0
    mrr10 = 0.0
    evaluated = 0
    skipped = 0
    skipped_reasons = {"missing_target_idx": 0, "empty_history": 0, "not_enough_negatives": 0}
    model_seq_len = int(config.get("max_seq_len", args.max_history_len))
    with torch.no_grad():
        for sample_id, sample in split_samples.iterrows():
            target = int(sample.target_movie_id)
            target_idx = item_to_idx.get(target)
            if target_idx is None:
                skipped += 1
                skipped_reasons["missing_target_idx"] += 1
                continue
            model_input = make_input(sample, item_to_idx, args.max_history_len, model_seq_len, device)
            if model_input is None:
                skipped += 1
                skipped_reasons["empty_history"] += 1
                continue
            seen = set(parse_ids(sample.seen_movie_ids))
            forbidden = set(seen)
            forbidden.add(target)
            negatives = sample_negatives(rng, movie_pool, forbidden, args.num_negatives)
            if negatives is None:
                skipped += 1
                skipped_reasons["not_enough_negatives"] += 1
                continue
            scored_movie_ids = [target] + negatives
            scored_indices = [target_idx] + [item_to_idx[mid] for mid in negatives]
            logits = model.score_last(*model_input)[0]
            scores = logits[torch.tensor(scored_indices, dtype=torch.long, device=device)].float().detach().cpu().numpy()
            order = np.argsort(-scores)
            target_rank = int(np.where(order == 0)[0][0]) + 1
            hits10 += int(target_rank <= 10)
            if target_rank <= 10:
                ndcg10 += dcg(target_rank)
                mrr10 += 1.0 / target_rank
            evaluated += 1
            if sample_file is not None:
                sample_file.write(
                    json.dumps(
                        {
                            "sample_id": int(sample_id),
                            "user_id": int(sample.user_id),
                            "target_movie_id": target,
                            "negative_movie_ids": negatives,
                            "scored_movie_ids": scored_movie_ids,
                            "target_rank": target_rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if evaluated % 1000 == 0:
                print(f"sampled eval {args.split}: evaluated={evaluated} skipped={skipped}", flush=True)
    if sample_file is not None:
        sample_file.close()
    denom = max(evaluated, 1)
    metrics = {
        "run_name": args.run_name,
        "split": args.split,
        "protocol": "sampled_100_negative" if args.num_negatives == 100 else f"sampled_{args.num_negatives}_negative",
        "num_negatives": int(args.num_negatives),
        "seed": int(args.seed),
        "max_history_len": int(args.max_history_len),
        "model_max_seq_len": int(model_seq_len),
        "checkpoint_path": str(ckpt_path),
        "evaluated_samples": int(evaluated),
        "skipped_samples": int(skipped),
        "skipped_reasons": skipped_reasons,
        "Hit@10": float(hits10 / denom),
        "NDCG@10": float(ndcg10 / denom),
        "MRR@10": float(mrr10 / denom),
        "sampled_items_path": str(samples_path) if args.save_samples else "",
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
