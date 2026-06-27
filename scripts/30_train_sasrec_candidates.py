from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from common import OUT, POPULARITY, SAMPLES, TRAIN_INTERACTIONS, parse_ids, write_json
from stage_a_experiment_utils import ensure_dir, eval_candidates, top_popularity


V22_OUT = OUT / "v2_2"
V10_OUT = OUT / "v10_sasrec_retrieval"
DEFAULT_SEQ = "seq_md5_ln5_rd0p5_sd0p7_confidence_log_count"


def safe_token(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "").replace("+", "").replace("/", "_")


def default_run_name(args: argparse.Namespace) -> str:
    return (
        f"len{args.max_seq_len}_d{args.embedding_dim}_l{args.num_layers}"
        f"_do{safe_token(args.dropout)}_lr{safe_token(args.learning_rate)}"
    )


class SasRecSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], item_to_idx: dict[int, int], max_seq_len: int, max_examples: int = 0):
        self.examples: list[list[int]] = []
        self.max_seq_len = max_seq_len
        for seq in sequences:
            idx_seq = [item_to_idx[movie_id] for movie_id in seq if movie_id in item_to_idx]
            if len(idx_seq) < 2:
                continue
            for end in range(2, len(idx_seq) + 1):
                window = idx_seq[max(0, end - (max_seq_len + 1)) : end]
                if len(window) >= 2:
                    self.examples.append(window)
                    if max_examples > 0 and len(self.examples) >= max_examples:
                        return
        if not self.examples:
            raise ValueError("No SASRec sequence examples were built.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.examples[idx]
        inputs = seq[:-1][-self.max_seq_len :]
        labels = seq[1:][-self.max_seq_len :]
        input_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        label_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        input_arr[: len(inputs)] = np.array(inputs, dtype=np.int64)
        label_arr[: len(labels)] = np.array(labels, dtype=np.int64)
        return (
            torch.tensor(input_arr, dtype=torch.long),
            torch.tensor(label_arr, dtype=torch.long),
            torch.tensor(len(inputs), dtype=torch.long),
        )


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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids)
        logits = hidden @ self.item_embedding.weight.T
        logits[..., 0] = -1e9
        return logits

    def score_last(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids)
        last_pos = lengths.clamp_min(1).sub(1)
        last_hidden = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last_pos]
        logits = last_hidden @ self.item_embedding.weight.T
        logits[:, 0] = -1e9
        return logits


def train_sequences_from_interactions() -> list[list[int]]:
    train = pd.read_csv(TRAIN_INTERACTIONS)
    sequences = []
    for _, group in train.sort_values(["user_id", "timestamp", "movie_id"]).groupby("user_id", sort=False):
        seq = group.movie_id.astype(int).tolist()
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def train_sequences_from_samples(samples: pd.DataFrame) -> list[list[int]]:
    sequences = []
    for sample in samples[samples.split == "train"].itertuples():
        seq = parse_ids(sample.model_history_movie_ids) + [int(sample.target_movie_id)]
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def build_item_maps(samples: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    item_ids = set(pd.read_csv(POPULARITY).movie_id.astype(int).tolist())
    if TRAIN_INTERACTIONS.exists():
        item_ids.update(pd.read_csv(TRAIN_INTERACTIONS).movie_id.astype(int).tolist())
    train_samples = samples[samples.split == "train"]
    item_ids.update(train_samples.target_movie_id.astype(int).tolist())
    for value in train_samples.model_history_movie_ids.tolist():
        item_ids.update(parse_ids(value))
    item_to_idx = {movie_id: idx + 1 for idx, movie_id in enumerate(sorted(item_ids))}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    return item_to_idx, idx_to_item


def collate(batch):
    input_ids, labels, lengths = zip(*batch)
    return torch.stack(input_ids), torch.stack(labels), torch.stack(lengths)


def make_input(sample: pd.Series, item_to_idx: dict[int, int], max_seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    hist = [item_to_idx[mid] for mid in parse_ids(sample.model_history_movie_ids) if mid in item_to_idx][-max_seq_len:]
    if not hist:
        return None
    arr = np.zeros(max_seq_len, dtype=np.int64)
    arr[: len(hist)] = np.array(hist, dtype=np.int64)
    return (
        torch.tensor(arr, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor([len(hist)], dtype=torch.long, device=device),
    )


def build_candidates(
    model: SASRec,
    samples: pd.DataFrame,
    split: str,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    pop_ranked: list[tuple[int, float]],
    top_k: int,
    max_seq_len: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for n, (sample_id, sample) in enumerate(samples.iterrows(), start=1):
            seen = set(parse_ids(sample.seen_movie_ids))
            used = set()
            ranked: list[tuple[int, float]] = []
            model_input = make_input(sample, item_to_idx, max_seq_len, device)
            if model_input is not None:
                logits = model.score_last(*model_input)[0].float().detach().cpu().numpy()
                for movie_id in seen:
                    idx = item_to_idx.get(movie_id)
                    if idx is not None:
                        logits[idx] = -1e9
                logits[0] = -1e9
                take = min(len(logits) - 1, max(top_k * 4, top_k + len(seen) + 50))
                top_idx = np.argpartition(-logits, take)[:take]
                top_idx = top_idx[np.argsort(-logits[top_idx])]
                for idx in top_idx.tolist():
                    movie_id = idx_to_item.get(int(idx))
                    if movie_id is None or movie_id in seen or movie_id in used:
                        continue
                    ranked.append((movie_id, float(logits[idx])))
                    used.add(movie_id)
                    if len(ranked) >= top_k:
                        break
            for movie_id, pop_score in pop_ranked:
                if len(ranked) >= top_k:
                    break
                if movie_id in seen or movie_id in used:
                    continue
                ranked.append((int(movie_id), float(-1e6 + pop_score)))
                used.add(int(movie_id))
            for rank, (movie_id, score) in enumerate(ranked[:top_k], start=1):
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "user_id": int(sample.user_id),
                        "split": split,
                        "target_movie_id": int(sample.target_movie_id),
                        "candidate_movie_id": int(movie_id),
                        "retrieval_rank": rank,
                        "retrieval_score": float(score),
                        "method": "sasrec",
                    }
                )
            if n % 1000 == 0:
                print(f"sasrec candidates {split}: {n}/{len(samples)}", flush=True)
    return pd.DataFrame(rows)


def best_seq_name() -> str:
    path = V22_OUT / "best_seq_configs.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric in ["HR@1", "NDCG@30", "TargetIn@30", "HR@5"]:
            for row in (payload.get("best_configs") or {}).get(metric, []):
                config = str(row.get("config"))
                if (V22_OUT / f"candidates_{config}_val.csv").exists() and (V22_OUT / f"candidates_{config}_test.csv").exists():
                    return config
    return DEFAULT_SEQ


def load_best_seq(split: str, seq_name: str) -> pd.DataFrame:
    path = V22_OUT / f"candidates_{seq_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing best_seq candidates: {path}")
    return pd.read_csv(path)


def rrf_fusion(base: pd.DataFrame, sasrec: pd.DataFrame, split: str, config: str, sasrec_weight: float, top_k: int) -> pd.DataFrame:
    base_rank = base[["sample_id", "candidate_movie_id", "retrieval_rank"]].rename(columns={"retrieval_rank": "base_rank"})
    sas_rank = sasrec[["sample_id", "candidate_movie_id", "retrieval_rank"]].rename(columns={"retrieval_rank": "sasrec_rank"})
    merged = base_rank.merge(sas_rank, on=["sample_id", "candidate_movie_id"], how="outer")
    merged["retrieval_score"] = (1.0 / (60.0 + merged["base_rank"])).fillna(0.0) + sasrec_weight * (
        1.0 / (60.0 + merged["sasrec_rank"])
    ).fillna(0.0)
    meta = pd.concat(
        [
            base[["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id"]],
            sasrec[["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["sample_id", "candidate_movie_id"], keep="first")
    merged = merged.merge(meta, on=["sample_id", "candidate_movie_id"], how="left")
    rows = []
    for _, group in merged.groupby("sample_id", sort=False):
        ranked = group.sort_values(["retrieval_score", "candidate_movie_id"], ascending=[False, True]).head(top_k)
        for rank, item in enumerate(ranked.itertuples(index=False), start=1):
            row = item._asdict()
            row["retrieval_rank"] = rank
            row["method"] = config
            row["split"] = split
            rows.append(row)
    cols = ["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id", "retrieval_rank", "retrieval_score", "method"]
    return pd.DataFrame(rows)[cols]


def run_fusion(
    seq_name: str,
    sasrec_frames: dict[str, pd.DataFrame],
    samples_by_split: dict[str, pd.DataFrame],
    weights: list[float],
    top_k: int,
    top_n_test: int,
    out_dir: Path,
):
    val_rows = []
    config_lookup = {}
    for weight in weights:
        config = f"rrf_best_seq_sasrec_w{str(weight).replace('.', 'p')}"
        cand = rrf_fusion(load_best_seq("val", seq_name), sasrec_frames["val"], "val", config, weight, top_k)
        metrics = eval_candidates(cand, samples_by_split["val"])
        val_rows.append(
            {
                "version": "v10_sasrec_retrieval",
                "stage": "sasrec_fusion",
                "split": "val",
                "config": config,
                "best_seq_source": seq_name,
                "sasrec_weight": weight,
                **metrics,
            }
        )
        config_lookup[config] = weight
        cand.to_csv(out_dir / f"candidates_{config}_val.csv", index=False)
        print(f"val fusion {config} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    val_df = pd.DataFrame(val_rows)
    selected = set()
    best_configs = {}
    for metric in ["HR@1", "NDCG@30", "MRR@30", "TargetIn@30", "TargetIn@50"]:
        top = val_df.sort_values([metric, "HR@1", "NDCG@30"], ascending=False).head(top_n_test)
        best_configs[metric] = top.to_dict("records")
        selected.update(top["config"].tolist())
    test_rows = []
    for config in sorted(selected):
        weight = config_lookup[config]
        cand = rrf_fusion(load_best_seq("test", seq_name), sasrec_frames["test"], "test", config, weight, top_k)
        cand.to_csv(out_dir / f"candidates_{config}_test.csv", index=False)
        metrics = eval_candidates(cand, samples_by_split["test"])
        test_rows.append(
            {
                "version": "v10_sasrec_retrieval",
                "stage": "sasrec_fusion",
                "split": "test",
                "config": config,
                "best_seq_source": seq_name,
                "sasrec_weight": weight,
                **metrics,
            }
        )
        print(f"test fusion {config} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    return val_df, pd.DataFrame(test_rows), {"best_configs": best_configs, "selected_for_test": sorted(selected)}


def write_run_commands(out_dir: Path) -> None:
    (out_dir / "run_commands.txt").write_text(
        "SMOKE:\n"
        "python scripts/32_train_sasrec_candidates.py --max_epochs 2 --max_train_samples 50000 --patience 1 --top_k 200 --run_name sasrec_smoke\n\n"
        "NORMAL:\n"
        "python scripts/32_train_sasrec_candidates.py \\\n"
        "  --max_seq_len 50 \\\n"
        "  --embedding_dim 128 \\\n"
        "  --num_heads 4 \\\n"
        "  --num_layers 2 \\\n"
        "  --dropout 0.2 \\\n"
        "  --batch_size 256 \\\n"
        "  --max_epochs 10 \\\n"
        "  --patience 3 \\\n"
        "  --learning_rate 0.001 \\\n"
        "  --weight_decay 0.00001 \\\n"
        "  --early_metric NDCG@30 \\\n"
        "  --top_k 200 \\\n"
        "  --run_name len50_d128_l2_do0p2_lr0p001\n\n"
        "python scripts/19_collect_experiments_summary.py\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_metric", choices=["HR@1", "NDCG@30"], default="NDCG@30")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--sasrec_weights", default="0.25,0.5,1.0")
    parser.add_argument("--top_n_test", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(V10_OUT)
    args.run_name = args.run_name.strip() or default_run_name(args)
    out_dir = ensure_dir(V10_OUT / "runs" / args.run_name)
    write_run_commands(out_dir)
    write_json(out_dir / "run_config.json", vars(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = pd.read_csv(SAMPLES)
    val_samples = samples[samples.split == "val"]
    test_samples = samples[samples.split == "test"]
    samples_by_split = {"val": val_samples, "test": test_samples}
    item_to_idx, idx_to_item = build_item_maps(samples)
    sequences = train_sequences_from_interactions() if TRAIN_INTERACTIONS.exists() else train_sequences_from_samples(samples)
    dataset = SasRecSequenceDataset(sequences, item_to_idx, args.max_seq_len, args.max_train_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    model = SASRec(len(item_to_idx), args.max_seq_len, args.embedding_dim, args.num_heads, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    pop_ranked = top_popularity()
    print(
        f"sasrec sequence_examples={len(dataset)} items={len(item_to_idx)} max_seq_len={args.max_seq_len} "
        f"dim={args.embedding_dim} layers={args.num_layers} amp={args.amp} device={device}",
        flush=True,
    )

    best_metric = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch_idx, (input_ids, labels, _) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            tokens = int(labels.ne(0).sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            if batch_idx % 100 == 0:
                print(f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={loss.item():.6f}", flush=True)
        val_cand = build_candidates(model, val_samples, "val", item_to_idx, idx_to_item, pop_ranked, args.top_k, args.max_seq_len, device)
        val_metrics = eval_candidates(val_cand, val_samples)
        metric = float(val_metrics[args.early_metric])
        history.append({"epoch": epoch, "loss": total_loss / max(total_tokens, 1), **val_metrics})
        print(f"epoch={epoch} token_loss={history[-1]['loss']:.6f} val_{args.early_metric}={metric:.6f}", flush=True)
        if metric > best_metric:
            best_metric = metric
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping epoch={epoch} stale={stale} best_{args.early_metric}={best_metric:.6f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_to_idx": item_to_idx,
            "config": vars(args),
            "history": history,
        },
        out_dir / "sasrec_model.pt",
    )

    sasrec_frames = {}
    sasrec_rows = []
    for split, split_samples in [("val", val_samples), ("test", test_samples)]:
        cand = build_candidates(model, split_samples, split, item_to_idx, idx_to_item, pop_ranked, args.top_k, args.max_seq_len, device)
        sasrec_frames[split] = cand
        cand.to_csv(out_dir / f"candidates_sasrec_{split}.csv", index=False)
        metrics = eval_candidates(cand, split_samples)
        sasrec_rows.append(
            {
                "version": "v10_sasrec_retrieval",
                "stage": "sasrec",
                "split": split,
                "config": args.run_name,
                "run_name": args.run_name,
                **metrics,
            }
        )
        print(f"saved sasrec {split} HR@1={metrics['HR@1']:.6f} NDCG@30={metrics['NDCG@30']:.6f}", flush=True)
    sasrec_df = pd.DataFrame(sasrec_rows)
    sasrec_df[sasrec_df.split == "val"].to_csv(out_dir / "sasrec_val_metrics.csv", index=False)
    sasrec_df[sasrec_df.split == "test"].to_csv(out_dir / "sasrec_test_metrics.csv", index=False)

    seq_name = best_seq_name()
    weights = [float(x) for x in str(args.sasrec_weights).split(",") if x]
    fusion_val, fusion_test, selection = run_fusion(seq_name, sasrec_frames, samples_by_split, weights, args.top_k, args.top_n_test, out_dir)
    fusion_val["run_name"] = args.run_name
    fusion_val["config"] = fusion_val["config"].map(lambda config: f"{args.run_name}/{config}")
    fusion_test["run_name"] = args.run_name
    fusion_test["config"] = fusion_test["config"].map(lambda config: f"{args.run_name}/{config}")
    fusion_val.to_csv(out_dir / "fusion_val_metrics.csv", index=False)
    fusion_test.to_csv(out_dir / "fusion_test_metrics.csv", index=False)
    write_json(
        out_dir / "best_sasrec_configs.json",
        {
            "selection_split": "val",
            "run_name": args.run_name,
            "best_seq_source": seq_name,
            "sasrec_weights": weights,
            "top_k": args.top_k,
            "top_n_test": args.top_n_test,
            "history": history,
            **selection,
        },
    )
    print(f"saved v10 outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
