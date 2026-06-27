from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import OUT, POPULARITY, SAMPLES, TRAIN_INTERACTIONS, parse_ids, write_json


V10_OUT = OUT / "v10_sasrec_retrieval"


def load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SASREC_MOD = load_module("32_train_sasrec_candidates.py", "sasrec_train_for_sampled_blend")
CROSS_MOD = load_module("36_train_cross_attention_reranker.py", "crossattn_train_for_sampled_blend")


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def beta_token(value: float) -> str:
    return str(value).replace(".", "p")


def sasrec_run_dir(run_name: str) -> Path:
    path = V10_OUT / "runs" / run_name
    if not path.exists():
        raise FileNotFoundError(f"Missing SASRec run dir: {path}")
    return path


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.crossattn_dir)


def load_sasrec(args: argparse.Namespace, device: torch.device):
    model_path = sasrec_run_dir(args.sasrec_run_name) / "sasrec_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing SASRec checkpoint: {model_path}")
    payload = safe_torch_load(model_path, map_location=device)
    config = payload.get("config", {})
    item_to_idx = {int(k): int(v) for k, v in payload["item_to_idx"].items()}
    max_seq_len = int(config.get("max_seq_len", args.max_history_len))
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
    return model, item_to_idx, config, model_path


def load_crossattn(args: argparse.Namespace, device: torch.device):
    cross_dir = Path(args.crossattn_dir)
    model_path = cross_dir / "crossattn_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CrossAttention checkpoint: {model_path}")
    payload = safe_torch_load(model_path, map_location=device)
    item_to_idx = {int(k): int(v) for k, v in payload["item_to_idx"].items()}
    ckpt_args = payload.get("args", {})
    text_matrix, genre_matrix, pop_map, _, _ = CROSS_MOD.feature_matrices(item_to_idx)
    model = CROSS_MOD.CrossAttentionReranker(
        len(item_to_idx),
        text_matrix,
        genre_matrix,
        int(ckpt_args.get("max_history_len", args.max_history_len)),
        int(ckpt_args.get("embedding_dim", 128)),
        int(ckpt_args.get("text_projection_dim", 128)),
        int(ckpt_args.get("genre_projection_dim", 32)),
        int(ckpt_args.get("num_heads", 4)),
        float(ckpt_args.get("dropout", 0.2)),
        int(ckpt_args.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, item_to_idx, pop_map, ckpt_args, model_path


def all_movie_ids(samples: pd.DataFrame, sasrec_item_to_idx: dict[int, int], cross_item_to_idx: dict[int, int]) -> list[int]:
    movies = set(int(x) for x in sasrec_item_to_idx).intersection(set(int(x) for x in cross_item_to_idx))
    if POPULARITY.exists():
        movies.update(int(x) for x in pd.read_csv(POPULARITY).movie_id.astype(int).tolist() if int(x) in sasrec_item_to_idx and int(x) in cross_item_to_idx)
    if TRAIN_INTERACTIONS.exists():
        movies.update(int(x) for x in pd.read_csv(TRAIN_INTERACTIONS).movie_id.astype(int).tolist() if int(x) in sasrec_item_to_idx and int(x) in cross_item_to_idx)
    movies.update(int(x) for x in samples.target_movie_id.astype(int).tolist() if int(x) in sasrec_item_to_idx and int(x) in cross_item_to_idx)
    return sorted(movies)


def make_sasrec_input(sample: pd.Series, item_to_idx: dict[int, int], max_history_len: int, model_seq_len: int, device: torch.device):
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


def sample_negatives(rng: np.random.Generator, movie_pool: np.ndarray, forbidden: set[int], num_negatives: int) -> list[int] | None:
    allowed = [int(movie_id) for movie_id in movie_pool.tolist() if int(movie_id) not in forbidden]
    if len(allowed) < num_negatives:
        return None
    return [int(x) for x in rng.choice(np.array(allowed, dtype=np.int64), size=num_negatives, replace=False).tolist()]


def z_norm(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / max(float(values.std()), 1e-6)).astype(np.float32)


def rank_metrics(scores: np.ndarray, target_pos: int = 0) -> tuple[int, float, float, int]:
    order = np.argsort(-scores)
    rank = int(np.where(order == target_pos)[0][0]) + 1
    hit10 = int(rank <= 10)
    ndcg10 = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    mrr10 = 1.0 / rank if rank <= 10 else 0.0
    return hit10, ndcg10, mrr10, rank


def aggregate(metric_rows: list[tuple[int, float, float]]) -> dict:
    denom = max(len(metric_rows), 1)
    return {
        "Hit@10": float(sum(row[0] for row in metric_rows) / denom),
        "NDCG@10": float(sum(row[1] for row in metric_rows) / denom),
        "MRR@10": float(sum(row[2] for row in metric_rows) / denom),
    }


def crossattn_scores_for_sample(
    model,
    sample_id: int,
    sample: pd.Series,
    candidate_movie_ids: list[int],
    sasrec_scores: np.ndarray,
    item_to_idx: dict[int, int],
    pop_map: dict[int, float],
    max_history_len: int,
    device: torch.device,
) -> np.ndarray:
    ranks = np.arange(1, len(candidate_movie_ids) + 1, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "sample_id": [int(sample_id)] * len(candidate_movie_ids),
            "user_id": [int(sample.user_id)] * len(candidate_movie_ids),
            "split": [str(sample.split)] * len(candidate_movie_ids),
            "target_movie_id": [int(sample.target_movie_id)] * len(candidate_movie_ids),
            "candidate_movie_id": [int(x) for x in candidate_movie_ids],
            "retrieval_rank": ranks.astype(int),
            "retrieval_score": [float(x) for x in sasrec_scores.tolist()],
            "method": ["sampled_sasrec"] * len(candidate_movie_ids),
        }
    )
    sample_df = pd.DataFrame([sample]).set_index(pd.Index([int(sample_id)]))
    ds = CROSS_MOD.CrossAttentionDataset(
        frame,
        sample_df,
        item_to_idx,
        pop_map,
        len(candidate_movie_ids),
        max_history_len,
        require_label=False,
    )
    if len(ds) == 0:
        raise ValueError("CrossAttentionDataset produced no rows")
    batch = CROSS_MOD.collate([ds[0]])
    with torch.no_grad():
        scores = model(
            batch["history_ids"].to(device),
            batch["history_mask"].to(device),
            batch["candidate_ids"].to(device),
            batch["candidate_mask"].to(device),
            batch["numeric"].to(device),
        )
    return scores[0, : len(candidate_movie_ids)].float().detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_run_name", default="len200_d128_l2_do0p2_lr0p001")
    parser.add_argument("--crossattn_dir", default="outputs/v14_cross_attention_reranker_len200_beta0p5_fulltrain")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history_len", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="")
    parser.set_defaults(save_samples=True)
    parser.add_argument("--save_samples", dest="save_samples", action="store_true")
    parser.add_argument("--no_save_samples", dest="save_samples", action="store_false")
    parser.add_argument("--limit_samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    sasrec_model, sasrec_item_to_idx, sasrec_config, sasrec_ckpt = load_sasrec(args, device)
    cross_model, cross_item_to_idx, pop_map, cross_args, cross_ckpt = load_crossattn(args, device)
    samples = pd.read_csv(SAMPLES)
    split_samples = samples[samples.split == args.split].copy()
    if args.limit_samples > 0:
        split_samples = split_samples.head(args.limit_samples)
    movie_pool = np.array(all_movie_ids(samples, sasrec_item_to_idx, cross_item_to_idx), dtype=np.int64)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"sampled_{args.num_negatives}neg_{args.split}_seed{args.seed}_blend_score_beta{beta_token(args.beta)}"
    metrics_path = out_dir / f"{suffix}_metrics.json"
    items_path = out_dir / f"{suffix}_items.jsonl"
    sample_file = items_path.open("w", encoding="utf-8") if args.save_samples else None

    metric_rows = {"sasrec_only": [], "crossattn_only": [], "blend": []}
    evaluated = 0
    skipped = 0
    skipped_reasons = {
        "missing_target_mapping": 0,
        "empty_history": 0,
        "not_enough_negatives": 0,
        "crossattn_error": 0,
    }
    sasrec_seq_len = int(sasrec_config.get("max_seq_len", args.max_history_len))
    cross_history_len = int(cross_args.get("max_history_len", args.max_history_len))

    with torch.no_grad():
        for sample_id, sample in split_samples.iterrows():
            sample_id = int(sample_id)
            target = int(sample.target_movie_id)
            if target not in sasrec_item_to_idx or target not in cross_item_to_idx:
                skipped += 1
                skipped_reasons["missing_target_mapping"] += 1
                continue
            model_input = make_sasrec_input(sample, sasrec_item_to_idx, args.max_history_len, sasrec_seq_len, device)
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
            candidate_movie_ids = [target] + negatives
            missing = [mid for mid in candidate_movie_ids if mid not in sasrec_item_to_idx or mid not in cross_item_to_idx]
            if missing:
                skipped += 1
                skipped_reasons["missing_target_mapping"] += 1
                continue
            sasrec_indices = torch.tensor([sasrec_item_to_idx[mid] for mid in candidate_movie_ids], dtype=torch.long, device=device)
            sasrec_logits = sasrec_model.score_last(*model_input)[0]
            sasrec_scores = sasrec_logits[sasrec_indices].float().detach().cpu().numpy()
            sasrec_order = np.argsort(-sasrec_scores)
            ordered_movie_ids = [candidate_movie_ids[int(i)] for i in sasrec_order.tolist()]
            ordered_sasrec_scores = sasrec_scores[sasrec_order]
            try:
                cross_scores_ordered = crossattn_scores_for_sample(
                    cross_model,
                    sample_id,
                    sample,
                    ordered_movie_ids,
                    ordered_sasrec_scores,
                    cross_item_to_idx,
                    pop_map,
                    cross_history_len,
                    device,
                )
            except Exception:
                skipped += 1
                skipped_reasons["crossattn_error"] += 1
                continue
            restore = np.empty_like(sasrec_order)
            restore[sasrec_order] = np.arange(len(sasrec_order))
            cross_scores = cross_scores_ordered[restore]
            blend_scores = z_norm(sasrec_scores) + args.beta * z_norm(cross_scores)

            sasrec_hit, sasrec_ndcg, sasrec_mrr, sasrec_rank = rank_metrics(sasrec_scores)
            cross_hit, cross_ndcg, cross_mrr, cross_rank = rank_metrics(cross_scores)
            blend_hit, blend_ndcg, blend_mrr, blend_rank = rank_metrics(blend_scores)
            metric_rows["sasrec_only"].append((sasrec_hit, sasrec_ndcg, sasrec_mrr))
            metric_rows["crossattn_only"].append((cross_hit, cross_ndcg, cross_mrr))
            metric_rows["blend"].append((blend_hit, blend_ndcg, blend_mrr))
            evaluated += 1
            if sample_file is not None:
                sample_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "user_id": int(sample.user_id),
                            "target_movie_id": target,
                            "negative_movie_ids": negatives,
                            "scored_movie_ids": candidate_movie_ids,
                            "sasrec_scores": [float(x) for x in sasrec_scores.tolist()],
                            "crossattn_scores": [float(x) for x in cross_scores.tolist()],
                            "blend_scores": [float(x) for x in blend_scores.tolist()],
                            "sasrec_target_rank": int(sasrec_rank),
                            "crossattn_target_rank": int(cross_rank),
                            "blend_target_rank": int(blend_rank),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if evaluated % 1000 == 0:
                print(f"sampled crossattn blend {args.split}: evaluated={evaluated} skipped={skipped}", flush=True)
    if sample_file is not None:
        sample_file.close()

    blend_metrics = aggregate(metric_rows["blend"])
    metrics = {
        "sasrec_run_name": args.sasrec_run_name,
        "crossattn_dir": str(args.crossattn_dir),
        "split": args.split,
        "protocol": "sampled_100_negative_crossattn_blend" if args.num_negatives == 100 else f"sampled_{args.num_negatives}_negative_crossattn_blend",
        "num_negatives": int(args.num_negatives),
        "seed": int(args.seed),
        "beta": float(args.beta),
        "max_history_len": int(args.max_history_len),
        "evaluated_samples": int(evaluated),
        "skipped_samples": int(skipped),
        "skipped_reasons": skipped_reasons,
        "sasrec_checkpoint": str(sasrec_ckpt),
        "crossattn_checkpoint": str(cross_ckpt),
        "items_path": str(items_path) if args.save_samples else "",
        "sasrec_only": aggregate(metric_rows["sasrec_only"]),
        "crossattn_only": aggregate(metric_rows["crossattn_only"]),
        "blend": blend_metrics,
        **blend_metrics,
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
