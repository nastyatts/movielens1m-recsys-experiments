from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import EMB, MOVIE_ORDER, OUT, SAMPLES, parse_ids, write_json


LLM_MOVIE_TOKEN_DATA = Path(__file__).resolve().parents[1] / "data" / "llm_movie_tokens"
MOVIE_TOKEN_OUT = OUT / "movie_token_lm"


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row["sample_id"])] = row
    return rows


def load_sampled_items(path: str) -> dict[int, dict]:
    if not path:
        return {}
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row["sample_id"])] = {
                    "sample_id": int(row["sample_id"]),
                    "user_id": int(row.get("user_id", 0)),
                    "target_movie_id": int(row["target_movie_id"]),
                    "negative_movie_ids": [int(x) for x in row["negative_movie_ids"]],
                    "scored_movie_ids": [int(x) for x in row.get("scored_movie_ids", [row["target_movie_id"]] + row["negative_movie_ids"])],
                }
    return rows


def load_base_model(model_name: str, tokenizer_len: int):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    model.resize_token_embeddings(tokenizer_len)
    return model


def load_movie_token_rows(model, run_dir: Path) -> None:
    path = run_dir / "movie_token_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing movie token embedding rows: {path}")
    payload = safe_torch_load(path, map_location="cpu")
    token_ids = torch.tensor(payload["token_ids"], dtype=torch.long, device=model.get_input_embeddings().weight.device)
    with torch.no_grad():
        input_rows = payload["input_embeddings"].to(model.get_input_embeddings().weight.device, dtype=model.get_input_embeddings().weight.dtype)
        model.get_input_embeddings().weight[token_ids] = input_rows
        output_emb = model.get_output_embeddings()
        if output_emb is not None and "output_embeddings" in payload:
            output_rows = payload["output_embeddings"].to(output_emb.weight.device, dtype=output_emb.weight.dtype)
            output_emb.weight[token_ids] = output_rows


def movie_text_vectors_for_tokens(token_map: dict) -> tuple[list[int], torch.Tensor]:
    emb = np.load(EMB).astype(np.float32)
    order = np.load(MOVIE_ORDER).astype(int).tolist()
    pos = {int(movie_id): i for i, movie_id in enumerate(order)}
    token_ids = []
    vectors = []
    for movie_id_str, token_id in token_map["movie_id_to_token_id"].items():
        token_ids.append(int(token_id))
        movie_id = int(movie_id_str)
        if movie_id in pos:
            vectors.append(emb[pos[movie_id]])
        else:
            vectors.append(np.zeros(emb.shape[1], dtype=np.float32))
    return token_ids, torch.tensor(np.stack(vectors), dtype=torch.float32)


def load_projection(run_dir: Path, hidden_size: int, device: torch.device) -> nn.Linear:
    path = run_dir / "projection.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing projection.pt: {path}")
    state = safe_torch_load(path, map_location=device)
    input_dim = int(state["weight"].shape[1])
    projection = nn.Linear(input_dim, hidden_size).to(device)
    projection.load_state_dict(state)
    projection.eval()
    return projection


def load_movie_output_head(run_dir: Path, hidden_size: int, num_movies: int, device: torch.device) -> nn.Linear:
    path = run_dir / "movie_output_head.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing movie_output_head.pt: {path}")
    head = nn.Linear(hidden_size, num_movies, bias=False).to(device)
    head.load_state_dict(safe_torch_load(path, map_location=device))
    head.eval()
    return head


def sync_input_embeddings_from_projection(model, projection: nn.Linear, movie_token_ids: list[int], movie_text_vectors: torch.Tensor) -> None:
    input_emb = model.get_input_embeddings()
    device = input_emb.weight.device
    token_ids_tensor = torch.tensor(movie_token_ids, dtype=torch.long, device=device)
    if movie_text_vectors.shape[1] != projection.weight.shape[1]:
        raise ValueError(
            f"Projection input dim mismatch: vectors={movie_text_vectors.shape[1]} projection={projection.weight.shape[1]}"
        )
    with torch.no_grad():
        projected = projection(movie_text_vectors.to(projection.weight.device))
        input_emb.weight[token_ids_tensor] = projected.to(device=device, dtype=input_emb.weight.dtype)


def token_embeds_with_projection(model, input_ids: torch.Tensor, token_to_movie_pos: torch.Tensor, projection: nn.Linear, movie_text_vectors: torch.Tensor) -> torch.Tensor:
    token_embeds = model.get_input_embeddings()(input_ids)
    token_pos = token_to_movie_pos[input_ids]
    mask = token_pos.ge(0)
    if mask.any():
        unique_pos = torch.unique(token_pos[mask])
        projected = projection(movie_text_vectors[unique_pos].to(token_embeds.device)).to(dtype=token_embeds.dtype)
        projected_by_pos = torch.zeros(
            (movie_text_vectors.shape[0], token_embeds.shape[-1]),
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )
        projected_by_pos[unique_pos] = projected
        token_embeds = token_embeds.clone()
        token_embeds[mask] = projected_by_pos[token_pos[mask]]
    return token_embeds


def backbone_forward(model, token_embeds: torch.Tensor, attention_mask: torch.Tensor):
    base_model = model.base_model.model
    if hasattr(base_model, "model"):
        return base_model.model(inputs_embeds=token_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
    if hasattr(base_model, "transformer"):
        return base_model.transformer(inputs_embeds=token_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
    outputs = model(
        inputs_embeds=token_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return type("BackboneOutput", (), {"last_hidden_state": outputs.hidden_states[-1]})()


def generate_sampled_items(samples: pd.DataFrame, rows_by_sample: dict[int, dict], token_map: dict, split: str, num_negatives: int, seed: int) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    movie_pool = np.array(sorted(int(x) for x in token_map["movie_id_to_token_id"].keys()), dtype=np.int64)
    out = {}
    for sample_id, sample in samples[samples.split == split].iterrows():
        sample_id = int(sample_id)
        if sample_id not in rows_by_sample:
            continue
        target = int(sample.target_movie_id)
        forbidden = set(parse_ids(sample.seen_movie_ids))
        forbidden.add(target)
        allowed = [int(movie_id) for movie_id in movie_pool.tolist() if int(movie_id) not in forbidden]
        if len(allowed) < num_negatives:
            continue
        negatives = [int(x) for x in rng.choice(np.array(allowed, dtype=np.int64), size=num_negatives, replace=False).tolist()]
        out[sample_id] = {
            "sample_id": sample_id,
            "user_id": int(sample.user_id),
            "target_movie_id": target,
            "negative_movie_ids": negatives,
            "scored_movie_ids": [target] + negatives,
        }
    return out


def dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampled_items_path", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) if args.output_dir else MOVIE_TOKEN_OUT / "runs" / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing movie-token LM run dir: {run_dir}")
    token_map = json.loads((run_dir / "movie_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(args.model_name, len(tokenizer))
    load_movie_token_rows(model, run_dir)
    model = PeftModel.from_pretrained(model, str(run_dir))
    model.eval()

    rows_by_sample = load_rows(LLM_MOVIE_TOKEN_DATA / f"{args.split}_movie_token_lm.jsonl")
    samples = pd.read_csv(SAMPLES)
    if args.sampled_items_path:
        sampled = load_sampled_items(args.sampled_items_path)
    else:
        sampled = generate_sampled_items(samples, rows_by_sample, token_map, args.split, args.num_negatives, args.seed)

    movie_ids = [int(x) for x in token_map["movie_id_to_token_id"].keys()]
    movie_token_id_list = [int(token_map["movie_id_to_token_id"][str(mid)]) for mid in movie_ids]
    movie_pos_by_id = {movie_id: pos for pos, movie_id in enumerate(movie_ids)}
    movie_token_ids = torch.tensor(movie_token_id_list, dtype=torch.long, device=model.device)
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    _, movie_text_vectors = movie_text_vectors_for_tokens(token_map)
    projection = load_projection(run_dir, hidden_size, model.device)
    movie_output_head = load_movie_output_head(run_dir, hidden_size, len(movie_ids), model.device)
    sync_input_embeddings_from_projection(model, projection, movie_token_id_list, movie_text_vectors)
    vocab_size = model.get_input_embeddings().weight.shape[0]
    token_to_movie_pos = torch.full((vocab_size,), -1, dtype=torch.long, device=model.device)
    token_to_movie_pos[movie_token_ids] = torch.arange(len(movie_token_id_list), dtype=torch.long, device=model.device)
    movie_text_vectors = movie_text_vectors.to(model.device)
    movie_output_head = movie_output_head.to(model.device)

    candidate_items = [row for sample_id, row in sorted(sampled.items()) if sample_id in rows_by_sample]
    predictions_path = run_dir / f"sampled_{args.num_negatives}neg_{args.split}_predictions_seed{args.seed}.jsonl"
    metrics_path = run_dir / f"sampled_{args.num_negatives}neg_{args.split}_seed{args.seed}_metrics.json"
    evaluated = 0
    skipped = 0
    hit10 = 0
    ndcg10 = 0.0
    mrr10 = 0.0
    with predictions_path.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for start in range(0, len(candidate_items), args.batch_size):
            batch_items = candidate_items[start : start + args.batch_size]
            prompts = [rows_by_sample[int(row["sample_id"])]["input_text"] for row in batch_items]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            last_pos = inputs["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(inputs["input_ids"].shape[0], device=model.device)
            token_embeds = token_embeds_with_projection(model, inputs["input_ids"], token_to_movie_pos, projection, movie_text_vectors)
            output = backbone_forward(model, token_embeds, inputs["attention_mask"])
            last_hidden = output.last_hidden_state[batch_idx, last_pos]
            movie_scores = movie_output_head(last_hidden.to(movie_output_head.weight.dtype)).float().detach().cpu()
            for row_idx, item_row in enumerate(batch_items):
                target = int(item_row["target_movie_id"])
                scored_movie_ids = [int(x) for x in item_row.get("scored_movie_ids", [target] + item_row["negative_movie_ids"])]
                if target not in scored_movie_ids:
                    scored_movie_ids = [target] + scored_movie_ids
                missing = [movie_id for movie_id in scored_movie_ids if movie_id not in movie_pos_by_id]
                if missing:
                    skipped += 1
                    continue
                positions = [movie_pos_by_id[movie_id] for movie_id in scored_movie_ids]
                scores = movie_scores[row_idx, positions].numpy()
                order = np.argsort(-scores)
                target_local_pos = scored_movie_ids.index(target)
                target_rank = int(np.where(order == target_local_pos)[0][0]) + 1
                evaluated += 1
                hit10 += int(target_rank <= 10)
                if target_rank <= 10:
                    ndcg10 += dcg(target_rank)
                    mrr10 += 1.0 / target_rank
                f_out.write(
                    json.dumps(
                        {
                            "sample_id": int(item_row["sample_id"]),
                            "user_id": int(item_row.get("user_id", rows_by_sample[int(item_row["sample_id"])]["user_id"])),
                            "target_movie_id": target,
                            "negative_movie_ids": [int(x) for x in item_row["negative_movie_ids"]],
                            "scored_movie_ids": scored_movie_ids,
                            "scores": [float(x) for x in scores.tolist()],
                            "target_rank": target_rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if start and start % (args.batch_size * 50) == 0:
                print(f"movie-token sampled eval {args.split}: {start}/{len(candidate_items)}", flush=True)

    denom = max(evaluated, 1)
    metrics = {
        "run_name": args.run_name,
        "split": args.split,
        "protocol": "sampled_100_negative" if args.num_negatives == 100 else f"sampled_{args.num_negatives}_negative",
        "num_negatives": int(args.num_negatives),
        "seed": int(args.seed),
        "model_name": args.model_name,
        "evaluated_samples": int(evaluated),
        "skipped_samples": int(skipped + (len(sampled) - len(candidate_items))),
        "Hit@10": float(hit10 / denom),
        "NDCG@10": float(ndcg10 / denom),
        "MRR@10": float(mrr10 / denom),
        "ValidOutputRate": 1.0 if evaluated > 0 else 0.0,
        "sampled_items_path": args.sampled_items_path,
        "predictions_path": str(predictions_path),
        "scoring_head": "movie_output_head",
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
