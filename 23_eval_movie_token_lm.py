from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import EMB, MOVIE_ORDER, OUT, SAMPLES, parse_ids, write_json


LLM_MOVIE_TOKEN_DATA = Path(__file__).resolve().parents[1] / "data" / "llm_movie_tokens"
MOVIE_TOKEN_OUT = OUT / "movie_token_lm"


def load_rows(path: Path, max_samples: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def load_candidate_lookup(path: str) -> dict[int, set[int]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    lookup: dict[int, set[int]] = defaultdict(set)
    for row in df.itertuples(index=False):
        lookup[int(row.sample_id)].add(int(row.candidate_movie_id))
    return lookup


def load_base_model(model_name: str, load_in_4bit: bool, tokenizer_len: int):
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    model.resize_token_embeddings(tokenizer_len)
    return model


def load_movie_token_rows(model, run_dir: Path) -> None:
    path = run_dir / "movie_token_embeddings.pt"
    if not path.exists():
        return
    payload = torch.load(path, map_location="cpu")
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


def load_projection(run_dir: Path, text_dim: int, hidden_size: int, device: torch.device) -> nn.Linear | None:
    path = run_dir / "projection.pt"
    if not path.exists():
        return None
    projection = nn.Linear(text_dim, hidden_size).to(device)
    projection.load_state_dict(torch.load(path, map_location=device))
    projection.eval()
    return projection


def load_movie_output_head(run_dir: Path, hidden_size: int, num_movies: int, device: torch.device) -> nn.Linear | None:
    path = run_dir / "movie_output_head.pt"
    if not path.exists():
        return None
    head = nn.Linear(hidden_size, num_movies, bias=False).to(device)
    head.load_state_dict(torch.load(path, map_location=device))
    head.eval()
    return head


def sync_input_embeddings_from_projection(model, projection: nn.Linear, movie_token_ids: list[int], movie_text_vectors: torch.Tensor) -> None:
    input_emb = model.get_input_embeddings()
    device = input_emb.weight.device
    token_ids_tensor = torch.tensor(movie_token_ids, dtype=torch.long, device=device)
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


def dcg_at_rank(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def compute_metrics(ranks: list[int | None], total: int) -> dict:
    hits = {k: 0 for k in [1, 5, 10, 30, 50, 100, 200]}
    ndcg30 = 0.0
    mrr30 = 0.0
    for rank in ranks:
        if rank is None:
            continue
        for k in hits:
            hits[k] += int(rank <= k)
        if rank <= 30:
            ndcg30 += dcg_at_rank(rank)
            mrr30 += 1.0 / rank
    denom = max(total, 1)
    return {
        "HR@1": hits[1] / denom,
        "HR@5": hits[5] / denom,
        "HR@10": hits[10] / denom,
        "TargetIn@30": hits[30] / denom,
        "TargetIn@50": hits[50] / denom,
        "TargetIn@100": hits[100] / denom,
        "TargetIn@200": hits[200] / denom,
        "NDCG@30": ndcg30 / denom,
        "MRR@30": mrr30 / denom,
        "ValidOutputRate": 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--candidate_file", default="")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.adapter_dir) if args.adapter_dir else MOVIE_TOKEN_OUT / "runs" / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")
    token_map = json.loads((run_dir / "movie_token_map.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(args.model_name, args.load_in_4bit, len(tokenizer))
    load_movie_token_rows(model, run_dir)
    model = PeftModel.from_pretrained(model, str(run_dir))
    model.eval()

    rows = load_rows(LLM_MOVIE_TOKEN_DATA / f"{args.split}_movie_token_lm.jsonl", args.max_samples)
    samples = pd.read_csv(SAMPLES)
    candidate_lookup = load_candidate_lookup(args.candidate_file)
    movie_ids = [int(x) for x in token_map["movie_id_to_token_id"].keys()]
    movie_token_id_list = [int(token_map["movie_id_to_token_id"][str(mid)]) for mid in movie_ids]
    movie_token_ids = torch.tensor(movie_token_id_list, dtype=torch.long, device=model.device)
    movie_id_by_pos = {pos: movie_id for pos, movie_id in enumerate(movie_ids)}
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    _, movie_text_vectors = movie_text_vectors_for_tokens(token_map)
    projection = load_projection(run_dir, int(movie_text_vectors.shape[1]), hidden_size, model.device)
    movie_output_head = load_movie_output_head(run_dir, hidden_size, len(movie_ids), model.device)
    scoring_head = "movie_output_head" if projection is not None and movie_output_head is not None else "lm_head"
    token_to_movie_pos = None
    if scoring_head == "movie_output_head":
        sync_input_embeddings_from_projection(model, projection, movie_token_id_list, movie_text_vectors)
        vocab_size = model.get_input_embeddings().weight.shape[0]
        token_to_movie_pos = torch.full((vocab_size,), -1, dtype=torch.long, device=model.device)
        token_to_movie_pos[movie_token_ids] = torch.arange(len(movie_token_id_list), dtype=torch.long, device=model.device)
        movie_text_vectors = movie_text_vectors.to(model.device)
        movie_output_head = movie_output_head.to(model.device)

    ranks: list[int | None] = []
    candidate_rows = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompts = [row["input_text"] for row in batch_rows]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            last_pos = inputs["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(inputs["input_ids"].shape[0], device=model.device)
            if scoring_head == "movie_output_head":
                token_embeds = token_embeds_with_projection(model, inputs["input_ids"], token_to_movie_pos, projection, movie_text_vectors)
                output = backbone_forward(model, token_embeds, inputs["attention_mask"])
                last_hidden = output.last_hidden_state[batch_idx, last_pos]
                movie_scores = movie_output_head(last_hidden.to(movie_output_head.weight.dtype)).float().detach().cpu()
            else:
                output = model(**inputs)
                logits = output.logits[batch_idx, last_pos]
                movie_scores = logits[:, movie_token_ids].float().detach().cpu()
            for row_idx, row in enumerate(batch_rows):
                sample_id = int(row["sample_id"])
                sample = samples.loc[sample_id]
                scores = movie_scores[row_idx].clone()
                seen = set(parse_ids(sample.seen_movie_ids))
                allowed = candidate_lookup.get(sample_id, set()) if args.candidate_file else None
                valid_positions = []
                for pos, movie_id in movie_id_by_pos.items():
                    if movie_id in seen:
                        continue
                    if allowed is not None and movie_id not in allowed:
                        continue
                    valid_positions.append(pos)
                target = int(row["target_movie_id"])
                if not valid_positions:
                    ranks.append(None)
                    continue
                valid_scores = scores[valid_positions]
                sorted_pos = torch.argsort(valid_scores, descending=True)
                rank = None
                for rank_idx, local_pos_tensor in enumerate(sorted_pos[:200].tolist(), start=1):
                    pos = valid_positions[int(local_pos_tensor)]
                    movie_id = movie_id_by_pos[pos]
                    score = float(scores[pos])
                    candidate_rows.append(
                        {
                            "sample_id": sample_id,
                            "user_id": int(row["user_id"]),
                            "split": args.split,
                            "target_movie_id": target,
                            "candidate_movie_id": movie_id,
                            "retrieval_rank": rank_idx,
                            "retrieval_score": score,
                            "method": "movie_token_lm_full_catalog" if not args.candidate_file else "movie_token_lm_restricted",
                        }
                    )
                    if movie_id == target:
                        rank = rank_idx
                ranks.append(rank)
            if start and start % (args.batch_size * 50) == 0:
                print(f"evaluated {start}/{len(rows)}", flush=True)

    metrics = compute_metrics(ranks, len(rows))
    metrics.update(
        {
            "version": "movie_token_lm",
            "stage": "next_movie_token_lm",
            "split": args.split,
            "run_name": args.run_name,
            "model_name": args.model_name,
            "evaluated_samples": len(rows),
            "candidate_file": args.candidate_file,
            "mode": "restricted" if args.candidate_file else "full_catalog",
            "scoring_head": scoring_head,
        }
    )
    write_json(run_dir / f"eval_{args.split}_metrics.json", metrics)
    pd.DataFrame(candidate_rows).to_csv(run_dir / f"candidates_movie_token_lm_{args.split}.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
