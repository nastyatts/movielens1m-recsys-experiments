from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import LLM_DATA, MOVIE_METADATA, SAMPLES, parse_ids, write_json


LLM_MOVIE_TOKEN_DATA = Path(__file__).resolve().parents[1] / "data" / "llm_movie_tokens"
MOVIE_TOKEN_MAP = LLM_DATA / "movie_token_map.json"


def movie_token(movie_id: int) -> str:
    return f"<movie_{int(movie_id)}>"


def collect_movie_ids(samples: pd.DataFrame) -> list[int]:
    movie_ids = set()
    if MOVIE_METADATA.exists():
        movie_ids.update(pd.read_csv(MOVIE_METADATA).movie_id.astype(int).tolist())
    movie_ids.update(samples.target_movie_id.astype(int).tolist())
    for value in samples.model_history_movie_ids.tolist():
        movie_ids.update(parse_ids(value))
    return sorted(int(x) for x in movie_ids)


def build_movie_token_map(model_name: str, movie_ids: list[int]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokens = [movie_token(movie_id) for movie_id in movie_ids]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    movie_id_to_token = {str(movie_id): movie_token(movie_id) for movie_id in movie_ids}
    movie_id_to_token_id = {str(movie_id): int(tokenizer.convert_tokens_to_ids(movie_token(movie_id))) for movie_id in movie_ids}
    token_id_to_movie_id = {str(token_id): int(movie_id) for movie_id, token_id in movie_id_to_token_id.items()}
    return {
        "model_name": model_name,
        "num_movies": len(movie_ids),
        "movie_id_to_token": movie_id_to_token,
        "movie_id_to_token_id": movie_id_to_token_id,
        "token_id_to_movie_id": token_id_to_movie_id,
    }


def write_split(samples: pd.DataFrame, split: str, token_map: dict, max_history_len: int, max_samples: int) -> dict:
    split_samples = samples[samples.split == split]
    if max_samples > 0 and len(split_samples) > max_samples:
        split_samples = split_samples.sample(max_samples, random_state=42).sort_index()
    out_path = LLM_MOVIE_TOKEN_DATA / f"{split}_movie_token_lm.jsonl"
    count = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample_id, sample in split_samples.iterrows():
            target = int(sample.target_movie_id)
            target_token = token_map["movie_id_to_token"].get(str(target))
            if target_token is None:
                skipped += 1
                continue
            history = [mid for mid in parse_ids(sample.model_history_movie_ids)[-max_history_len:] if str(mid) in token_map["movie_id_to_token"]]
            history_tokens = [token_map["movie_id_to_token"][str(mid)] for mid in history]
            input_text = "User history: " + " ".join(history_tokens) + "\nNext movie: "
            target_text = target_token
            record = {
                "sample_id": int(sample_id),
                "split": split,
                "user_id": int(sample.user_id),
                "target_movie_id": target,
                "history_movie_ids": [int(x) for x in history],
                "input_text": input_text,
                "target_text": target_text,
                "full_text": input_text + target_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return {"path": str(out_path), "rows": count, "skipped": skipped, "split_samples": int(len(split_samples))}


def write_run_commands() -> None:
    (LLM_MOVIE_TOKEN_DATA / "run_commands.txt").write_text(
        "BUILD DATA:\n"
        "python scripts/40_build_movie_token_lm_data.py --model_name meta-llama/Llama-3.2-3B-Instruct --max_history_len 50\n\n"
        "SMOKE TRAIN:\n"
        "python scripts/41_train_movie_token_lm_qlora.py --model_name meta-llama/Llama-3.2-1B-Instruct --run_name smoke_1b --max_train_samples 1000 --max_val_samples 300 --max_steps 100\n\n"
        "SMOKE EVAL:\n"
        "python scripts/42_eval_movie_token_lm.py --model_name meta-llama/Llama-3.2-1B-Instruct --run_name smoke_1b --split val --max_samples 500\n\n"
        "NORMAL TRAIN:\n"
        "python scripts/41_train_movie_token_lm_qlora.py --model_name meta-llama/Llama-3.2-3B-Instruct --run_name llama32_3b_steps300 --max_train_samples 10000 --max_val_samples 500 --max_steps 300\n\n"
        "NORMAL EVAL:\n"
        "python scripts/42_eval_movie_token_lm.py --model_name meta-llama/Llama-3.2-3B-Instruct --run_name llama32_3b_steps300 --split test\n",
        encoding="utf-8",
    )


def print_sanity(samples: pd.DataFrame, token_map: dict, model_name: str, max_history_len: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(token_map["movie_id_to_token"].values())})
    movie_token_ids = set(int(x) for x in token_map["movie_id_to_token_id"].values())
    print("SANITY movie-token LM rows:", flush=True)
    for sample_id, sample in samples[samples.split == "train"].head(3).iterrows():
        history = [mid for mid in parse_ids(sample.model_history_movie_ids)[-max_history_len:] if str(mid) in token_map["movie_id_to_token"]]
        history_tokens = [token_map["movie_id_to_token"][str(mid)] for mid in history]
        input_text = "User history: " + " ".join(history_tokens) + "\nNext movie: "
        target_text = token_map["movie_id_to_token"][str(int(sample.target_movie_id))]
        ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        print(
            json.dumps(
                {
                    "sample_id": int(sample_id),
                    "input_text": input_text,
                    "target_text": target_text,
                    "target_token_ids": ids,
                    "is_single_movie_token": len(ids) == 1 and int(ids[0]) in movie_token_ids,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    args = parser.parse_args()

    LLM_MOVIE_TOKEN_DATA.mkdir(parents=True, exist_ok=True)
    LLM_DATA.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SAMPLES)
    movie_ids = collect_movie_ids(samples)
    token_map = build_movie_token_map(args.model_name, movie_ids)
    write_json(MOVIE_TOKEN_MAP, token_map)
    stats = {
        "model_name": args.model_name,
        "max_history_len": int(args.max_history_len),
        "movie_token_map": str(MOVIE_TOKEN_MAP),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        stats["splits"][split] = write_split(samples, split, token_map, args.max_history_len, args.max_samples_per_split)
    write_json(LLM_MOVIE_TOKEN_DATA / "movie_token_lm_data_stats.json", stats)
    write_run_commands()
    print_sanity(samples, token_map, args.model_name, args.max_history_len)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
