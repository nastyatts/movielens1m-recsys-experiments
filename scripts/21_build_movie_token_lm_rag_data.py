from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import LLM_DATA, MOVIE_METADATA, OUT, SAMPLES, parse_ids, write_json


LLM_MOVIE_TOKEN_DATA = Path(__file__).resolve().parents[1] / "data" / "llm_movie_tokens"
MOVIE_TOKEN_MAP = LLM_DATA / "movie_token_map.json"


def movie_token(movie_id: int) -> str:
    return f"<movie_{int(movie_id)}>"


def collect_movie_ids(samples: pd.DataFrame, metadata: pd.DataFrame) -> list[int]:
    movie_ids = set()
    if not metadata.empty and "movie_id" in metadata.columns:
        movie_ids.update(metadata.movie_id.astype(int).tolist())
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


def load_or_build_token_map(model_name: str, samples: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    if MOVIE_TOKEN_MAP.exists():
        return json.loads(MOVIE_TOKEN_MAP.read_text(encoding="utf-8"))
    movie_ids = collect_movie_ids(samples, metadata)
    token_map = build_movie_token_map(model_name, movie_ids)
    LLM_DATA.mkdir(parents=True, exist_ok=True)
    write_json(MOVIE_TOKEN_MAP, token_map)
    return token_map


def load_metadata() -> dict[int, dict]:
    if not MOVIE_METADATA.exists():
        return {}
    df = pd.read_csv(MOVIE_METADATA)
    if "movie_id" not in df.columns:
        return {}
    return {int(row.movie_id): row._asdict() for row in df.itertuples(index=False)}


def clean_text(value, max_chars: int = 140) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def extract_year(meta: dict) -> str:
    year = clean_text(meta.get("year", ""), 16)
    if year:
        return year
    title = clean_text(meta.get("title", ""), 240)
    match = re.search(r"\((\d{4})\)", title)
    return match.group(1) if match else ""


def movie_line(movie_id: int, token_map: dict, metadata: dict[int, dict], rank: int | None = None, score: float | None = None) -> str:
    token = token_map["movie_id_to_token"].get(str(int(movie_id)), movie_token(movie_id))
    meta = metadata.get(int(movie_id), {})
    title = clean_text(meta.get("title", f"Movie {movie_id}"), 120)
    year = extract_year(meta)
    genres = clean_text(meta.get("genres", ""), 120)
    parts = [f"{token} Title: {title}"]
    if year:
        parts.append(f"Year: {year}")
    if genres:
        parts.append(f"Genres: {genres}")
    if "avg_rating" in meta and clean_text(meta.get("avg_rating", ""), 24):
        parts.append(f"AvgRating: {clean_text(meta.get('avg_rating'), 24)}")
    if rank is not None:
        parts.append(f"Rank: {int(rank)}")
    if score is not None:
        parts.append(f"Score: {float(score):.6g}")
    return " | ".join(parts)


def candidate_score(row) -> float:
    if hasattr(row, "retrieval_score") and pd.notna(row.retrieval_score):
        return float(row.retrieval_score)
    if hasattr(row, "score") and pd.notna(row.score):
        return float(row.score)
    rank = int(getattr(row, "retrieval_rank", getattr(row, "rank", 1)))
    return float(1.0 / max(rank, 1))


def candidate_rank(row, fallback_rank: int) -> int:
    for col in ["retrieval_rank", "rank", "rerank_rank"]:
        if hasattr(row, col):
            value = getattr(row, col)
            if pd.notna(value):
                return int(value)
    return int(fallback_rank)


def load_candidates(path: Path, max_retrieved_items: int) -> dict[int, list[dict]]:
    if not path.exists():
        print(f"WARNING: missing candidate file, retrieved context will be empty: {path}", flush=True)
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "candidate_movie_id" not in df.columns:
        print(f"WARNING: candidate file has incompatible columns, skipped: {path}", flush=True)
        return {}
    rank_col = next((col for col in ["retrieval_rank", "rank", "rerank_rank"] if col in df.columns), None)
    sort_cols = ["sample_id", "candidate_movie_id"]
    if rank_col:
        sort_cols = ["sample_id", rank_col, "candidate_movie_id"]
    df = df.sort_values(sort_cols)
    out: dict[int, list[dict]] = {}
    for sample_id, group in df.groupby("sample_id", sort=False):
        rows = []
        for fallback_rank, row in enumerate(group.head(max_retrieved_items).itertuples(index=False), start=1):
            rows.append(
                {
                    "candidate_movie_id": int(row.candidate_movie_id),
                    "rank": candidate_rank(row, fallback_rank),
                    "score": candidate_score(row),
                }
            )
        out[int(sample_id)] = rows
    return out


def build_prompt(
    sample,
    sample_id: int,
    token_map: dict,
    metadata: dict[int, dict],
    candidates_by_sample: dict[int, list[dict]],
    max_history_items: int,
    max_retrieved_items: int,
) -> tuple[str, list[int], list[int]]:
    target = int(sample.target_movie_id)
    history = [mid for mid in parse_ids(sample.model_history_movie_ids)[-max_history_items:] if str(mid) in token_map["movie_id_to_token"]]
    candidate_rows = []
    seen_candidates = set()
    for cand in candidates_by_sample.get(int(sample_id), []):
        movie_id = int(cand["candidate_movie_id"])
        if str(movie_id) not in token_map["movie_id_to_token"] or movie_id in seen_candidates:
            continue
        seen_candidates.add(movie_id)
        candidate_rows.append(cand)
        if len(candidate_rows) >= max_retrieved_items:
            break

    lines = ["User history:"]
    if history:
        lines.extend(movie_line(movie_id, token_map, metadata) for movie_id in history)
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append("Retrieved candidates:")
    if candidate_rows:
        lines.extend(
            movie_line(int(cand["candidate_movie_id"]), token_map, metadata, int(cand["rank"]), float(cand["score"]))
            for cand in candidate_rows
        )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("Next movie: ")
    return "\n".join(lines), [int(x) for x in history], [int(c["candidate_movie_id"]) for c in candidate_rows]


def output_paths() -> list[Path]:
    return [LLM_MOVIE_TOKEN_DATA / f"{split}_movie_token_lm.jsonl" for split in ["train", "val", "test"]]


def write_split(
    samples: pd.DataFrame,
    split: str,
    token_map: dict,
    metadata: dict[int, dict],
    candidates_by_sample: dict[int, list[dict]],
    max_history_items: int,
    max_retrieved_items: int,
    max_samples: int,
) -> dict:
    split_samples = samples[samples.split == split]
    if max_samples > 0 and len(split_samples) > max_samples:
        split_samples = split_samples.sample(max_samples, random_state=42).sort_index()
    out_path = LLM_MOVIE_TOKEN_DATA / f"{split}_movie_token_lm.jsonl"
    count = 0
    skipped = 0
    total_history_items = 0
    total_retrieved_items = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample_id, sample in split_samples.iterrows():
            target = int(sample.target_movie_id)
            target_token = token_map["movie_id_to_token"].get(str(target))
            if target_token is None:
                skipped += 1
                continue
            input_text, history, retrieved = build_prompt(
                sample,
                int(sample_id),
                token_map,
                metadata,
                candidates_by_sample,
                max_history_items,
                max_retrieved_items,
            )
            record = {
                "sample_id": int(sample_id),
                "split": split,
                "user_id": int(sample.user_id),
                "target_movie_id": target,
                "history_movie_ids": history,
                "retrieved_movie_ids": retrieved,
                "input_text": input_text,
                "target_text": target_token,
                "full_text": input_text + target_token,
                "prompt_mode": "rag_enriched_movie_tokens",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            total_history_items += len(history)
            total_retrieved_items += len(retrieved)
    return {
        "path": str(out_path),
        "rows": count,
        "skipped": skipped,
        "split_samples": int(len(split_samples)),
        "avg_history_items": float(total_history_items / count) if count else 0.0,
        "avg_retrieved_items": float(total_retrieved_items / count) if count else 0.0,
    }


def write_run_commands(args: argparse.Namespace) -> None:
    (LLM_MOVIE_TOKEN_DATA / "run_commands_rag_enriched.txt").write_text(
        "SMOKE BUILD:\n"
        "python scripts/43_build_movie_token_lm_rag_data.py \\\n"
        "  --max_history_items 20 \\\n"
        "  --max_retrieved_items 30 \\\n"
        "  --max_samples_per_split 1000 \\\n"
        "  --overwrite\n\n"
        "SMOKE TRAIN:\n"
        "python scripts/41_train_movie_token_lm_qlora.py \\\n"
        "  --model_name meta-llama/Llama-3.2-1B-Instruct \\\n"
        "  --run_name smoke_rag_enriched_movie_head_1b \\\n"
        "  --max_train_samples 1000 \\\n"
        "  --max_val_samples 200 \\\n"
        "  --max_steps 50 \\\n"
        "  --max_seq_length 512 \\\n"
        "  --per_device_train_batch_size 1 \\\n"
        "  --gradient_accumulation_steps 8 \\\n"
        "  --learning_rate 2e-4 \\\n"
        "  --projection_learning_rate 1e-3 \\\n"
        "  --movie_head_learning_rate 1e-3 \\\n"
        "  --overwrite\n\n"
        "SMOKE EVAL:\n"
        "python scripts/42_eval_movie_token_lm.py \\\n"
        "  --model_name meta-llama/Llama-3.2-1B-Instruct \\\n"
        "  --run_name smoke_rag_enriched_movie_head_1b \\\n"
        "  --split val \\\n"
        "  --max_samples 200 \\\n"
        "  --batch_size 4\n",
        encoding="utf-8",
    )


def print_sanity(samples: pd.DataFrame, token_map: dict, model_name: str, metadata: dict[int, dict], max_history_items: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(token_map["movie_id_to_token"].values())})
    movie_token_ids = set(int(x) for x in token_map["movie_id_to_token_id"].values())
    print("SANITY RAG-enriched movie-token LM rows:", flush=True)
    empty_candidates: dict[int, list[dict]] = {}
    for sample_id, sample in samples[samples.split == "train"].head(3).iterrows():
        input_text, _, _ = build_prompt(sample, int(sample_id), token_map, metadata, empty_candidates, max_history_items, 0)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--candidate_dir", default=str(OUT / "v14_cross_attention_reranker"))
    parser.add_argument("--candidate_template", default="candidates_blend_crossattn_score_beta0p5_{split}.csv")
    parser.add_argument("--max_history_items", type=int, default=20)
    parser.add_argument("--max_retrieved_items", type=int, default=30)
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LLM_MOVIE_TOKEN_DATA.mkdir(parents=True, exist_ok=True)
    LLM_DATA.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_paths() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output JSONL exists: {existing}. Use --overwrite to replace.")

    samples = pd.read_csv(SAMPLES)
    metadata = load_metadata()
    metadata_df = pd.DataFrame.from_records(list(metadata.values())) if metadata else pd.DataFrame()
    token_map = load_or_build_token_map(args.model_name, samples, metadata_df)
    candidate_dir = Path(args.candidate_dir)

    stats = {
        "prompt_mode": "rag_enriched_movie_tokens",
        "model_name": args.model_name,
        "movie_token_map": str(MOVIE_TOKEN_MAP),
        "candidate_dir": str(candidate_dir),
        "candidate_template": args.candidate_template,
        "max_history_items": int(args.max_history_items),
        "max_retrieved_items": int(args.max_retrieved_items),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        candidate_path = candidate_dir / args.candidate_template.format(split=split)
        candidates_by_sample = load_candidates(candidate_path, args.max_retrieved_items)
        stats["splits"][split] = write_split(
            samples,
            split,
            token_map,
            metadata,
            candidates_by_sample,
            args.max_history_items,
            args.max_retrieved_items,
            args.max_samples_per_split,
        )
        stats["splits"][split]["candidate_file"] = str(candidate_path)
        stats["splits"][split]["candidate_samples"] = int(len(candidates_by_sample))
    write_json(LLM_MOVIE_TOKEN_DATA / "movie_token_lm_rag_data_stats.json", stats)
    write_run_commands(args)
    print_sanity(samples, token_map, args.model_name, metadata, args.max_history_items)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
