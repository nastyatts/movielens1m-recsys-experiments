from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import OUT, SAMPLES, write_json
from stage_a_experiment_utils import ensure_dir, eval_candidates


V22_OUT = OUT / "v2_2"
V6_OUT = OUT / "v6_rag_ablation"
V7_OUT = OUT / "v7_candidate_pool_fusion"
FALLBACK_BASE = "rrf_seq_md3_ln5_rd0p5_sd0p7_confidence_log_count1p0_item2vec0p5"
KEEP_METHODS = {"base_only", "bm25_only", "hybrid_alpha0p25_only"}


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def weight_token(value: float) -> str:
    return str(value).replace(".", "p")


def parse_union_budgets(value: str) -> list[tuple[int, int]]:
    budgets = []
    for part in parse_csv_list(value):
        pieces = part.split(":")
        if len(pieces) != 2:
            raise ValueError(f"Invalid union budget={part}. Expected format base_n:rag_n")
        base_n, rag_n = int(pieces[0]), int(pieces[1])
        if base_n < 0 or rag_n < 0:
            raise ValueError(f"Invalid union budget={part}. Values must be non-negative")
        budgets.append((base_n, rag_n))
    return budgets


def has_base_candidates(config: str) -> bool:
    return (V22_OUT / f"candidates_{config}_val.csv").exists() and (V22_OUT / f"candidates_{config}_test.csv").exists()


def best_base_recall_name() -> str:
    sort_cols = ["TargetIn@50", "TargetIn@100", "TargetIn@200", "TargetIn@30", "NDCG@30", "HR@1"]
    metrics_path = V22_OUT / "rrf_grid_val_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        cols = [col for col in sort_cols if col in df.columns]
        if cols and "config" in df.columns:
            for row in df.sort_values(cols, ascending=False).itertuples(index=False):
                config = str(getattr(row, "config"))
                if has_base_candidates(config):
                    print(f"selected base recall config from rrf_grid_val_metrics.csv: {config}", flush=True)
                    return config

    best_path = V22_OUT / "best_rrf_configs.json"
    if best_path.exists():
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        for metric in ["TargetIn@50", "TargetIn@100", "TargetIn@200", "TargetIn@30", "NDCG@30"]:
            for row in (payload.get("best_configs") or {}).get(metric, []):
                config = str(row.get("config"))
                if has_base_candidates(config):
                    print(f"selected base recall config from best_rrf_configs.json: {config}", flush=True)
                    return config

    if has_base_candidates(FALLBACK_BASE):
        print(f"selected fallback base recall config: {FALLBACK_BASE}", flush=True)
        return FALLBACK_BASE
    raise FileNotFoundError(
        "No base recall candidate files found. Expected outputs/v2_2/candidates_<config>_val.csv "
        "and candidates_<config>_test.csv."
    )


def candidate_path(source: str, split: str, base_config: str) -> Path:
    if source == "base":
        return V22_OUT / f"candidates_{base_config}_{split}.csv"
    if source in {"bm25", "hybrid_alpha0p25"}:
        return V6_OUT / f"candidates_{source}_{split}.csv"
    raise ValueError(f"unknown source={source}")


def load_candidates(source: str, split: str, base_config: str, top_k: int) -> pd.DataFrame:
    path = candidate_path(source, split, base_config)
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate file for source={source} split={split}: {path}")
    cols = [
        "sample_id",
        "user_id",
        "split",
        "target_movie_id",
        "candidate_movie_id",
        "retrieval_rank",
        "retrieval_score",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in cols)
    df = df.sort_values(["sample_id", "retrieval_rank", "candidate_movie_id"]).groupby("sample_id", sort=False).head(top_k)
    return df.reset_index(drop=True)


def recalc_rank(df: pd.DataFrame, method: str, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["retrieval_rank"] = out.groupby("sample_id").cumcount() + 1
    out = out[out["retrieval_rank"] <= top_k].copy()
    out["retrieval_score"] = 1.0 / out["retrieval_rank"].astype(float)
    out["method"] = method
    cols = [
        "sample_id",
        "user_id",
        "split",
        "target_movie_id",
        "candidate_movie_id",
        "retrieval_rank",
        "retrieval_score",
        "method",
    ]
    return out[cols]


def only_source_candidates(source_df: pd.DataFrame, method: str, top_k: int) -> pd.DataFrame:
    df = source_df.sort_values(["sample_id", "retrieval_rank", "candidate_movie_id"]).copy()
    return recalc_rank(df, method, top_k)


def budgeted_union_candidates(
    base: pd.DataFrame,
    rag: pd.DataFrame,
    method: str,
    base_n: int,
    rag_n: int,
    top_k: int,
) -> pd.DataFrame:
    if base_n + rag_n > top_k:
        raise ValueError(f"Invalid budget for {method}: base_n + rag_n must be <= top_k")

    rag_by_sample = {
        int(sample_id): group.sort_values(["retrieval_rank", "candidate_movie_id"])
        for sample_id, group in rag.groupby("sample_id", sort=False)
    }
    rows = []
    for sample_id, base_group in base.groupby("sample_id", sort=False):
        base_group = base_group.sort_values(["retrieval_rank", "candidate_movie_id"])
        selected = []
        used = set()

        for row in base_group.head(base_n).itertuples(index=False):
            item = row._asdict()
            selected.append(item)
            used.add(int(item["candidate_movie_id"]))

        added_rag = 0
        rag_group = rag_by_sample.get(int(sample_id))
        if rag_group is not None:
            for row in rag_group.itertuples(index=False):
                item = row._asdict()
                movie_id = int(item["candidate_movie_id"])
                if movie_id in used:
                    continue
                selected.append(item)
                used.add(movie_id)
                added_rag += 1
                if added_rag >= rag_n:
                    break

        for row in base_group.iloc[base_n:].itertuples(index=False):
            if len(selected) >= top_k:
                break
            item = row._asdict()
            movie_id = int(item["candidate_movie_id"])
            if movie_id in used:
                continue
            selected.append(item)
            used.add(movie_id)

        for rank, item in enumerate(selected[:top_k], start=1):
            item["retrieval_rank"] = rank
            item["retrieval_score"] = 1.0 / float(rank)
            item["method"] = method
            rows.append(item)

    out = pd.DataFrame(rows)
    cols = [
        "sample_id",
        "user_id",
        "split",
        "target_movie_id",
        "candidate_movie_id",
        "retrieval_rank",
        "retrieval_score",
        "method",
    ]
    return out[cols] if not out.empty else out


def rrf_candidates(base: pd.DataFrame, rag: pd.DataFrame, method: str, rag_weight: float, top_k: int, k_rrf: float = 60.0) -> pd.DataFrame:
    base_rank = base[["sample_id", "candidate_movie_id", "retrieval_rank"]].rename(columns={"retrieval_rank": "base_rank"})
    rag_rank = rag[["sample_id", "candidate_movie_id", "retrieval_rank"]].rename(columns={"retrieval_rank": "rag_rank"})
    merged = base_rank.merge(rag_rank, on=["sample_id", "candidate_movie_id"], how="outer")
    merged["retrieval_score"] = (1.0 / (k_rrf + merged["base_rank"])).fillna(0.0) + rag_weight * (
        1.0 / (k_rrf + merged["rag_rank"])
    ).fillna(0.0)

    meta = pd.concat(
        [
            base[["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id"]],
            rag[["sample_id", "user_id", "split", "target_movie_id", "candidate_movie_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["sample_id", "candidate_movie_id"], keep="first")
    merged = merged.merge(meta, on=["sample_id", "candidate_movie_id"], how="left")

    rows = []
    for _, group in merged.groupby("sample_id", sort=False):
        ranked = group.sort_values(["retrieval_score", "candidate_movie_id"], ascending=[False, True]).head(top_k).copy()
        ranked["retrieval_rank"] = range(1, len(ranked) + 1)
        rows.append(ranked)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["method"] = method
    cols = [
        "sample_id",
        "user_id",
        "split",
        "target_movie_id",
        "candidate_movie_id",
        "retrieval_rank",
        "retrieval_score",
        "method",
    ]
    return out[cols]


def build_variants(split: str, args: argparse.Namespace, base_config: str) -> dict[str, pd.DataFrame]:
    base = load_candidates("base", split, base_config, args.top_k)
    variants: dict[str, pd.DataFrame] = {"base_only": only_source_candidates(base, "base_only", args.top_k)}
    for rag_source in args.rag_sources:
        rag = load_candidates(rag_source, split, base_config, args.top_k)
        rag_only_name = f"{rag_source}_only"
        variants[rag_only_name] = only_source_candidates(rag, rag_only_name, args.top_k)
        for base_n, rag_n in args.union_budgets:
            if base_n + rag_n > args.top_k:
                raise ValueError(f"Invalid --union_budgets entry {base_n}:{rag_n}; sum must be <= top_k={args.top_k}")
            union_name = f"union_base{base_n}_{rag_source}{rag_n}"
            variants[union_name] = budgeted_union_candidates(base, rag, union_name, base_n, rag_n, args.top_k)
        for weight in args.rrf_weights:
            name = f"rrf_base_{rag_source}_w{weight_token(weight)}"
            variants[name] = rrf_candidates(base, rag, name, weight, args.top_k)
    return variants


def write_run_commands() -> None:
    (V7_OUT / "run_commands.txt").write_text(
        "SMOKE:\n"
        "python scripts/29_fuse_retrieval_with_rag.py \\\n"
        "  --top_k 200 \\\n"
        "  --rag_sources bm25,hybrid_alpha0p25 \\\n"
        "  --rrf_weights 0.5,1.0 \\\n"
        "  --union_budgets 150:50 \\\n"
        "  --top_n_test 3\n\n"
        "python scripts/19_collect_experiments_summary.py\n\n"
        "NORMAL:\n"
        "python scripts/29_fuse_retrieval_with_rag.py \\\n"
        "  --top_k 200 \\\n"
        "  --rag_sources bm25,hybrid_alpha0p25 \\\n"
        "  --rrf_weights 0.25,0.5,1.0 \\\n"
        "  --union_budgets 150:50,100:100 \\\n"
        "  --top_n_test 5\n\n"
        "python scripts/19_collect_experiments_summary.py\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--rag_sources", default="bm25,hybrid_alpha0p25")
    parser.add_argument("--rrf_weights", default="0.25,0.5,1.0")
    parser.add_argument("--union_budgets", default="150:50,100:100")
    parser.add_argument("--top_n_test", type=int, default=5)
    parser.add_argument("--base_config", default="")
    args = parser.parse_args()
    args.rag_sources = parse_csv_list(args.rag_sources)
    args.rrf_weights = [float(value) for value in parse_csv_list(args.rrf_weights)]
    args.union_budgets = parse_union_budgets(args.union_budgets)
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(V7_OUT)
    write_run_commands()
    samples = pd.read_csv(SAMPLES)
    val_samples = samples[samples.split == "val"]
    test_samples = samples[samples.split == "test"]
    base_config = args.base_config.strip() or best_base_recall_name()
    print(
        f"candidate pool fusion base={base_config} rag_sources={args.rag_sources} "
        f"rrf_weights={args.rrf_weights} union_budgets={args.union_budgets} top_k={args.top_k}",
        flush=True,
    )

    val_variants = build_variants("val", args, base_config)
    print(f"val variants={len(val_variants)}: {sorted(val_variants)}", flush=True)
    val_rows = []
    for name, cand in val_variants.items():
        out = V7_OUT / f"candidates_{name}_val.csv"
        cand.to_csv(out, index=False)
        metrics = eval_candidates(cand, val_samples)
        val_rows.append(
            {
                "version": "v7_candidate_pool_fusion",
                "stage": "candidate_pool_fusion",
                "split": "val",
                "config": name,
                "base_config": base_config,
                **metrics,
            }
        )
        print(
            f"val config={name} TargetIn@50={metrics['TargetIn@50']:.6f} "
            f"TargetIn@100={metrics['TargetIn@100']:.6f} TargetIn@200={metrics['TargetIn@200']:.6f}",
            flush=True,
        )

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(V7_OUT / "fusion_val_metrics.csv", index=False)
    selected = set()
    best_configs = {}
    for metric in ["TargetIn@50", "TargetIn@100", "TargetIn@200", "NDCG@30"]:
        top = val_df.sort_values([metric, "TargetIn@50", "NDCG@30", "HR@1"], ascending=False).head(args.top_n_test)
        best_configs[metric] = top.to_dict("records")
        selected.update(top["config"].astype(str).tolist())
    selected.update(KEEP_METHODS.intersection(set(val_variants)))
    print(f"selected_for_test={sorted(selected)}", flush=True)

    test_variants = build_variants("test", args, base_config)
    test_rows = []
    for name in sorted(selected):
        cand = test_variants.get(name)
        if cand is None:
            print(f"skip selected config without test candidates: {name}", flush=True)
            continue
        out = V7_OUT / f"candidates_{name}_test.csv"
        cand.to_csv(out, index=False)
        metrics = eval_candidates(cand, test_samples)
        test_rows.append(
            {
                "version": "v7_candidate_pool_fusion",
                "stage": "candidate_pool_fusion",
                "split": "test",
                "config": name,
                "base_config": base_config,
                **metrics,
            }
        )
        print(
            f"test config={name} TargetIn@50={metrics['TargetIn@50']:.6f} "
            f"TargetIn@100={metrics['TargetIn@100']:.6f} TargetIn@200={metrics['TargetIn@200']:.6f}",
            flush=True,
        )

    pd.DataFrame(test_rows).to_csv(V7_OUT / "fusion_test_metrics.csv", index=False)
    write_json(
        V7_OUT / "best_candidate_pool_fusion_configs.json",
        {
            "selection_split": "val",
            "base_config": base_config,
            "rag_sources": args.rag_sources,
            "rrf_weights": args.rrf_weights,
            "union_budgets": [f"{base_n}:{rag_n}" for base_n, rag_n in args.union_budgets],
            "top_k": args.top_k,
            "top_n_test": args.top_n_test,
            "always_tested": sorted(KEEP_METHODS.intersection(set(val_variants))),
            "selected_for_test": sorted(selected),
            "best_configs": json.loads(json.dumps(best_configs, default=float)),
        },
    )
    print(f"saved metrics to {V7_OUT / 'fusion_val_metrics.csv'} and {V7_OUT / 'fusion_test_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
