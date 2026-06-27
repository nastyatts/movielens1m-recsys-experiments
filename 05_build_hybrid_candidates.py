from __future__ import annotations

import argparse

import pandas as pd

from common import OUT, SAMPLES


def source_frame(split: str, source: str) -> pd.DataFrame:
    path = OUT / f"candidates_{source}_{split}.csv"
    df = pd.read_csv(path)
    return df.rename(
        columns={
            "retrieval_rank": f"{source}_rank",
            "retrieval_score": f"{source}_score",
        }
    )[["sample_id", "candidate_movie_id", f"{source}_rank", f"{source}_score"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--k_rrf", type=float, default=60.0)
    parser.add_argument("--w_faiss", type=float, default=1.0)
    parser.add_argument("--w_item", type=float, default=1.5)
    parser.add_argument("--w_pop", type=float, default=0.6)
    args = parser.parse_args()

    samples = pd.read_csv(SAMPLES)
    weights = {"faiss": args.w_faiss, "item_item": args.w_item, "popularity": args.w_pop}
    for split in args.splits:
        base = None
        for source in ["faiss", "item_item", "popularity"]:
            df = source_frame(split, source)
            base = df if base is None else base.merge(df, on=["sample_id", "candidate_movie_id"], how="outer")
        score = pd.Series(0.0, index=base.index)
        for source, weight in weights.items():
            rank_col = f"{source}_rank"
            score = score + (weight / (args.k_rrf + base[rank_col])).fillna(0.0)
        base["retrieval_score"] = score
        sample_cols = samples[["user_id", "split", "target_movie_id"]]
        base = base.merge(sample_cols, left_on="sample_id", right_index=True, how="left")
        rows = []
        for sample_id, group in base.groupby("sample_id", sort=False):
            ranked = group.sort_values(["retrieval_score", "candidate_movie_id"], ascending=[False, True]).head(args.top_k)
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                item = row._asdict()
                item["retrieval_rank"] = rank
                item["method"] = "hybrid"
                rows.append(item)
        out = OUT / f"candidates_hybrid_{split}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"hybrid {split}: {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
