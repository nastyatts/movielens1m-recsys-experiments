from __future__ import annotations

import argparse

import faiss
import numpy as np
import pandas as pd

from common import EMB, MOVIE_ORDER, OUT, SAMPLES, movie_pos, parse_ids, recency_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--history_window", type=int, default=20)
    parser.add_argument("--recency_decay", type=float, default=0.85)
    parser.add_argument("--search_multiplier", type=int, default=6)
    args = parser.parse_args()

    emb = np.load(EMB).astype(np.float32)
    order = np.load(MOVIE_ORDER).astype(int)
    pos = movie_pos(order)
    ids = [int(x) for x in order.tolist()]
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    samples = pd.read_csv(SAMPLES)

    for split in args.splits:
        rows = []
        split_samples = samples[samples.split == split]
        search_k = min(len(ids), max(args.top_k * args.search_multiplier, args.top_k + args.history_window + 10))
        for sample_id, row in split_samples.iterrows():
            hist_all = parse_ids(row.model_history_movie_ids)
            seen_all = set(parse_ids(row.seen_movie_ids))
            hist = hist_all[-args.history_window:] if args.history_window > 0 else hist_all
            hist_pos = [pos[mid] for mid in hist if mid in pos]
            if not hist_pos:
                continue
            query = recency_mean(emb[hist_pos], args.recency_decay)
            scores, idxs = index.search(query.reshape(1, -1), search_k)
            rank = 1
            for score, idx in zip(scores[0], idxs[0]):
                mid = ids[int(idx)]
                if mid in seen_all:
                    continue
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "user_id": int(row.user_id),
                        "split": split,
                        "target_movie_id": int(row.target_movie_id),
                        "candidate_movie_id": mid,
                        "retrieval_rank": rank,
                        "retrieval_score": float(score),
                    }
                )
                rank += 1
                if rank > args.top_k:
                    break
        out = OUT / f"candidates_faiss_{split}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        df.to_csv(OUT / f"candidates_raw_{split}.csv", index=False)
        print(f"{split}: {out} rows={len(rows)} samples={len(split_samples)}")


if __name__ == "__main__":
    main()
