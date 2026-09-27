"""Run blocking for a split and cache candidates; on train also report recall."""
import sys
import time

import numpy as np
import polars as pl

from blocking import add_tokens, block


COLS = ["entity_id", "country", "core", "alt", "ad", "nums", "skel", "pnum", "nonlat", "ad_empty"]


def load(work, split):
    s1 = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=COLS)
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=COLS) for i in (2, 3)])
    return s1, q


def main(work, split, k, frac=1.0, tag=""):
    t = time.time()
    s1, q = load(work, split)
    if frac < 1.0:
        q = q.sample(fraction=frac, seed=0)
    cand = block(s1, q, log=lambda m: print(m, flush=True), tokenize=add_tokens)
    cand.write_parquet(f"{work}/{split}_cand{tag}.parquet")
    q.select("entity_id").write_parquet(f"{work}/{split}_qids{tag}.parquet")
    print("pairs", cand.height, "time", round(time.time() - t), flush=True)
    if split == "train":
        gt = (pl.read_parquet(f"{work}/train_gt.parquet")
              .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
        qid = pl.DataFrame({"qi": np.arange(q.height, dtype=np.int32), "matched_entity_ids": q["entity_id"]})
        sid = pl.DataFrame({"si": np.arange(s1.height, dtype=np.int32), "source1_entity_id": s1["entity_id"]})
        truth = gt.join(qid, on="matched_entity_ids").join(sid, on="source1_entity_id").select("qi", "si")
        hit = truth.join(cand, on=["qi", "si"], how="left")
        print("union recall", round(float(hit["rk_c"].is_not_null().mean()), 5))
        for col in ["rk_n", "rk_a", "rk_c"]:
            for kk in [1, 3, 5, 10, 15]:
                print(f"{col} recall@{kk}", round(float((hit[col].fill_null(999) < kk).mean()), 5))
        hit = hit.with_columns(pl.min_horizontal("rk_n", "rk_a", "rk_c").alias("brank"))
        qq = q.select(pl.col("nonlat"), pl.col("ad_empty"), pl.col("country")).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
        h2 = hit.join(qq, on="qi")
        print(h2.group_by("country", "nonlat", "ad_empty").agg(pl.len(), (pl.col("brank").fill_null(9999) < 1).mean().alias("r1"),
                                                             pl.col("brank").is_not_null().mean().alias("rK")).sort("country", "nonlat", "ad_empty"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
         sys.argv[5] if len(sys.argv) > 5 else "")
