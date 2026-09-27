"""Final decision layer: test-distribution calibration + expected-F0.5 assignment.

Usage: python run_final.py <work> <train_scores.parquet> <test_scores.parquet> <score_col> <out_dir> <raw_test_s1.tsv>

train scores need (qi, si, <score>, y) out-of-fold; test scores need (qi, si, <score>).
"""
import json
import os
import sys

import numpy as np
import polars as pl

from calibrate import band_table, best_per_query, calibrate, simulate_f05
from common import query_ids, s1_ids

N_POS_TRAIN = 7638365


def expf_mask(b, c=0.0, low=0.05):
    d = (b.select("si", "pc").with_row_index("r").filter(pl.col("pc") > low).sort(["si", "pc"], descending=[False, True]))
    d = d.with_columns(pl.col("pc").cum_sum().over("si").alias("ctp"), pl.col("pc").cum_count().over("si").alias("k"),
                       pl.col("pc").sum().over("si").alias("S"), (1 - pl.col("pc")).log().sum().over("si").alias("lp0"))
    d = d.with_columns((1.25 * pl.col("ctp") / (0.25 * (pl.col("S") + c) + pl.col("k"))).alias("ef"),
                       (pl.col("lp0").exp() * np.exp(-c)).alias("ef0"))
    kb = d.group_by("si").agg(pl.col("ef").arg_max().alias("am"), pl.col("ef").max().alias("efm"), pl.col("ef0").first())
    kb = kb.with_columns(pl.when(pl.col("efm") > pl.col("ef0")).then(pl.col("am") + 1).otherwise(0).alias("ks"))
    keep = d.join(kb.select("si", "ks"), on="si").filter(pl.col("k") <= pl.col("ks"))["r"].to_numpy()
    m = np.zeros(b.height, bool)
    m[keep] = True
    return m


def main(work, train_path, test_path, score, out_dir, raw_s1):
    tr = pl.read_parquet(train_path, columns=["qi", "si", score, "y"])
    te = pl.read_parquet(test_path, columns=["qi", "si", score])
    btr, bte = best_per_query(tr, score), best_per_query(te, score)
    table, npos = band_table(btr, bte, score, N_POS_TRAIN)
    n_s1 = len(s1_ids(work, "test"))
    print(f"estimated test positives per S1: {npos / n_s1:.3f}")
    print(table.select("b", "lo", "n_tr", "prec_tr", "n_te", "prec_te"))
    bte = calibrate(bte, table, score)
    extra = max((npos - bte["pc"].sum()) / n_s1, 0.0)
    rules = {f"thr{t}": (lambda b, t=t: b[score].to_numpy() > t) for t in (0.7, 0.9, 0.95, 0.97)}
    rules.update({f"expf_c{c}": (lambda b, c=c: expf_mask(b, c)) for c in (0.0, 0.2)})
    res = {k: simulate_f05(bte, n_s1, extra, r, reps=3) for k, r in rules.items()}
    for k, v in res.items():
        print(f"simulated test F0.5 {k}: {v:.5f}")
    best = max(res, key=res.get)
    mask = rules[best](bte)
    pred = bte.filter(pl.Series(mask)).select("si", "qi")
    qid, sid = query_ids(work, "test"), s1_ids(work, "test")
    os.makedirs(out_dir, exist_ok=True)
    order = pl.read_csv(raw_s1, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"]).rename(
        {"entity_id": "source1_entity_id"})
    g = (pred.with_columns(pl.Series("q", qid).gather(pred["qi"]).alias("q"),
                           pl.Series("s", sid).gather(pred["si"]).alias("source1_entity_id"))
         .sort("q").group_by("source1_entity_id").agg(pl.col("q").unique(maintain_order=True).str.join(",").alias("matched_entity_ids")))
    order.join(g, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).write_csv(
        f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    json.dump({"rule": best, "simulated": res, "score": score, "accepted": int(mask.sum())},
              open(f"{out_dir}/decision.json", "w"), indent=1)
    print("chosen", best, "accepted", int(mask.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:7])
