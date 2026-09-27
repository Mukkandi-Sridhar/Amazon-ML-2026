"""Final decision: expected-F0.5 assignment on model scores + twin gate.

Usage: python run_final2.py <work> <train_oof.parquet> <test_scores.parquet> <score> <out_dir> <raw_test_s1.tsv> [gate_T]

Train-tuned expected-F0.5 per entity on the raw (train-calibrated) scores, then
drop pairs whose house numbers differ by a test-specific twin offset unless the
score exceeds gate_T (default 0.99).  See twin_gate.py for the evidence.
"""
import json
import os
import sys

import numpy as np
import polars as pl

from calibrate import best_per_query
from common import query_ids, s1_ids
from run_final import expf_mask
from twin_gate import apply_gate, gate_table, house_offset


def main(work, train_path, test_path, score, out_dir, raw_s1, gate_t=0.99):
    gate_t = float(gate_t)
    tr = pl.read_parquet(train_path, columns=["qi", "si", score, "y"])
    te = pl.read_parquet(test_path, columns=["qi", "si", score])
    btr = house_offset(best_per_query(tr, score), work, "train")
    bte = house_offset(best_per_query(te, score), work, "test")
    n_tr, n_te = len(s1_ids(work, "train")), len(s1_ids(work, "test"))
    table = gate_table(btr, bte, score, n_tr, n_te, 3.40 / 3.46)
    bte = apply_gate(bte, table, score)
    bte = bte.with_columns(pl.col(score).alias("pc"))
    acc = expf_mask(bte, 0.0)
    gate = (bte["is_twin"] & (bte[score] <= gate_t)).to_numpy()
    rem = bte.filter(pl.Series(acc & gate))
    tp = float(rem["prec_te"].sum())
    print(f"expF accepted {int(acc.sum())}; twin gate removes {rem.height} (est. TP {tp:.0f}, FP {rem.height - tp:.0f}, "
          f"est dF {((rem.height - tp) * 0.2 - tp * 0.06) / n_te:+.4f})")
    final = acc & ~gate
    pred = bte.filter(pl.Series(final)).select("si", "qi")
    qid, sid = query_ids(work, "test"), s1_ids(work, "test")
    os.makedirs(out_dir, exist_ok=True)
    order = pl.read_csv(raw_s1, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"]).rename(
        {"entity_id": "source1_entity_id"})
    g = (pred.with_columns(pl.Series("q", qid).gather(pred["qi"]).alias("q"),
                           pl.Series("s", sid).gather(pred["si"]).alias("source1_entity_id"))
         .sort("q").group_by("source1_entity_id").agg(pl.col("q").unique(maintain_order=True).str.join(",").alias("matched_entity_ids")))
    order.join(g, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).write_csv(
        f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    json.dump({"score": score, "gate_t": gate_t, "accepted": int(final.sum()), "gated": int(rem.height)},
              open(f"{out_dir}/decision.json", "w"), indent=1)
    print("final accepted", int(final.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:8])
