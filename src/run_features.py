"""Stage-1 cascade + stage-2 string features for every blocking chunk of a split.

Reads <work>/<split>_cand/part-*.parquet, keeps the top-M candidates per query
by the stage-1 model, computes pairwise string features and writes
<work>/<split>_feat/part-*.parquet.
"""
import glob
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from features import (Q_STR, S1_STR, STAGE1_FEATS, diff_features, idf_tables, number_features, query_meta,
                      stage1_features, string_features)

TOP_M = 8


def main(work, split, stage1_model):
    t = time.time()
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["entity_id", "dom", "nonlat", "ad_empty"] + Q_STR)
                   for i in (2, 3)])
    qmeta = query_meta(q)
    qs = q.select(Q_STR)
    del q
    s1s = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=S1_STR)
    idf = idf_tables(s1s)
    m1 = lgb.Booster(model_file=stage1_model)
    out = f"{work}/{split}_feat"
    os.makedirs(out, exist_ok=True)
    parts = sorted(glob.glob(f"{work}/{split}_cand/part-*.parquet"))
    print("loaded", round(time.time() - t), "parts", len(parts), flush=True)
    for p in parts:
        dst = f"{out}/{os.path.basename(p)}"
        if os.path.exists(dst):
            continue
        t1 = time.time()
        c = stage1_features(pl.read_parquet(p), qmeta)
        p1 = m1.predict(c.select(STAGE1_FEATS).to_numpy().astype(np.float32), num_threads=4)
        c = c.with_columns(pl.Series("p1", p1.astype(np.float32)))
        c = c.with_columns(pl.col("p1").rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias("p1rk"))
        c = c.filter(pl.col("p1rk") <= TOP_M)
        c = c.with_columns(pl.col("p1").max().over("qi").alias("p1max"), pl.col("p1").sum().over("qi").alias("p1sum"))
        c = string_features(c, s1s, qs)
        c = pl.concat([c, number_features(c.select("qi", "si"), s1s, qs), diff_features(c.select("qi", "si"), s1s, qs, idf)],
                      how="horizontal")
        c.write_parquet(dst)
        print(os.path.basename(p), c.height, round(time.time() - t1), "s", flush=True)
    print("done", round(time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
