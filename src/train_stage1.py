"""Train the stage-1 cascade model (cheap retrieval-score features).

Uses a random 8% sample of train queries (seed 0).  Candidate sets of a
query do not depend on other queries (IDF is computed on Source 1 only), so
the full-run blocking chunks already contain exactly the sample's candidates.
Of the sampled queries, 40% are used for fitting and 30% for a recall report.
"""
import glob
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import query_ids, truth_pairs
from features import STAGE1_FEATS, query_meta, stage1_features

PARAMS = dict(objective="binary", learning_rate=0.15, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.5, bagging_freq=1, verbose=-1, num_threads=4)
ROUNDS = 150


def main(work, gt_path, out_model):
    qid = query_ids(work, "train")
    full = pl.DataFrame({"entity_id": qid}).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
    sample = full.sample(fraction=0.08, seed=0).with_row_index("local")
    sample = sample.with_columns((pl.col("local") % 10).alias("mod")).select("qi", "mod").filter(pl.col("mod") <= 6)
    q = pl.concat([pl.read_parquet(f"{work}/train_s{i}_norm.parquet", columns=["entity_id", "dom", "nonlat", "ad_empty", "core", "ad", "nums"]) for i in (2, 3)])
    qmeta = query_meta(q)
    del q
    truth = truth_pairs(work, gt_path).with_columns(pl.lit(1, pl.Int8).alias("y"))
    parts = []
    for p in sorted(glob.glob(f"{work}/train_cand/part-*.parquet")):
        c = pl.read_parquet(p).join(sample, on="qi", how="inner")
        parts.append(stage1_features(c, qmeta))
    f = pl.concat(parts).join(truth, on=["qi", "si"], how="left").with_columns(pl.col("y").fill_null(0))
    tr = f.filter((pl.col("mod") >= 3) & (pl.col("mod") <= 6))
    m = lgb.train(PARAMS, lgb.Dataset(tr.select(STAGE1_FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy()), ROUNDS)
    m.save_model(out_model)
    va = f.filter(pl.col("mod") < 3)
    va = va.select("qi", "y").with_columns(pl.Series("p", m.predict(va.select(STAGE1_FEATS).to_numpy().astype(np.float32))))
    va = va.with_columns(pl.col("p").rank("ordinal", descending=True).over("qi").alias("rk"))
    pos = va["y"].sum()
    for k in (1, 3, 6, 10):
        print(f"stage-1 recall@{k} (of blocked positives): {va.filter((pl.col('y') == 1) & (pl.col('rk') <= k)).height / pos:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
