"""Stage-2 matcher: 2-fold cross-fitted LightGBM on train, averaged for test.

Writes <work>/<split>_p2/part-*.parquet with (qi, si, p2 [, y]) plus the
columns later stages need.
"""
import glob
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import truth_pairs
from features import DIFF_FEATS, NUM_FEATS, STAGE1_FEATS, STR_FEATS

AMB_FEATS = ["s_core_cnt", "s_skel_cnt", "q_core_cnt", "q_skel_cnt"]
FEATS = STAGE1_FEATS + ["p1", "p1rk", "p1max", "p1sum"] + STR_FEATS + NUM_FEATS + DIFF_FEATS + AMB_FEATS
CATEGORICAL = [FEATS.index("lg_q"), FEATS.index("lg_s")]
KEEP = ["qi", "si", "src", "p1", "cos_n", "cos_a", "n_tset", "a_tset", "pnum_eq", "num_cq", "sk_cat", "ad_empty",
        "n_ratio", "nm_tset", "sk_ratio", "a_ratio", "num_tset", "pn_absdiff", "pn_highdiff_rel", "pn_lowdiff", "st_eq",
        "city_ratio", "q_nnum", "s_nnum", "nonlat", "dom", "dn_q_idf", "dn_s_idf", "da_q_idf", "da_s_idf", "lg_q", "lg_s",
        "n_keyed"] + AMB_FEATS


def ambiguity(work, split):
    """How many Source-1 records (same country) share a record's exact core name / name skeleton.

    Returns per-S1-row and per-query-row count arrays; a unique name makes a name-only match safe.
    """
    s1 = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["country", "core", "skel"])
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["country", "core", "skel"]) for i in (2, 3)])
    cc = s1.group_by("country", "core").len().rename({"len": "cc"})
    sc = s1.group_by("country", "skel").len().rename({"len": "sc"})
    s1c = s1.join(cc, on=["country", "core"], how="left", maintain_order="left").join(sc, on=["country", "skel"], how="left", maintain_order="left")
    qc = q.join(cc, on=["country", "core"], how="left", maintain_order="left").join(sc, on=["country", "skel"], how="left", maintain_order="left")
    return (s1c["cc"].fill_null(0).to_numpy().astype(np.float32), s1c["sc"].fill_null(0).to_numpy().astype(np.float32),
            qc["cc"].fill_null(0).to_numpy().astype(np.float32), qc["sc"].fill_null(0).to_numpy().astype(np.float32))


def add_amb(d, amb):
    s_cc, s_sc, q_cc, q_sc = amb
    qi = d["qi"].to_numpy()
    si = d["si"].to_numpy()
    return d.with_columns(pl.Series("s_core_cnt", s_cc[si]), pl.Series("s_skel_cnt", s_sc[si]),
                          pl.Series("q_core_cnt", q_cc[qi]), pl.Series("q_skel_cnt", q_sc[qi]))
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3)
ROUNDS = 700


def fold_of(qi):
    return qi % 2


def train(work, gt_path, sample_mod=4):
    truth = truth_pairs(work, gt_path).with_columns(pl.lit(1, pl.Int8).alias("y"))
    amb = ambiguity(work, "train")
    parts = sorted(glob.glob(f"{work}/train_feat/part-*.parquet"))
    models = []
    for k in (0, 1):
        t = time.time()
        rows = []
        for p in parts:
            d = pl.read_parquet(p, columns=["qi", "si"] + [f for f in FEATS if f not in AMB_FEATS])
            d = d.filter((fold_of(pl.col("qi")) != k) & ((pl.col("qi") // 2) % sample_mod == 0))
            rows.append(add_amb(d, amb))
        d = pl.concat(rows).join(truth, on=["qi", "si"], how="left").with_columns(pl.col("y").fill_null(0))
        print(f"fold {k}: train rows {d.height} pos {d['y'].sum()}", flush=True)
        ds = lgb.Dataset(d.select(FEATS).to_numpy().astype(np.float32), d["y"].to_numpy(), free_raw_data=True,
                         categorical_feature=CATEGORICAL)
        del d
        m = lgb.train(PARAMS, ds, ROUNDS)
        m.save_model(f"{work}/stage2_fold{k}.txt")
        models.append(m)
        print(f"fold {k}: trained in {time.time()-t:.0f}s", flush=True)
    return models


PRED_KW = dict(pred_early_stop=True, pred_early_stop_freq=25, pred_early_stop_margin=10.0, num_threads=4)


def predict(work, split, models, gt_path=None):
    out = f"{work}/{split}_p2"
    os.makedirs(out, exist_ok=True)
    truth = truth_pairs(work, gt_path).with_columns(pl.lit(1, pl.Int8).alias("y")) if split == "train" else None
    amb = ambiguity(work, split)
    for p in sorted(glob.glob(f"{work}/{split}_feat/part-*.parquet")):
        if os.path.exists(f"{out}/{os.path.basename(p)}"):
            continue
        t = time.time()
        d = add_amb(pl.read_parquet(p, columns=[c for c in dict.fromkeys(KEEP + FEATS) if c not in AMB_FEATS]), amb)
        X = d.select(FEATS).to_numpy().astype(np.float32)
        if split == "train":
            f = fold_of(d["qi"].to_numpy())
            p2 = np.empty(len(X), dtype=np.float64)
            for k in (0, 1):
                p2[f == k] = models[k].predict(X[f == k], **PRED_KW)
        else:
            p2 = (models[0].predict(X, **PRED_KW) + models[1].predict(X, **PRED_KW)) / 2
        d = d.select(KEEP).with_columns(pl.Series("p2", p2.astype(np.float32)))
        if truth is not None:
            d = d.join(truth, on=["qi", "si"], how="left").with_columns(pl.col("y").fill_null(0))
        d.write_parquet(f"{out}/{os.path.basename(p)}")
        print(split, os.path.basename(p), round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    work, gt_path = sys.argv[1], sys.argv[2]
    what = sys.argv[3] if len(sys.argv) > 3 else "all"
    if what in ("all", "train"):
        models = train(work, gt_path)
    else:
        models = [lgb.Booster(model_file=f"{work}/stage2_fold{k}.txt") for k in (0, 1)]
    if what in ("all", "train", "predict_train"):
        predict(work, "train", models, gt_path)
    if what in ("all", "predict_test"):
        predict(work, "test", models)
