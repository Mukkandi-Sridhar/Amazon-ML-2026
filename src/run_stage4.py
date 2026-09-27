"""Stage 4: second context round on stage-3 scores + claimant peer similarity.

True variants of one business are noisy copies of the same base record, so they
resemble *each other*; a record of a neighbouring / twin business that claims the
same Source-1 entity looks different from that entity's other claimants.  For
every (query, entity) pair we compare the query with the entity's top other
claimants (fuzzy name / address similarity, house-number agreement), recompute
the query/entity context and consensus features on the sharper stage-3 scores,
and fit a final cross-fitted LightGBM.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import s1_ids, truth_pairs
from metric import macro_f05
from run_stage3 import (BASE, CONS, CTX, FEATS as F3, P2_MIN, assign, assign_expf, attr_feats, consensus,
                        consensus_attrs, context, query_ids)

CTX3 = [f + "_3" for f in CTX + CONS]
PEER = ["pr_n", "pr_nmax", "pr_nw", "pr_amax", "pr_aw", "pr_pn_any", "pr_maxp", "pr_nmin"]
ATTR3 = attr_feats("_3")
FEATS = F3 + ["p3"] + CTX3 + ATTR3 + PEER
P4_MIN = 0.01  # stage 4 only re-scores pairs with p2 >= 0.01 (keeps 99.9% of reachable true pairs, halves memory)


def load(work, split):
    import glob
    cols = ["qi", "si", "y"] + BASE if split == "train" else ["qi", "si"] + BASE
    parts = []
    for p in sorted(glob.glob(f"{work}/{split}_p2/part-*.parquet")):
        d = pl.read_parquet(p, columns=cols).filter(pl.col("p2") >= P4_MIN)
        parts.append(d.with_columns([pl.col(c).cast(pl.Float32) for c, t in d.schema.items() if t == pl.Float64]))
    return pl.concat(parts)


def _f32(d):
    return d.with_columns([pl.col(c).cast(pl.Float32) for c, t in d.schema.items() if t == pl.Float64])
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=4)
ROUNDS = 500


def peers(d, qstr, score="p3", k=3, min_p=0.01):
    d = d.with_row_index("rid")
    top = (d.filter(pl.col(score) >= 0.2).sort(["si", score], descending=[False, True])
           .group_by("si", maintain_order=True).head(k + 1).select("si", pl.col("qi").alias("pq"), pl.col(score).alias("pp")))
    x = (d.filter(pl.col(score) >= min_p).select("rid", "qi", "si").join(top, on="si")
         .filter(pl.col("pq") != pl.col("qi")).sort(["rid", "pp"], descending=[False, True])
         .group_by("rid", maintain_order=True).head(k))
    ns, as_ = [], []
    step = 3_000_000
    for s in range(0, x.height, step):
        a = x["qi"].slice(s, step).to_numpy()
        b = x["pq"].slice(s, step).to_numpy()
        ns.append(cpdist(qstr["core"].gather(a).fill_null("").to_list(), qstr["core"].gather(b).fill_null("").to_list(),
                         scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
        as_.append(cpdist(qstr["ad"].gather(a).fill_null("").to_list(), qstr["ad"].gather(b).fill_null("").to_list(),
                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
    a_p = qstr["pnum"].gather(x["qi"]).fill_null("")
    b_p = qstr["pnum"].gather(x["pq"]).fill_null("")
    x = x.with_columns(pl.Series("ns", np.concatenate(ns) if ns else np.zeros(0, np.float32)),
                       pl.Series("as", np.concatenate(as_) if as_ else np.zeros(0, np.float32)),
                       ((a_p == b_p) & (a_p != "")).alias("pe"))
    w = pl.col("pp")
    agg = x.group_by("rid").agg(
        pl.len().cast(pl.Int8).alias("pr_n"), pl.col("ns").max().alias("pr_nmax"), pl.col("ns").min().alias("pr_nmin"),
        ((pl.col("ns") * w).sum() / w.sum()).alias("pr_nw"), pl.col("as").max().alias("pr_amax"),
        ((pl.col("as") * w).sum() / w.sum()).alias("pr_aw"), pl.col("pe").any().cast(pl.Int8).alias("pr_pn_any"),
        w.max().alias("pr_maxp"))
    d = d.join(agg, on="rid", how="left").drop("rid")
    return d.with_columns([pl.col(c).fill_null(-1) for c in PEER])


def build(work, split, p3):
    t = time.time()
    qs = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["pnum", "core", "ad"]) for i in (2, 3)])
    s1 = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["pnum", "core"])
    strings = (qs.select("pnum", "core"), s1)
    d = _f32(consensus(_f32(context(load(work, split))), work, split, strings=strings))
    d = d.join(p3.select("qi", "si", pl.col("p3").cast(pl.Float32)), on=["qi", "si"], how="left").with_columns(pl.col("p3").fill_null(0.0))
    d = _f32(context(d, "p3", "_3"))
    d = _f32(consensus(d, work, split, "p3", "_3", strings=strings))
    del strings
    d = _f32(consensus_attrs(d, work, split, "p3", "_3"))
    print(split, "context+consensus", d.shape, round(d.estimated_size() / 1e9, 2), "GB", round(time.time() - t), "s", flush=True)
    d = _f32(peers(d, qs))
    keep = ["qi", "si", "y"] if "y" in d.columns else ["qi", "si"]
    d = d.select(keep + FEATS).with_columns([pl.col(c).cast(pl.Float32) for c in FEATS if d.schema[c] == pl.Float64])
    print(split, "built", d.shape, round(time.time() - t), "s", flush=True)
    return d


def _predict(models, d, fold=None, step=2_000_000):
    out = np.empty(d.height, dtype=np.float32)
    for s in range(0, d.height, step):
        X = d.slice(s, step).select(FEATS).to_numpy().astype(np.float32)
        if fold is None:
            out[s:s + step] = (models[0].predict(X) + models[1].predict(X)) / 2
        else:
            f = fold[s:s + step]
            p = np.empty(len(X), dtype=np.float32)
            for k in (0, 1):
                if (f == k).any():
                    p[f == k] = models[k].predict(X[f == k])
            out[s:s + step] = p
    return out


def tune(work, gt_path):
    d = build(work, "train", pl.read_parquet(f"{work}/train_oof.parquet"))
    fold = d["qi"].to_numpy() % 2
    sub = (d["qi"].to_numpy() // 2) % 2 == 0
    models = []
    for k in (0, 1):
        t = time.time()
        tr = d.filter(pl.Series((fold != k) & sub))
        m = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy()), ROUNDS)
        del tr
        m.save_model(f"{work}/stage4_fold{k}.txt")
        models.append(m)
        print(f"stage4 fold {k} trained {time.time()-t:.0f}s", flush=True)
    d = d.with_columns(pl.Series("p4", _predict(models, d, fold)))
    imp = sorted(zip(models[0].feature_importance("gain"), FEATS), reverse=True)
    print("top features", [(n, int(g)) for g, n in imp[:20]], flush=True)
    truth = truth_pairs(work, gt_path)
    uni = pl.Series(np.arange(len(s1_ids(work, "train")), dtype=np.int32))
    cfg = {"rule": "thr", "score": "p4", "thr": 0.5, "oof_f05": -1.0}
    for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        f, _ = macro_f05(assign(d, "p4", thr), truth, uni)
        print(f"p4 thr={thr:.2f} macroF0.5={f:.5f}", flush=True)
        if f > cfg["oof_f05"]:
            cfg.update({"thr": thr, "oof_f05": f})
    for c in (0.0, 0.1):
        f, _ = macro_f05(assign_expf(d, "p4", c), truth, uni)
        print(f"p4 expected-F rule c={c} macroF0.5={f:.5f}", flush=True)
        if f > cfg["oof_f05"]:
            cfg.update({"rule": "expf", "c": c, "oof_f05": f})
    print("chosen", cfg, flush=True)
    json.dump(cfg, open(f"{work}/threshold4.json", "w"))
    d.select("qi", "si", "p3", "p4", "y").write_parquet(f"{work}/train_oof4.parquet")


def write_outputs(work, split, out_dir, raw_s1_path):
    cfg = json.load(open(f"{work}/threshold4.json"))
    d = build(work, split, pl.read_parquet(f"{work}/{split}_scores.parquet"))
    models = [lgb.Booster(model_file=f"{work}/stage4_fold{k}.txt") for k in (0, 1)]
    d = d.select("qi", "si").with_columns(pl.Series("p4", _predict(models, d)))
    pred = assign_expf(d, "p4", cfg["c"]) if cfg["rule"] == "expf" else assign(d, "p4", cfg["thr"])
    qid = query_ids(work, split)
    sid = s1_ids(work, split)
    os.makedirs(out_dir, exist_ok=True)
    s1_order = pl.read_csv(raw_s1_path, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])
    s1_order = s1_order.rename({"entity_id": "source1_entity_id"})
    g = (pred.with_columns(pl.Series("q", qid).gather(pred["qi"]).alias("q"),
                           pl.Series("s", sid).gather(pred["si"]).alias("source1_entity_id"))
         .sort("q").group_by("source1_entity_id").agg(pl.col("q").unique(maintain_order=True).str.join(",").alias("matched_entity_ids")))
    res = s1_order.join(g, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    res.write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    d.write_parquet(f"{work}/{split}_scores4.parquet")
    print("written", pred.height, "matches for", pred["si"].n_unique(), "S1 entities", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sys.argv[2], sys.argv[3])
    else:
        write_outputs(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
