"""Stage-3 context model + assignment + submission writer.

Stage-2 probabilities are aggregated per query (margin over the query's other
candidates) and per Source-1 entity (how many other strong claims it has, where
this pair ranks among them).  A second LightGBM (2-fold cross-fitted on train)
re-scores every pair.  Each query is then assigned to its single best Source-1
entity if the score clears a threshold tuned for macro F0.5 on train OOF.
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import query_ids, s1_ids, truth_pairs
from metric import macro_f05

BASE = ["p2", "p1", "cos_n", "cos_a", "n_tset", "a_tset", "pnum_eq", "num_cq", "sk_cat", "ad_empty", "src",
        "n_ratio", "nm_tset", "sk_ratio", "a_ratio", "num_tset", "pn_absdiff", "pn_highdiff_rel", "pn_lowdiff", "st_eq",
        "city_ratio", "q_nnum", "s_nnum", "nonlat", "dom", "s_core_cnt", "s_skel_cnt", "q_core_cnt", "q_skel_cnt"]
CTX = ["p2rk", "q_p2max", "q_marg", "q_p2sum", "q_n", "s_n", "s_p2sum_o", "s_p2max_o", "s_srk", "s_ntop_o",
       "s_ntop_same_src_o", "s_top_p2mean_o", "is_qtop"]
# Consensus among the queries claiming the same Source-1 entity: when several
# independent S2/S3 records agree on a house number / name that differs from
# Source 1, the Source-1 record is usually the noisy one (a true match), whereas
# a lone deviating record is more often a different (neighbouring) business.
CONS = ["c_pn_same", "c_pn_same_p2", "c_core_same", "c_core_same_p2", "c_pn_is_mode", "c_mode_eq_s1", "c_core_is_mode",
        "c_cmode_eq_s1", "c_pn_s_agree_p2"]
FEATS = BASE + CTX + CONS
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=3)
ROUNDS = 400


P2_MIN = 1e-3  # pairs below this stage-2 probability are never assigned; dropping them keeps stage 3 in memory


def load(work, split):
    cols = ["qi", "si", "y"] + BASE if split == "train" else ["qi", "si"] + BASE
    return pl.concat([pl.read_parquet(p, columns=cols).filter(pl.col("p2") >= P2_MIN)
                      for p in sorted(glob.glob(f"{work}/{split}_p2/part-*.parquet"))])


def all_pairs(work, split):
    return pl.concat([pl.read_parquet(p, columns=["qi", "si"]) for p in sorted(glob.glob(f"{work}/{split}_p2/part-*.parquet"))])


def context(d):
    d = d.with_columns(
        pl.col("p2").rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias("p2rk"),
        pl.col("p2").max().over("qi").alias("q_p2max"),
        pl.col("p2").sum().over("qi").alias("q_p2sum"),
        pl.len().over("qi").cast(pl.Int16).alias("q_n"),
    )
    second = pl.col("p2").top_k(2).min().over("qi")
    d = d.with_columns(
        pl.when(pl.col("p2rk") == 1).then(pl.col("p2") - second).otherwise(pl.col("p2") - pl.col("q_p2max")).alias("q_marg"),
        ((pl.col("p2rk") == 1) & (pl.col("p2") > 0.5)).cast(pl.Int8).alias("is_qtop"),
    )
    top = pl.col("is_qtop") == 1
    d = d.with_columns(
        pl.len().over("si").cast(pl.Int32).alias("s_n"),
        (pl.col("p2").sum().over("si") - pl.col("p2")).alias("s_p2sum_o"),
        pl.col("p2").rank("ordinal", descending=True).over("si").cast(pl.Int32).alias("s_srk"),
        (pl.col("is_qtop").cast(pl.Int32).sum().over("si") - pl.col("is_qtop")).alias("s_ntop_o"),
        (pl.col("is_qtop").cast(pl.Int32).sum().over(["si", "src"]) - pl.col("is_qtop")).alias("s_ntop_same_src_o"),
        ((pl.when(top).then(pl.col("p2")).otherwise(0.0).sum().over("si") - pl.when(top).then(pl.col("p2")).otherwise(0.0))
         / pl.max_horizontal(pl.col("is_qtop").cast(pl.Int32).sum().over("si") - pl.col("is_qtop"), 1)).alias("s_top_p2mean_o"),
    )
    # best score among the *other* rows of the same S1 entity
    g = d.group_by("si").agg(pl.col("p2").max().alias("_m1"), pl.col("p2").top_k(2).min().alias("_m2"))
    d = d.join(g, on="si", how="left")
    d = d.with_columns(
        pl.when(pl.col("s_n") == 1).then(0.0).when(pl.col("p2") >= pl.col("_m1")).then(pl.col("_m2"))
        .otherwise(pl.col("_m1")).alias("s_p2max_o")).drop("_m1", "_m2")
    return d


def consensus(d, work, split):
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["pnum", "core"]) for i in (2, 3)])
    s1 = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["pnum", "core"])
    qi = d["qi"].to_numpy()
    si = d["si"].to_numpy()
    d = d.with_columns(q["pnum"].gather(qi).fill_null("").alias("_qp"), s1["pnum"].gather(si).fill_null("").alias("_sp"),
                       q["core"].gather(qi).fill_null("").alias("_qc"), s1["core"].gather(si).fill_null("").alias("_sc"))
    del q, s1
    hasp = pl.col("_qp") != ""
    d = d.with_columns(
        pl.when(hasp).then(pl.len().over(["si", "_qp"]) - 1).otherwise(-1).cast(pl.Int32).alias("c_pn_same"),
        pl.when(hasp).then(pl.col("p2").sum().over(["si", "_qp"]) - pl.col("p2")).otherwise(-1.0).alias("c_pn_same_p2"),
        (pl.len().over(["si", "_qc"]) - 1).cast(pl.Int32).alias("c_core_same"),
        (pl.col("p2").sum().over(["si", "_qc"]) - pl.col("p2")).alias("c_core_same_p2"),
        (pl.when(pl.col("_qp") == pl.col("_sp")).then(pl.col("p2")).otherwise(0.0).sum().over("si")
         - pl.when(pl.col("_qp") == pl.col("_sp")).then(pl.col("p2")).otherwise(0.0)).alias("c_pn_s_agree_p2"),
    )
    pm = (d.filter(hasp).group_by(["si", "_qp"]).agg(pl.col("p2").sum().alias("w"))
          .sort(["si", "w"], descending=[False, True]).group_by("si", maintain_order=True).first()
          .select("si", pl.col("_qp").alias("_pmode")))
    cm = (d.group_by(["si", "_qc"]).agg(pl.col("p2").sum().alias("w"))
          .sort(["si", "w"], descending=[False, True]).group_by("si", maintain_order=True).first()
          .select("si", pl.col("_qc").alias("_cmode")))
    d = d.join(pm, on="si", how="left").join(cm, on="si", how="left")
    d = d.with_columns(
        pl.when(hasp).then((pl.col("_qp") == pl.col("_pmode")).cast(pl.Int8)).otherwise(-1).cast(pl.Int8).alias("c_pn_is_mode"),
        pl.when(pl.col("_pmode").is_null()).then(-1).otherwise((pl.col("_pmode") == pl.col("_sp")).cast(pl.Int8)).cast(pl.Int8).alias("c_mode_eq_s1"),
        (pl.col("_qc") == pl.col("_cmode")).cast(pl.Int8).alias("c_core_is_mode"),
        (pl.col("_cmode") == pl.col("_sc")).cast(pl.Int8).alias("c_cmode_eq_s1"),
    )
    return d.drop("_qp", "_sp", "_qc", "_sc", "_pmode", "_cmode")


def assign(d, score, thr):
    best = d.sort(score, descending=True).group_by("qi", maintain_order=True).first()
    return best.filter(pl.col(score) > thr).select("si", "qi")


def assign_expf(d, score, c=0.0, low=0.05):
    """Per-entity expected-F0.5 maximisation.

    Every query first picks its best Source-1 entity.  For each entity, its claimants
    are sorted by probability and the prefix length k maximising the plug-in
    expected F0.5 = 1.25*sum_{i<=k} p_i / (0.25*(sum_i p_i + c) + k) is kept; k = 0
    (predict "no match") is chosen when P(no true match) = prod(1-p_i)*exp(-c) is larger.
    `c` is the expected number of true matches the candidates missed.
    """
    b = (d.sort(score, descending=True).group_by("qi", maintain_order=True).first()
         .select("qi", "si", pl.col(score).alias("p")).filter(pl.col("p") > low)
         .sort(["si", "p"], descending=[False, True]))
    b = b.with_columns(pl.col("p").cum_sum().over("si").alias("ctp"), pl.col("p").cum_count().over("si").alias("k"),
                       pl.col("p").sum().over("si").alias("S"), (1 - pl.col("p")).log().sum().over("si").alias("lp0"))
    b = b.with_columns((1.25 * pl.col("ctp") / (0.25 * (pl.col("S") + c) + pl.col("k"))).alias("ef"),
                       (pl.col("lp0").exp() * np.exp(-c)).alias("ef0"))
    kb = b.group_by("si").agg(pl.col("ef").arg_max().alias("am"), pl.col("ef").max().alias("efm"), pl.col("ef0").first())
    kb = kb.with_columns(pl.when(pl.col("efm") > pl.col("ef0")).then(pl.col("am") + 1).otherwise(0).alias("kstar"))
    return b.join(kb.select("si", "kstar"), on="si").filter(pl.col("k") <= pl.col("kstar")).select("si", "qi")


def decide(d, cfg):
    if cfg.get("rule") == "expf":
        return assign_expf(d, cfg["score"], cfg["c"])
    return assign(d, cfg["score"], cfg["thr"])


def tune(work, gt_path):
    t = time.time()
    d = consensus(context(load(work, "train")), work, "train")
    print("context", d.shape, round(time.time() - t), flush=True)
    p3 = np.zeros(d.height, dtype=np.float32)
    fold = (d["qi"].to_numpy() % 2)
    sub = ((d["qi"].to_numpy() // 2) % 2 == 0)
    X = d.select(FEATS).to_numpy().astype(np.float32)
    y = d["y"].to_numpy()
    for k in (0, 1):
        tr = (fold != k) & sub
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS)
        m.save_model(f"{work}/stage3_fold{k}.txt")
        p3[fold == k] = m.predict(X[fold == k])
        print(f"stage3 fold {k} done {time.time()-t:.0f}s", flush=True)
    del X
    d = d.with_columns(pl.Series("p3", p3))
    d.select("qi", "si", "p2", "p3", "y").write_parquet(f"{work}/train_oof.parquet")
    truth = truth_pairs(work, gt_path)
    uni = pl.Series(np.arange(len(s1_ids(work, "train")), dtype=np.int32))
    found = d.filter(pl.col("y") == 1).height
    print(f"truth pairs {truth.height}, in candidates {found} ({found/truth.height:.4f})")
    best = {}
    for score in ("p2", "p3"):
        res = []
        for thr in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
            f, _ = macro_f05(assign(d, score, thr), truth, uni)
            res.append((f, thr))
            print(f"{score} thr={thr:.2f} macroF0.5={f:.5f}", flush=True)
        best[score] = max(res)
    print("best", best)
    cfg = {"rule": "thr", "score": "p3", "thr": best["p3"][1], "oof_f05": best["p3"][0], "oof_f05_p2": best["p2"][0]}
    for c in (0.0, 0.1, 0.2):
        f, _ = macro_f05(assign_expf(d, "p3", c), truth, uni)
        print(f"p3 expected-F rule c={c} macroF0.5={f:.5f}", flush=True)
        if f > cfg["oof_f05"]:
            cfg.update({"rule": "expf", "c": c, "oof_f05": f})
    print("chosen", cfg)
    json.dump(cfg, open(f"{work}/threshold.json", "w"))


def write_outputs(work, split, out_dir, raw_s1_path):
    """Predict with stage-3 fold models, assign and write both submission files."""
    import os
    cfg = json.load(open(f"{work}/threshold.json"))
    d = consensus(context(load(work, split)), work, split)
    X = d.select(FEATS).to_numpy().astype(np.float32)
    ms = [lgb.Booster(model_file=f"{work}/stage3_fold{k}.txt") for k in (0, 1)]
    d = d.with_columns(pl.Series("p3", ((ms[0].predict(X) + ms[1].predict(X)) / 2).astype(np.float32)))
    del X
    qid = query_ids(work, split)
    sid = s1_ids(work, split)
    pred = decide(d, cfg)
    os.makedirs(out_dir, exist_ok=True)
    s1_order = pl.read_csv(raw_s1_path, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])
    s1_order = s1_order.rename({"entity_id": "source1_entity_id"})

    def to_rows(pairs, col):
        g = (pairs.with_columns(pl.Series("q", qid).gather(pairs["qi"]).alias("q"),
                                pl.Series("s", sid).gather(pairs["si"]).alias("source1_entity_id"))
             .sort("q").group_by("source1_entity_id").agg(pl.col("q").unique(maintain_order=True).str.join(",").alias(col)))
        return s1_order.join(g, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))

    to_rows(pred, "matched_entity_ids").write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    d.select("qi", "si", "p2", "p3").write_parquet(f"{work}/{split}_scores.parquet")
    del d
    to_rows(all_pairs(work, split), "candidate_entity_ids").write_csv(f"{out_dir}/candidate_pairs.tsv", separator="\t", quote_style="never")
    print("written", pred.height, "matches for", pred["si"].n_unique(), "S1 entities")


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sys.argv[2], sys.argv[3])
    else:
        write_outputs(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
