"""Label-shift calibration of match scores for the test distribution.

The test set contains many more "twin" distractors (neighbouring businesses
with a slightly different house number / legal form) than train, so a score
calibrated on train over-states the match probability on test.

Assumption (label shift on positives): true matches produce the same score
distribution P(score | match) on test as on train.  Then, with S = per-query
best score and bands b:
    N_pos_test = n_test(top band) * prec_train(top band) / P(top band | match)_train
    E[#matches in band b on test] = N_pos_test * P(b | match)_train
    calibrated precision of band b on test = E[#matches] / n_test(b)
The calibrated probabilities are then used to choose the decision threshold
by simulating the expected macro F0.5 on test.
"""
import numpy as np
import polars as pl

BANDS = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]


def best_per_query(d, score):
    return d.sort(score, descending=True).group_by("qi", maintain_order=True).first()


def band_table(train_best, test_best, score, n_pos_train, top=0.999):
    edges = [-np.inf] + BANDS + [np.inf]
    def band(x):
        return np.searchsorted(np.array(BANDS), x, side="left")
    tr = train_best.with_columns(pl.Series("b", band(train_best[score].to_numpy())))
    te = test_best.with_columns(pl.Series("b", band(test_best[score].to_numpy())))
    a = tr.group_by("b").agg(pl.len().alias("n_tr"), pl.col("y").sum().alias("pos_tr"))
    b = te.group_by("b").agg(pl.len().alias("n_te"))
    t = a.join(b, on="b", how="full", coalesce=True).fill_null(0).sort("b")
    topb = len(BANDS)
    r = t.filter(pl.col("b") == topb)
    p_top = r["pos_tr"][0] / n_pos_train
    npos_te = r["n_te"][0] * (r["pos_tr"][0] / r["n_tr"][0]) / p_top
    t = t.with_columns((pl.col("pos_tr") / n_pos_train * npos_te).alias("pos_te"))
    t = t.with_columns((pl.col("pos_tr") / pl.col("n_tr")).alias("prec_tr"),
                       (pl.col("pos_te") / pl.col("n_te")).clip(0, 1).alias("prec_te"))
    # enforce monotonicity (isotonic, cumulative max from the low end)
    t = t.with_columns(pl.col("prec_te").fill_nan(0).cum_max().alias("prec_te"))
    lo = [edges[i] for i in t["b"].to_list()]
    t = t.with_columns(pl.Series("lo", lo))
    return t, npos_te


def calibrate(test_best, table, score):
    idx = np.searchsorted(np.array(BANDS), test_best[score].to_numpy(), side="left")
    m = dict(zip(table["b"].to_list(), table["prec_te"].to_list()))
    return test_best.with_columns(pl.Series("pc", np.array([m.get(i, 0.0) for i in idx], dtype=np.float64)))


def simulate_f05(best_cal, n_s1, extra_true_per_s1, rule, reps=3, seed=0):
    """Expected macro F0.5 on test when labels are drawn from calibrated probabilities.

    best_cal: frame (qi, si, pc) of every query's best candidate.  rule(best_cal) -> boolean accept mask.
    extra_true_per_s1: expected number of true matches per S1 that are not any query's best candidate.
    """
    rng = np.random.default_rng(seed)
    acc = rule(best_cal)
    si = best_cal["si"].to_numpy()
    pc = best_cal["pc"].to_numpy()
    out = []
    for _ in range(reps):
        y = rng.random(len(pc)) < pc
        tp = np.bincount(si, weights=(y & acc), minlength=n_s1)
        npred = np.bincount(si, weights=acc, minlength=n_s1)
        ntrue = np.bincount(si, weights=y, minlength=n_s1) + rng.poisson(extra_true_per_s1, n_s1)
        f = np.where(ntrue == 0, (npred == 0).astype(float),
                     np.where((npred == 0) | (tp == 0), 0.0,
                              1.25 * (tp / np.maximum(npred, 1)) * (tp / np.maximum(ntrue, 1)) /
                              (0.25 * (tp / np.maximum(npred, 1)) + tp / np.maximum(ntrue, 1))))
        out.append(f.mean())
    return float(np.mean(out))


def add_strata(best, work, split):
    """Stratum of each (query, best candidate) pair: house-number relation x exact core-name equality."""
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["pnum", "core"]) for i in (2, 3)])
    s = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["pnum", "core"])
    a = q["pnum"].gather(best["qi"]).fill_null("")
    b = s["pnum"].gather(best["si"]).fill_null("")
    ia = a.str.slice(0, 12).cast(pl.Int64, strict=False)
    ib = b.str.slice(0, 12).cast(pl.Int64, strict=False)
    tmp = pl.DataFrame({"a": a, "b": b, "d": (ia - ib).abs(), "ce": q["core"].gather(best["qi"]) == s["core"].gather(best["si"])})
    pn = (pl.when((pl.col("a") == "") | (pl.col("b") == "")).then(pl.lit("miss")).when(pl.col("d") == 0).then(pl.lit("eq"))
          .when(pl.col("d") <= 30).then(pl.lit("small")).otherwise(pl.lit("big")))
    st = tmp.select((pn + pl.when(pl.col("ce")).then(pl.lit("|name=")).otherwise(pl.lit("|name~"))).alias("stratum"))["stratum"]
    return best.with_columns(st)


def strata_table(train_best, test_best, score, n_pos_train, npos_te):
    """Per (stratum, band) test precision assuming P(stratum, band | match) is unchanged."""
    band = lambda x: np.searchsorted(np.array(BANDS), x, side="left")
    tr = train_best.with_columns(pl.Series("b", band(train_best[score].to_numpy())))
    te = test_best.with_columns(pl.Series("b", band(test_best[score].to_numpy())))
    a = tr.group_by("stratum", "b").agg(pl.len().alias("n_tr"), pl.col("y").sum().alias("pos_tr"))
    c = te.group_by("stratum", "b").agg(pl.len().alias("n_te"))
    t = a.join(c, on=["stratum", "b"], how="full", coalesce=True).fill_null(0)
    t = t.with_columns((pl.col("pos_tr") / n_pos_train * npos_te).alias("pos_te"))
    t = t.with_columns((pl.col("pos_tr") / pl.col("n_tr")).fill_nan(0).alias("prec_tr"),
                       (pl.col("pos_te") / pl.col("n_te")).fill_nan(0).clip(0, 1).alias("prec_te"))
    t = t.sort("stratum", "b").with_columns(pl.col("prec_te").cum_max().over("stratum").alias("prec_te"))
    return t


def calibrate_strata(test_best, table, score):
    band = np.searchsorted(np.array(BANDS), test_best[score].to_numpy(), side="left")
    d = test_best.with_columns(pl.Series("b", band))
    d = d.join(table.select("stratum", "b", pl.col("prec_te").alias("pc")), on=["stratum", "b"], how="left")
    return d.with_columns(pl.col("pc").fill_null(0.0))
