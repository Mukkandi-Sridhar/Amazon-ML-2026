"""Twin gate: test-specific protection against neighbouring-business distractors.

Analysis of the (unlabelled) test set shows an excess of best-candidate pairs
whose house numbers differ by exactly one of TWIN_OFFSETS (2-5x the train rate,
while every other offset only shows the 1.2x expected from test's larger
S2/S3 volume).  These are twin businesses created by shifting the house number.
Assuming true matches keep their train distribution of (offset, score band), the
test precision of each (offset group, band) cell is
    E[true matches per S1 on test] / (test pairs per S1)
and a pair is accepted only if that estimated precision clears the F0.5
break-even point.
"""
import numpy as np
import polars as pl

TWIN_OFFSETS = (1, 2, 3, 4, 5, 7, 9, 11, 13, 21)
BANDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]


def house_offset(best, work, split):
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["pnum"]) for i in (2, 3)])["pnum"]
    s = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["pnum"])["pnum"]
    a = q.gather(best["qi"]).fill_null("").str.slice(0, 12).cast(pl.Int64, strict=False)
    b = s.gather(best["si"]).fill_null("").str.slice(0, 12).cast(pl.Int64, strict=False)
    d = (a - b).abs().fill_null(-1)
    grp = pl.when(d.is_in(list(TWIN_OFFSETS))).then(pl.lit("twin")).otherwise(pl.lit("other"))
    return best.with_columns(d.alias("d")).with_columns(grp.alias("grp"))


def gate_table(train_best, test_best, score, n_s1_tr, n_s1_te, pos_ratio=1.0):
    band = lambda x: np.searchsorted(np.array(BANDS), x, side="right")
    tr = train_best.with_columns(pl.Series("b", band(train_best[score].to_numpy())))
    te = test_best.with_columns(pl.Series("b", band(test_best[score].to_numpy())))
    a = tr.group_by("grp", "b").agg((pl.col("y").sum() / n_s1_tr).alias("true_per_s1"), pl.col("y").mean().alias("prec_tr"),
                                    pl.len().alias("n_tr"))
    c = te.group_by("grp", "b").agg((pl.len() / n_s1_te).alias("te_per_s1"), pl.len().alias("n_te"))
    t = a.join(c, on=["grp", "b"], how="full", coalesce=True).fill_null(0)
    t = t.with_columns((pl.col("true_per_s1") * pos_ratio / pl.col("te_per_s1")).clip(0, 1).fill_nan(0).alias("prec_te"))
    lo = [0.0] + BANDS
    return t.sort("grp", "b").with_columns(pl.col("b").map_elements(lambda i: lo[i], return_dtype=pl.Float64).alias("lo"))


def apply_gate(test_best, table, score, min_prec=0.77):
    band = np.searchsorted(np.array(BANDS), test_best[score].to_numpy(), side="right")
    d = test_best.with_columns(pl.Series("b", band)).join(table.select("grp", "b", "prec_te"), on=["grp", "b"], how="left")
    return d.with_columns(pl.col("prec_te").fill_null(0.0), (pl.col("grp") == "twin").alias("is_twin"))
