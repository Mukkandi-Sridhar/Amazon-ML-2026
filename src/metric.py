"""Competition metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import polars as pl


def macro_f05(pred, truth, universe):
    """pred / truth: frames (si, qi) of matched pairs. universe: Series/list of all evaluated si."""
    u = pl.DataFrame({"si": universe}).unique()
    tp = pred.join(truth, on=["si", "qi"], how="inner").group_by("si").len().rename({"len": "tp"})
    npred = pred.group_by("si").len().rename({"len": "np"})
    ntrue = truth.group_by("si").len().rename({"len": "nt"})
    d = u.join(tp, on="si", how="left").join(npred, on="si", how="left").join(ntrue, on="si", how="left").fill_null(0)
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("nt")
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
         .when((pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
         .otherwise(1.25 * prec * rec / (0.25 * prec + rec)))
    d = d.with_columns(f.alias("f"))
    return float(d["f"].mean()), d
