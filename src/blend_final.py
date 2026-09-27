"""Blend the stage-3 scores of two full pipeline runs (v5 and v4) before the final decision.

Usage: python blend_final.py <v5_scores.parquet> <v4_scores.parquet> <out.parquet>
Pairs are those of the v5 run (its candidate set); where v4 also scored a pair the two
p3 scores are averaged 50/50, otherwise the v5 score is kept.  Train OOF macro F0.5:
v5 0.98419, v4 0.98412, blend 0.98444.  Then run run_final2.py with score column "pb".
"""
import sys

import polars as pl


def main(p5, p4, out, w=0.5):
    w = float(w)
    cols = [c for c in ("qi", "si", "p3", "y") if c in pl.read_parquet_schema(p5)]
    d = pl.read_parquet(p5, columns=cols).join(pl.read_parquet(p4, columns=["qi", "si", "p3"]).rename({"p3": "p4"}),
                                               on=["qi", "si"], how="left")
    d.with_columns(pl.when(pl.col("p4").is_null()).then(pl.col("p3"))
                   .otherwise(w * pl.col("p3") + (1 - w) * pl.col("p4")).alias("pb")).write_parquet(out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
