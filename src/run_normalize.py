"""Normalise every source file once and cache the result as parquet."""
import sys
import time
from multiprocessing import Pool

import polars as pl

from normalize import normalize_record

FIELDS = ["nm", "core", "skel", "alt", "legal", "dom", "nonlat", "ad", "nums", "pnum", "st", "city", "ad_empty"]


def _work(rows):
    names, addrs, countries = rows
    out = {f: [] for f in FIELDS}
    for n, a, c in zip(names, addrs, countries):
        d = normalize_record(n, a, c)
        for f in FIELDS:
            out[f].append(d[f])
    return out


def run(inp, outp, procs=4, chunk=100_000):
    df = pl.read_parquet(inp)
    n = df.height
    jobs = []
    for i in range(0, n, chunk):
        sl = df.slice(i, chunk)
        jobs.append((sl["business_name"].to_list(), sl["business_address"].to_list(), sl["country"].to_list()))
    del df
    parts = []
    with Pool(procs) as p:
        for r in p.imap(_work, jobs, chunksize=1):
            parts.append(pl.DataFrame(r))
    out = pl.concat([pl.read_parquet(inp), pl.concat(parts)], how="horizontal")
    out = out.with_columns(pl.col("dom").cast(pl.Int8), pl.col("nonlat").cast(pl.Int8), pl.col("ad_empty").cast(pl.Int8))
    out.write_parquet(outp)


if __name__ == "__main__":
    work = sys.argv[1]
    todo = sys.argv[2:] or ["train_1", "train_2", "train_3", "test_1", "test_2", "test_3"]
    for item in todo:
        split, s = item.split("_")
        t = time.time()
        run(f"{work}/{split}_s{s}.parquet", f"{work}/{split}_s{s}_norm.parquet")
        print(split, s, round(time.time() - t, 1), flush=True)
