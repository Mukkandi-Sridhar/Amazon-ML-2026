"""Full blocking for a split; candidate chunks are streamed to <work>/<split>_cand/part-XXXX.parquet."""
import os
import sys
import time

import polars as pl

from blocking import add_tokens, block


def main(work, split):
    t = time.time()
    s1 = pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["entity_id", "country", "core", "alt", "ad", "nums"])
    q = pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["entity_id", "country", "core", "alt", "ad", "nums"])
                   for i in (2, 3)])
    print("loaded", round(time.time() - t), flush=True)
    out = f"{work}/{split}_cand"
    os.makedirs(out, exist_ok=True)
    n = [0]

    def sink(df):
        df.write_parquet(f"{out}/part-{n[0]:04d}.parquet")
        n[0] += 1

    block(s1, q, log=lambda m: print(m, flush=True), sink=sink, tokenize=add_tokens)
    print("done", round(time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
