"""Convert the challenge TSV files to parquet (read with an explicit tab separator, no quoting)."""
import sys

import polars as pl


def main(data, work):
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = pl.read_csv(f"{data}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False)
            df.write_parquet(f"{work}/{split}_s{s}.parquet")
    gt = pl.read_csv(f"{data}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
    gt.write_parquet(f"{work}/train_gt.parquet")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
