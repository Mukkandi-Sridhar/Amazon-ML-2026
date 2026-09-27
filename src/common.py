"""Shared helpers: id <-> row-index mappings and ground-truth loading."""
import numpy as np
import polars as pl


def query_ids(work, split):
    return pl.concat([pl.read_parquet(f"{work}/{split}_s{i}_norm.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]


def s1_ids(work, split):
    return pl.read_parquet(f"{work}/{split}_s1_norm.parquet", columns=["entity_id"])["entity_id"]


def truth_pairs(work, gt_path):
    """Ground-truth (qi, si) pairs for the train split, in row-index space."""
    gt = (pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
          if gt_path.endswith(".tsv") else pl.read_parquet(gt_path))
    gt = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
          .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    q = query_ids(work, "train")
    s = s1_ids(work, "train")
    qid = pl.DataFrame({"qi": np.arange(len(q), dtype=np.int32), "matched_entity_ids": q})
    sid = pl.DataFrame({"si": np.arange(len(s), dtype=np.int32), "source1_entity_id": s})
    return gt.join(qid, on="matched_entity_ids").join(sid, on="source1_entity_id").select("qi", "si")
