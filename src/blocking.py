"""Candidate generation (blocking).

Every Source-2 / Source-3 record is a *query*; every Source-1 record is an
*index entry*.  Both are embedded as two sparse IDF-weighted bags of tokens:

* name channel   : normalised words, phonetic skeletons, concatenated skeleton
                   (catches ``beaconbiotechnologies.com``), DBA alias words
* address channel: address words and house/plot numbers

Three top-K cosine searches are run per country label (an open set -- any label
in the data is treated the same way): name only, address only, and the average
of both.  Their union is the candidate set; the exact name and address cosines
of every candidate pair are kept as features for the matcher.
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from normalize import skel


def _skel_map(values):
    return {v: skel(v) for v in values}


def add_tokens(df):
    """Adds `ntok` (name channel) and `atok` (address channel) list columns."""
    words = (df.select(pl.concat_list(pl.col("core").str.split(" "), pl.col("alt").str.split(" ")).explode().unique())
             .to_series().drop_nulls().to_list())
    wmap = _skel_map(words)
    cores = df.select(pl.col("core").unique()).to_series().to_list()
    cmap = {c: skel(c.replace(" ", "")) for c in cores}
    split_core = pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    w = split_core.list.eval("w:" + pl.element())
    k = split_core.list.eval("k:" + pl.element().replace_strict(wmap, default=""))
    alt = pl.col("alt").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
    aw = alt.list.eval("w:" + pl.element())
    ak = alt.list.eval("k:" + pl.element().replace_strict(wmap, default=""))
    c = pl.concat_list([pl.lit("c:") + pl.col("core").replace_strict(cmap, default="")])
    a = pl.col("ad").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.eval("a:" + pl.element())
    d = pl.col("nums").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 1)).list.eval("d:" + pl.element())
    ntok = pl.concat_list([w, k, aw, ak, c]).list.unique().list.eval(pl.element().filter(pl.element().str.len_chars() > 2))
    df = df.with_columns(ntok.alias("ntok"), pl.col("core").replace_strict(cmap, default="").alias("_cskel"),
                         pl.col("skel").fill_null("").alias("_skel"))
    # address bigrams: consecutive normalised address words (common Indian address words become specific in pairs)
    bg = (pl.col("ad").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
          .list.eval(("b:" + pl.element() + "_" + pl.element().shift(-1)).drop_nulls()))
    atok = pl.concat_list([a, d, bg]).list.unique()
    pn = pl.col("pnum").fill_null("")
    ok = pn != ""
    df = df.with_columns(
        atok.alias("atok"),
        pl.when(ok & (pl.col("core") != "")).then(pl.col("core") + "|" + pn).alias("key_core_pn"),
        pl.when(ok & (pl.col("_skel") != "")).then(pl.col("_skel") + "|" + pn).alias("key_skel_pn"),
        pl.when(ok & (pl.col("_cskel") != "")).then(pl.col("_cskel") + "|" + pn).alias("key_cat_pn"),
        pl.when(pl.col("core") != "").then(pl.col("core")).alias("key_core"),
        pl.when(pl.col("_skel") != "").then(pl.col("_skel")).alias("key_skel"),
        pl.when(pl.col("_cskel").str.len_chars() >= 4).then(pl.col("_cskel")).alias("key_cat"),
    )
    return df.drop("_cskel", "_skel")


def _csr(tok_series, vocab_df, n_rows):
    ex = pl.DataFrame({"row": np.arange(n_rows, dtype=np.int32), "tok": tok_series}).explode("tok").drop_nulls()
    ex = ex.join(vocab_df, on="tok", how="inner")
    m = sp.csr_matrix((ex["idf"].to_numpy().astype(np.float32), (ex["row"].to_numpy(), ex["col"].to_numpy())),
                      shape=(n_rows, vocab_df.height), dtype=np.float32)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return sp.diags((1.0 / norm).astype(np.float32)).dot(m).tocsr()


def _vocab(tok_series, n, max_df):
    v = (pl.DataFrame({"tok": tok_series}).select(pl.col("tok").explode()).drop_nulls().group_by("tok").len()
         .filter(pl.col("len") <= max_df))
    v = v.with_columns(np.log(n / pl.col("len")).cast(pl.Float32).alias("idf"))
    return v.with_row_index("col").with_columns(pl.col("col").cast(pl.Int32)).select("tok", "col", "idf")


def _topk(Q, ST, k, threads):
    C = sp_matmul_topn(Q, ST, top_n=k, threshold=1e-6, sort=True, n_threads=threads).tocoo()
    order = np.lexsort((-C.data, C.row))
    r, c = C.row[order], C.col[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1] if len(r) else np.array([], dtype=np.int64)
    rank = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)])) if len(r) else np.array([])
    return pl.DataFrame({"r": r.astype(np.int32), "c": c.astype(np.int32), "rk": rank.astype(np.int16)})


def _rowdot(A, B, ai, bi):
    out = np.empty(len(ai), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(ai), step):
        out[s:s + step] = np.asarray(A[ai[s:s + step]].multiply(B[bi[s:s + step]]).sum(axis=1)).ravel()
    return out


KEYS = {  # key column -> max number of Source-1 records sharing the key for it to be used
    "key_core_pn": 50, "key_skel_pn": 50, "key_cat_pn": 50, "key_core": 10, "key_skel": 10, "key_cat": 10}
KEY_FLAGS = ["kf_" + k[4:] for k in KEYS]


def _keyed(qs, s1c, key_frames):
    """Exact-key candidates for a chunk of queries: (r, c, kf_* flags)."""
    out = None
    for k, g in key_frames.items():
        qk = qs.select(pl.col(k)).with_row_index("r").drop_nulls()
        m = qk.join(g, on=k, how="inner").select(pl.col("r").cast(pl.Int32), pl.col("c").cast(pl.Int32),
                                                   pl.lit(1, pl.Int8).alias("kf_" + k[4:]))
        out = m if out is None else out.join(m, on=["r", "c"], how="full", coalesce=True)
    return out.with_columns([pl.col(f).fill_null(0) for f in KEY_FLAGS])


def block(s1, q, kn=10, ka=10, kc=15, max_df_frac=0.01, min_max_df=2000, chunk=400_000, threads=4, log=print, sink=None,
          tokenize=None):
    """s1, q: normalised frames (with `ntok`/`atok`, or raw fields plus a `tokenize` callable applied lazily per
    country / chunk to keep memory low). Returns frame of candidate pairs (row indices + channel scores)."""
    out = []
    for country in sorted(set(q["country"].unique().to_list())):
        t0 = time.time()
        s1c_idx = np.flatnonzero((s1["country"] == country).to_numpy())
        qc_idx = np.flatnonzero((q["country"] == country).to_numpy())
        if len(s1c_idx) == 0 or len(qc_idx) == 0:
            continue
        s1c = s1[s1c_idx]
        if tokenize is not None:
            s1c = tokenize(s1c)
        n = s1c.height
        max_df = max(min_max_df, int(max_df_frac * n))
        vn = _vocab(s1c["ntok"], n, max_df)
        va = _vocab(s1c["atok"], n, max_df)
        SN = _csr(s1c["ntok"], vn, n)
        SA = _csr(s1c["atok"], va, n)
        SNT, SAT = SN.T.tocsr(), SA.T.tocsr()
        SCT = (sp.hstack([SN, SA]).tocsr() * np.float32(1 / np.sqrt(2))).T.tocsr()
        key_frames = {}
        for k, cap in KEYS.items():
            g = s1c.select(pl.col(k)).with_row_index("c").drop_nulls()
            g = g.with_columns(pl.len().over(k).alias("_n")).filter(pl.col("_n") <= cap).drop("_n")
            key_frames[k] = g
        log(f"  [{country}] S1={n} Q={len(qc_idx)} vocab n={vn.height} a={va.height} build={time.time()-t0:.0f}s")
        for i in range(0, len(qc_idx), chunk):
            t1 = time.time()
            qi = qc_idx[i:i + chunk]
            qs = q[qi]
            if tokenize is not None:
                qs = tokenize(qs)
            QN = _csr(qs["ntok"], vn, len(qi))
            QA = _csr(qs["atok"], va, len(qi))
            QC = (sp.hstack([QN, QA]).tocsr() * np.float32(1 / np.sqrt(2))).tocsr()
            pn = _topk(QN, SNT, kn, threads).rename({"rk": "rk_n"})
            pa = _topk(QA, SAT, ka, threads).rename({"rk": "rk_a"})
            pc = _topk(QC, SCT, kc, threads).rename({"rk": "rk_c"})
            u = pc.join(pn, on=["r", "c"], how="full", coalesce=True).join(pa, on=["r", "c"], how="full", coalesce=True)
            u = u.join(_keyed(qs, s1c, key_frames), on=["r", "c"], how="full", coalesce=True)
            u = u.with_columns([pl.col(f).fill_null(0) for f in KEY_FLAGS])
            r = u["r"].to_numpy()
            c = u["c"].to_numpy()
            cos_n = _rowdot(QN, SN, r, c)
            cos_a = _rowdot(QA, SA, r, c)
            res = pl.DataFrame({
                "qi": qi[r].astype(np.int32), "si": s1c_idx[c].astype(np.int32),
                "cos_n": cos_n, "cos_a": cos_a,
                "rk_n": u["rk_n"].fill_null(99).cast(pl.Int16), "rk_a": u["rk_a"].fill_null(99).cast(pl.Int16),
                "rk_c": u["rk_c"].fill_null(99).cast(pl.Int16)}).with_columns([u[f].cast(pl.Int8) for f in KEY_FLAGS])
            if sink is not None:
                sink(res)
            else:
                out.append(res)
            log(f"    chunk {i}: {time.time()-t1:.0f}s pairs={u.height} ({u.height/len(qi):.1f}/q)")
    return pl.concat(out) if out else None
