"""Feature engineering for candidate pairs."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

QMETA_COLS = ["ad_empty", "nonlat", "dom", "src", "q_ntok", "q_natok"]


def query_meta(q):
    """Per-query flags. `q` must contain entity_id, core, ad, nums, dom, nonlat, ad_empty."""
    return q.select(
        pl.col("ad_empty").cast(pl.Int8), pl.col("nonlat").cast(pl.Int8), pl.col("dom").cast(pl.Int8),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"),
        pl.col("core").str.split(" ").list.len().cast(pl.Int16).alias("q_ntok"),
        (pl.col("ad").str.split(" ").list.len() + pl.col("nums").str.split(" ").list.len()).cast(pl.Int16).alias("q_natok"),
    )


STAGE1_FEATS = ["cos_n", "cos_a", "cos_c", "rk_n", "rk_a", "rk_c", "qmax_n", "qmax_a", "qmax_c", "gap_n", "gap_a",
                "gap_c", "urk_c", "urk_n", "urk_a", "marg_c", "n_cand"] + QMETA_COLS


def stage1_features(c, qmeta):
    """c: candidate pairs with qi, si, cos_n, cos_a, rk_*; qmeta: frame indexed by qi row."""
    c = c.with_columns(((pl.col("cos_n") + pl.col("cos_a")) / 2).alias("cos_c"))
    c = c.with_columns(
        pl.col("cos_n").max().over("qi").alias("qmax_n"),
        pl.col("cos_a").max().over("qi").alias("qmax_a"),
        pl.col("cos_c").max().over("qi").alias("qmax_c"),
        pl.len().over("qi").cast(pl.Int16).alias("n_cand"),
        pl.col("cos_c").rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias("urk_c"),
        pl.col("cos_n").rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias("urk_n"),
        pl.col("cos_a").rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias("urk_a"),
    )
    second = pl.col("cos_c").top_k(2).min().over("qi")
    c = c.with_columns(
        (pl.col("qmax_n") - pl.col("cos_n")).alias("gap_n"),
        (pl.col("qmax_a") - pl.col("cos_a")).alias("gap_a"),
        (pl.col("qmax_c") - pl.col("cos_c")).alias("gap_c"),
        pl.when(pl.col("urk_c") == 1).then(pl.col("cos_c") - second).otherwise(pl.col("cos_c") - pl.col("qmax_c")).alias("marg_c"),
    )
    qm = qmeta.with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
    return c.join(qm, on="qi", how="left")


# ------------------------------------------------------------------ stage 2 (string similarity)
S1_STR = ["core", "skel", "nm", "legal", "ad", "nums", "pnum", "st", "city"]
Q_STR = S1_STR + ["alt"]


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def string_features(c, s1s, qs, chunk=1_000_000):
    """c: pairs frame (qi, si ...). s1s / qs: frames holding the string columns, row-aligned to si / qi."""
    outs = []
    for start in range(0, c.height, chunk):
        part = c.slice(start, chunk)
        qi = part["qi"].to_numpy()
        si = part["si"].to_numpy()
        A = {k: qs[k].gather(qi) for k in Q_STR}
        B = {k: s1s[k].gather(si) for k in S1_STR}
        f = {}
        la = {k: A[k].fill_null("").to_list() for k in Q_STR}
        lb = {k: B[k].fill_null("").to_list() for k in S1_STR}
        f["n_ratio"] = _cp(la["core"], lb["core"], fuzz.ratio)
        f["n_tsort"] = _cp(la["core"], lb["core"], fuzz.token_sort_ratio)
        f["n_tset"] = _cp(la["core"], lb["core"], fuzz.token_set_ratio)
        f["n_part"] = _cp(la["core"], lb["core"], fuzz.partial_ratio)
        f["n_jw"] = _cp(la["core"], lb["core"], JaroWinkler.normalized_similarity)
        f["sk_ratio"] = _cp(la["skel"], lb["skel"], fuzz.ratio)
        f["sk_tset"] = _cp(la["skel"], lb["skel"], fuzz.token_set_ratio)
        f["sk_cat"] = _cp([x.replace(" ", "") for x in la["skel"]], [x.replace(" ", "") for x in lb["skel"]], fuzz.ratio)
        f["sk_catp"] = _cp([x.replace(" ", "") for x in la["skel"]], [x.replace(" ", "") for x in lb["skel"]], fuzz.partial_ratio)
        f["nm_tset"] = _cp(la["nm"], lb["nm"], fuzz.token_set_ratio)
        f["nm_ratio"] = _cp(la["nm"], lb["nm"], fuzz.ratio)
        f["alt_tset"] = _cp(la["alt"], lb["core"], fuzz.token_set_ratio)
        f["a_tset"] = _cp(la["ad"], lb["ad"], fuzz.token_set_ratio)
        f["a_tsort"] = _cp(la["ad"], lb["ad"], fuzz.token_sort_ratio)
        f["a_ratio"] = _cp(la["ad"], lb["ad"], fuzz.ratio)
        f["a_ptset"] = _cp(la["ad"], lb["ad"], fuzz.partial_token_set_ratio)
        f["num_tset"] = _cp(la["nums"], lb["nums"], fuzz.token_set_ratio)
        f["pnum_lev"] = _cp(la["pnum"], lb["pnum"], Levenshtein.distance).astype(np.float32)
        f["city_ratio"] = _cp(la["city"], lb["city"], fuzz.ratio)
        f["city_part"] = _cp(la["city"], lb["city"], fuzz.partial_ratio)
        fr = pl.DataFrame(f)
        a_nums = A["nums"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        b_nums = B["nums"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        a_ad = A["ad"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        b_ad = B["ad"].fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
        a_n = A["core"].fill_null("").str.split(" ")
        b_n = B["core"].fill_null("").str.split(" ")
        tmp = pl.DataFrame({"an": a_nums, "bn": b_nums, "aa": a_ad, "ba": b_ad, "anm": a_n, "bnm": b_n,
                            "ap": A["pnum"], "bp": B["pnum"], "ast": A["st"], "bst": B["st"], "al": A["legal"], "bl": B["legal"]})
        tmp = tmp.select(
            pl.col("an").list.len().cast(pl.Int16).alias("q_nnum"),
            pl.col("bn").list.len().cast(pl.Int16).alias("s_nnum"),
            pl.col("an").list.set_intersection("bn").list.len().cast(pl.Int16).alias("num_common"),
            pl.col("aa").list.set_intersection("ba").list.len().cast(pl.Int16).alias("ad_common"),
            pl.col("aa").list.len().cast(pl.Int16).alias("q_nad"),
            pl.col("ba").list.len().cast(pl.Int16).alias("s_nad"),
            pl.col("anm").list.set_intersection("bnm").list.len().cast(pl.Int16).alias("nm_common"),
            pl.col("anm").list.len().cast(pl.Int16).alias("q_nnm"),
            pl.col("bnm").list.len().cast(pl.Int16).alias("s_nnm"),
            (pl.col("anm").list.first() == pl.col("bnm").list.first()).cast(pl.Int8).alias("first_eq"),
            pl.when((pl.col("ap") == "") | (pl.col("bp") == "")).then(-1).when(pl.col("ap") == pl.col("bp")).then(1).otherwise(0).cast(pl.Int8).alias("pnum_eq"),
            pl.when((pl.col("ast") == "") | (pl.col("bst") == "")).then(-1).when(pl.col("ast") == pl.col("bst")).then(1).otherwise(0).cast(pl.Int8).alias("st_eq"),
            pl.when((pl.col("al") == "") | (pl.col("bl") == "")).then(-1).when(pl.col("al") == pl.col("bl")).then(1).otherwise(0).cast(pl.Int8).alias("legal_eq"),
            pl.col("bp").str.len_chars().cast(pl.Int8).alias("s_pnum_len"),
        )
        tmp = tmp.with_columns(
            (pl.col("num_common") / pl.max_horizontal(pl.col("q_nnum"), 1)).cast(pl.Float32).alias("num_cq"),
            (pl.col("num_common") / pl.max_horizontal(pl.col("s_nnum"), 1)).cast(pl.Float32).alias("num_cs"),
            (pl.col("ad_common") / pl.max_horizontal(pl.col("q_nad"), 1)).cast(pl.Float32).alias("ad_cq"),
            (pl.col("ad_common") / pl.max_horizontal(pl.col("s_nad"), 1)).cast(pl.Float32).alias("ad_cs"),
            (pl.col("nm_common") / pl.max_horizontal(pl.col("q_nnm"), 1)).cast(pl.Float32).alias("nm_cq"),
            (pl.col("nm_common") / pl.max_horizontal(pl.col("s_nnm"), 1)).cast(pl.Float32).alias("nm_cs"),
        )
        outs.append(pl.concat([fr, tmp], how="horizontal"))
    return pl.concat([c, pl.concat(outs)], how="horizontal")


STR_FEATS = ["n_ratio", "n_tsort", "n_tset", "n_part", "n_jw", "sk_ratio", "sk_tset", "sk_cat", "sk_catp", "nm_tset",
             "nm_ratio", "alt_tset", "a_tset", "a_tsort", "a_ratio", "a_ptset", "num_tset", "pnum_lev", "city_ratio",
             "city_part", "q_nnum", "s_nnum", "num_common", "ad_common", "q_nad", "s_nad", "nm_common", "q_nnm", "s_nnm",
             "first_eq", "pnum_eq", "st_eq", "legal_eq", "s_pnum_len", "num_cq", "num_cs", "ad_cq", "ad_cs", "nm_cq", "nm_cs"]


# ------------------------------------------------------------------ house-number geometry
# Hard negatives in this data are often *neighbouring* businesses (house number
# off by a small amount in the last digit) while true matches carry typo-style
# digit errors (often in the leading digit).  These features let the model tell
# the two apart.
NUM_FEATS = ["pn_absdiff", "pn_logdiff", "pn_reldiff", "pn_samelen", "pn_lendiff", "pn_hamming", "pn_lowdiff",
             "pn_highdiff", "pn_highdiff_rel", "pn_lastdig_eq", "pn_firstdig_eq", "pn_q_in_snums", "pn_s_in_qnums",
             "nums_minabs"]
_W = 10


def _digits_matrix(s):
    """Right-aligned fixed-width uint8 matrix of digit strings ('_' padding)."""
    s = s.fill_null("").str.slice(0, _W).str.pad_start(_W, "_")
    a = np.frombuffer("".join(s.to_list()).encode("ascii"), dtype=np.uint8).reshape(-1, _W)
    return a


def number_features(c, s1s, qs):
    qi = c["qi"].to_numpy()
    si = c["si"].to_numpy()
    pq = qs["pnum"].gather(qi).fill_null("")
    ps = s1s["pnum"].gather(si).fill_null("")
    both = ((pq != "") & (ps != "")).to_numpy()
    A = _digits_matrix(pq)
    B = _digits_matrix(ps)
    pad = np.uint8(ord("_"))
    present = (A != pad) | (B != pad)
    diff = (A != B) & present
    ndiff = diff.sum(1)
    pos = np.arange(_W)[::-1]  # position counted from the units digit
    low = np.where(diff.any(1), np.where(diff, pos, 99).min(1), -1)
    high = np.where(diff.any(1), np.where(diff, pos, -1).max(1), -1)
    lq = pq.str.len_chars().to_numpy()
    ls = ps.str.len_chars().to_numpy()
    iq = pq.str.slice(0, 12).cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    is_ = ps.str.slice(0, 12).cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    absd = np.where(both & (iq >= 0) & (is_ >= 0), np.abs(iq - is_), -1).astype(np.float64)
    mx = np.maximum(np.maximum(iq, is_), 1).astype(np.float64)
    out = {
        "pn_absdiff": np.minimum(absd, 1e7).astype(np.float32),
        "pn_logdiff": np.where(absd >= 0, np.log1p(np.maximum(absd, 0)), -1).astype(np.float32),
        "pn_reldiff": np.where(absd >= 0, absd / mx, -1).astype(np.float32),
        "pn_samelen": np.where(both, (lq == ls).astype(np.int8), -1).astype(np.int8),
        "pn_lendiff": np.where(both, np.abs(lq - ls), -1).astype(np.int8),
        "pn_hamming": np.where(both, ndiff, -1).astype(np.int8),
        "pn_lowdiff": np.where(both, low, -2).astype(np.int8),
        "pn_highdiff": np.where(both, high, -2).astype(np.int8),
        "pn_highdiff_rel": np.where(both & (high >= 0), (np.maximum(lq, ls) - 1 - high), -1).astype(np.int8),
        "pn_lastdig_eq": np.where(both, (A[:, -1] == B[:, -1]).astype(np.int8), -1).astype(np.int8),
        "pn_firstdig_eq": np.where(both, (pq.str.slice(0, 1) == ps.str.slice(0, 1)).to_numpy().astype(np.int8), -1).astype(np.int8),
    }
    qn = qs["nums"].gather(qi).fill_null("").str.split(" ")
    sn = s1s["nums"].gather(si).fill_null("").str.split(" ")
    tmp = pl.DataFrame({"pq": pq, "ps": ps, "qn": qn, "sn": sn}).select(
        pl.when(pl.col("pq") == "").then(-1).otherwise(pl.col("sn").list.contains(pl.col("pq")).cast(pl.Int8)).cast(pl.Int8).alias("pn_q_in_snums"),
        pl.when(pl.col("ps") == "").then(-1).otherwise(pl.col("qn").list.contains(pl.col("ps")).cast(pl.Int8)).cast(pl.Int8).alias("pn_s_in_qnums"),
    )
    # min |pnum_q - any number of s|
    sn_int = sn.list.eval(pl.element().str.slice(0, 12).cast(pl.Int64, strict=False))
    lens = sn_int.list.len().to_numpy()
    ex = pl.DataFrame({"g": np.arange(len(lens), dtype=np.int64), "q": iq, "sn": sn_int}).explode("sn")
    mins = (ex.with_columns((pl.col("sn") - pl.col("q")).abs().alias("d")).group_by("g").agg(pl.col("d").min()).sort("g"))
    m = np.full(len(lens), -1.0, dtype=np.float32)
    m[mins["g"].to_numpy()] = mins["d"].fill_null(-1).to_numpy().astype(np.float32)
    m[iq < 0] = -1
    out["pn_q_in_snums"] = tmp["pn_q_in_snums"].to_numpy()
    out["pn_s_in_qnums"] = tmp["pn_s_in_qnums"].to_numpy()
    out["nums_minabs"] = np.minimum(m, 1e7).astype(np.float32)
    return pl.DataFrame(out)
