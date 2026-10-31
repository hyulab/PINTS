"""Differential tests: optimised `pints` vs frozen `pints_ref` must agree BITWISE on random inputs.

These are the main guard against the agent loop over-fitting to the tuning datasets: they exercise
branches (LER cache states, chromosome edges, tiny windows, dtype variants, parser variants) that a
handful of real datasets may never reach.

Set PINTS_DIFF_EXAMPLES to scale the number of hypothesis examples (default 25).
"""
import gzip
import os
import struct
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints.stats_engine as se
import pints.io_engine as io
import pints_ref.calling_engine as rce
import pints_ref.stats_engine as rse
import pints_ref.io_engine as rio
from helpers import synth_coverage, write_subpeak_tabix, make_bam, make_bigwig

N = int(os.environ.get("PINTS_DIFF_EXAMPLES", "25"))
FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))
SLOW = settings(max_examples=max(3, N // 2), deadline=None, suppress_health_check=list(HealthCheck))
# production coverage arrays are always saved as signed ints (io_engine.NP_DT_NAME); the LER code
# writes -1 sentinels into same-dtype copies, so unsigned dtypes are out of contract.
DTYPES = [np.int8, np.int16, np.int32, np.int64]
CHR = "chrD"


# ------------------------------------------------------------------ bitwise comparison
def same(a, b, path="root"):
    if isinstance(a, pd.DataFrame):
        assert isinstance(b, pd.DataFrame), path
        assert list(a.columns) == list(b.columns), path
        assert a.shape == b.shape, path
        assert a.index.equals(b.index), path
        for c in a.columns:
            same(a[c].to_numpy(), b[c].to_numpy(), path + "." + str(c))
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, (path, a.dtype, b.dtype)
        assert a.shape == b.shape, path
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                same(x, y, "%s[%d]" % (path, i))
        else:
            assert a.tobytes() == b.tobytes(), path
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (path, i))
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            same(a[k], b[k], "%s{%s}" % (path, k))
        return
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        assert type(a) == type(b), (path, type(a), type(b))
        assert struct.pack("<d", float(a)) == struct.pack("<d", float(b)), (path, a, b)
        return
    assert type(a) == type(b), (path, type(a), type(b))
    assert a == b, (path, a, b)


def file_bytes(p):
    if p is None or not os.path.exists(p):
        return None
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as fh:
            return fh.read()
    with open(p, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------ strategies
cov_params = st.fixed_dictionaries({
    "seed": st.integers(0, 2 ** 31 - 1),
    "length": st.integers(2_000, 60_000),
    "n_peaks": st.integers(0, 60),
    "bg_rate": st.sampled_from([0.0, 0.005, 0.02, 0.1, 0.5]),
    "bg_pi": st.sampled_from([0.0, 0.5, 0.9, 0.99]),
    "dtype": st.sampled_from(DTYPES),
})


def make_cov(p):
    rng = np.random.default_rng(p["seed"])
    c = synth_coverage(rng, length=p["length"], n_peaks=p["n_peaks"], bg_rate=p["bg_rate"],
                       bg_pi=p["bg_pi"], dtype=np.int64)
    info = np.iinfo(p["dtype"])
    return np.clip(c, 0, info.max).astype(p["dtype"])


small_counts = st.lists(st.integers(0, 30), min_size=1, max_size=400).map(lambda x: np.array(x))


@st.composite
def zipish(draw):
    seed = draw(st.integers(0, 2 ** 31 - 1))
    n = draw(st.integers(1, 3000))
    pi = draw(st.floats(0, 0.999))
    lam = draw(st.floats(0.01, 40))
    dt = draw(st.sampled_from(DTYPES))
    rng = np.random.default_rng(seed)
    x = (rng.random(n) > pi) * rng.poisson(lam, n)
    return np.clip(x, 0, np.iinfo(dt).max).astype(dt)


# ------------------------------------------------------------------ pure functions
@FAST
@given(st.lists(st.tuples(st.integers(0, 500), st.integers(0, 60)), max_size=40), st.integers(0, 5))
def test_merge_intervals(raw, d):
    iv = [(s, s + w) for s, w in raw]
    same(ce.merge_intervals(list(iv), d), rce.merge_intervals(list(iv), d))


@FAST
@given(cov_params, st.integers(1, 300), st.integers(1, 300))
def test_sliding_window(p, w, s):
    cov = make_cov(p)
    same(list(ce.sliding_window(cov, w, s)), list(rce.sliding_window(cov, w, s)))


@FAST
@given(st.lists(st.integers(0, 50), min_size=3, max_size=600), st.floats(0, 1.5), st.integers(1, 6),
       st.integers(0, 3))
def test_cut_peaks(w, dt, cet, jd):
    w = np.array(w)
    same(ce.cut_peaks(w, dt, cet, join_distance=jd), rce.cut_peaks(w, dt, cet, join_distance=jd))


@FAST
@given(st.lists(st.tuples(st.integers(0, 1000), st.integers(1, 40), st.integers(1, 500)), max_size=30),
       st.floats(0, 1.5), st.integers(1, 6), st.integers(1, 60))
def test_quasi_max_score_segment(raw, dt, cet, md):
    raw = sorted(raw)
    c = [(s, s + w, n, n / w) for s, w, n in raw]
    same(ce.quasi_max_score_segment(c, dt, cet, md), rce.quasi_max_score_segment(c, dt, cet, md))


@FAST
@given(zipish())
def test_zip_fit(x):
    same(se.ZIP().fit(x), rse.ZIP().fit(x))


@FAST
@given(small_counts)
def test_zip_fit_small(x):
    same(se.ZIP().fit(x), rse.ZIP().fit(x))
    same(se.ZIP.zip_moment_estimators(x), rse.ZIP.zip_moment_estimators(x))


@FAST
@given(zipish(), st.floats(0, 1), st.floats(0, 60))
def test_zip_log_likelihood(x, pi, lam):
    same(se.ZIP.log_likelihood(x, pi, lam), rse.ZIP.log_likelihood(x, pi, lam))


@FAST
@given(zipish(), zipish())
def test_zip_lrt(a, b):
    same(se.ZIP().lrt(a, b), rse.ZIP().lrt(a, b))


@FAST
@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=1, max_size=50), st.sampled_from([-1, 0, 1]))
def test_iqr_threshold(d, direction):
    same(se.IQR.get_outlier_threshold(d, direction), rse.IQR.get_outlier_threshold(d, direction))


@FAST
@given(st.lists(st.tuples(st.floats(-100, 100), st.floats(-100, 100)), min_size=0, max_size=40))
def test_get_elbow(pts):
    X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        same(se.get_elbow(X, Y), rse.get_elbow(X, Y))


# ------------------------------------------------------------------ LER with cache evolution
@st.composite
def ler_case(draw):
    p = draw(cov_params.filter(lambda q: q["length"] > 12_000))
    cov = make_cov(p)
    L = len(cov)
    rng = np.random.default_rng(p["seed"] + 1)
    n = draw(st.integers(0, 40))
    starts = np.sort(rng.integers(0, L - 5, n))
    rows = []
    for i, s in enumerate(starts):
        e = int(min(L, s + rng.integers(1, 80)))
        mu = float(np.round(rng.gamma(1.0, 3.0), 6)) if rng.random() > 0.1 else 0.0
        rows.append((int(s), e, "p%d" % i, mu, 0.1, "+", 0.2, 1, int(s), 1))
    queries = []
    for _ in range(draw(st.integers(1, 8))):
        a = int(rng.integers(0, L))
        b = int(min(L, a + rng.integers(1, 200)))
        f = int(draw(st.sampled_from([500, 2500, 5000])))
        queries.append((max(a - f, 0), max(a - 1, 0), min(b + 1, L), min(b + f, L)))
    flags = draw(st.fixed_dictionaries({"disable_ler": st.booleans(), "enable_eler": st.booleans(),
                                        "peak_threshold": st.sampled_from([0.5, 2.0, 5.0, 50.0]),
                                        "pib": st.sampled_from([0, 1, 2])}))
    return cov, rows, queries, flags


@pytest.mark.parametrize("strategy", ["bgIQR", "pkIQR"])
@SLOW
@given(case=ler_case())
def test_remove_peaks_in_local_env(strategy, case):
    cov, rows, queries, fl = case
    with tempfile.TemporaryDirectory() as d:
        if not rows:
            rows = [(len(cov) - 2, len(cov) - 1, "x", 0.0, 0.1, "+", 0.2, 1, 0, 1)]
        gz = write_subpeak_tabix(os.path.join(d, "sp.bed"), CHR, rows)
        tbx_a = pysam.TabixFile(gz, index=gz + ".csi")
        tbx_b = pysam.TabixFile(gz, index=gz + ".csi")
        A = getattr(se, strategy); B = getattr(rse, strategy)
        ca, cb = {}, {}
        for q in queries:
            ra = A.remove_peaks_in_local_env(se.ZIP(), tbx_a, CHR, *q, 5, fl["pib"], cov, 0.1, ca,
                                             disable_ler=fl["disable_ler"], enable_eler=fl["enable_eler"],
                                             peak_threshold=fl["peak_threshold"])
            rb = B.remove_peaks_in_local_env(rse.ZIP(), tbx_b, CHR, *q, 5, fl["pib"], cov, 0.1, cb,
                                             disable_ler=fl["disable_ler"], enable_eler=fl["enable_eler"],
                                             peak_threshold=fl["peak_threshold"])
            same(ra, rb)
            same(ca, cb)


# ------------------------------------------------------------------ chromosome pipeline
def _set_globals(model="ZIP", iqr="bgIQR"):
    ce.stat_tester = getattr(se, model)()
    rce.stat_tester = getattr(rse, model)()
    ce.iqr_obj = getattr(se, iqr)()
    rce.iqr_obj = getattr(rse, iqr)()


pipeline_flags = st.fixed_dictionaries({
    "sensitive": st.booleans(), "disable_ler": st.booleans(), "enable_eler": st.booleans(),
    "iqr": st.sampled_from(["bgIQR", "pkIQR"]), "small": st.sampled_from([3, 5, 8]),
    "min_mu_percent": st.sampled_from([0.1, 0.4]), "disable_qc": st.booleans(),
    "donor": st.sampled_from([0.0, 0.3, 1.0]),
})


def _run_chrom(mod, d, cov_list, strand, fl):
    files = []
    for i, cov in enumerate(cov_list):
        f = os.path.join(d, "cov%d.npy" % i)
        np.save(f, cov)
        files.append({CHR: f})
    out = os.path.join(d, "s_%s.bed" % ("pl" if strand == "+" else "mn"))
    sp = mod.unified_element_definition(files, CHR, strand, out, ce_donor=fl["donor"])
    if sp is None:
        return None, None, None
    df = mod.check_window_chromosome(files[0][CHR], out, strand, CHR, sp, 0.1,
                                     small_peak_threshold=fl["small"], min_mu_percent=fl["min_mu_percent"],
                                     disable_qc=fl["disable_qc"], disable_ler=fl["disable_ler"],
                                     enable_eler=fl["enable_eler"], sensitive=fl["sensitive"])
    spb = out.replace(".bed", ce.__SUB_PEAK_TPL__ % CHR) + ".gz"
    return file_bytes(sp), df, file_bytes(spb)


@SLOW
@given(cov_params.filter(lambda q: q["length"] > 15_000), pipeline_flags, st.sampled_from(["+", "-"]))
def test_chromosome_pipeline(p, fl, strand):
    _set_globals(iqr=fl["iqr"])
    cov = make_cov(p)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        ra = _run_chrom(ce, da, [cov], strand, fl)
        rb = _run_chrom(rce, db, [cov], strand, fl)
        same(ra[0], rb[0])
        same(ra[1], rb[1])
        same(ra[2], rb[2])


@SLOW
@given(cov_params.filter(lambda q: q["length"] > 15_000), st.integers(2, 3))
def test_replicate_element_definition(p, nrep):
    covs = [make_cov(dict(p, seed=p["seed"] + k)) for k in range(nrep)]
    fl = dict(donor=0.3)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        out = []
        for mod, d in ((ce, da), (rce, db)):
            files = []
            for i, c in enumerate(covs):
                f = os.path.join(d, "c%d.npy" % i); np.save(f, c); files.append({CHR: f})
            sp = mod.unified_element_definition(files, CHR, "+", os.path.join(d, "r_pl.bed"))
            out.append(None if sp is None else file_bytes(sp))
        same(out[0], out[1])


# ------------------------------------------------------------------ filtering
@st.composite
def peak_table(draw):
    seed = draw(st.integers(0, 2 ** 31 - 1))
    n = draw(st.integers(25, 3000))
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(10 ** 7, n, replace=False))
    lens = rng.integers(1, 120, n)
    reads = rng.integers(1, 500, n)
    sig = rng.random(n) < draw(st.floats(0, 0.5))
    p = np.where(sig, rng.uniform(0, 1e-4, n), rng.uniform(0, 1, n))
    df = pd.DataFrame({"chromosome": np.where(rng.random(n) < 0.5, "chr1", "chr2"), "start": starts,
                       "end": starts + lens, "name": ["n%d" % i for i in range(n)], "pval": p,
                       "strand": "+", "reads": reads, "mu_0": rng.random(n), "pi_0": rng.random(n),
                       "mu_1": rng.gamma(1, 2, n), "pi_1": rng.random(n), "var_1": rng.random(n),
                       "summit": starts, "summit_val": rng.integers(1, 30, n), "ler_1": 0, "ler_2": 0,
                       "ler_3": 0, "non_zeros": rng.integers(1, 30, n)})
    variant = draw(st.sampled_from(["plain", "dup_index", "float_reads", "coarse_reads"]))
    if variant == "dup_index":       # as produced by pd.concat of per-chromosome frames
        cut = int(rng.integers(1, n))
        df.index = list(range(cut)) + list(range(n - cut))
    elif variant == "float_reads":   # input-normalised / bigWig signal
        df["reads"] = reads * rng.uniform(0.3, 1.7)
    elif variant == "coarse_reads":  # many tied read counts -> many identical quantile thresholds
        df["reads"] = rng.integers(1, 4, n)
    return df


@SLOW
@given(peak_table(), st.booleans(), st.sampled_from([0.5, 0.75, 0.95]), st.booleans())
def test_stratified_filtering(df, disable_small, tpt, disable_qc):
    import logging
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        kw = dict(disable_small=disable_small, top_peak_threshold=tpt, disable_qc=disable_qc)
        fa = ce.stratified_filtering(df.copy(), os.path.join(da, "x_pl.bed"), 0.1, **kw)
        fb = rce.stratified_filtering(df.copy(), os.path.join(db, "x_pl.bed"), 0.1, **kw)
        same(file_bytes(fa), file_bytes(fb))
        same(sorted(os.listdir(da)), sorted(os.listdir(db)))
        for f in os.listdir(da):
            if f.endswith(".tpt"):
                same(file_bytes(os.path.join(da, f)), file_bytes(os.path.join(db, f)))


@SLOW
@given(peak_table())
def test_independent_filtering(df):
    import logging
    lg = logging.getLogger("t")
    same(ce.independent_filtering(df.copy(), 0.1, logger=lg), rce.independent_filtering(df.copy(), 0.1, logger=lg))


# ------------------------------------------------------------------ pairing
@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.booleans(), st.sampled_from([0, 10]), st.sampled_from([0, 50]))
def test_pair(seed, stringent, min_len, div_min):
    rng = np.random.default_rng(seed)
    dfs = {}
    for strand, lab in (("+", "pl"), ("-", "mn")):
        n = int(rng.integers(5, 300))
        s = np.sort(rng.choice(200_000, n, replace=False))
        e = s + rng.integers(1, 100, n)
        padj = np.where(rng.random(n) < 0.4, rng.uniform(0, 0.05, n), rng.uniform(0, 1, n))
        dfs[lab] = pd.DataFrame({"chromosome": "chr1", "start": s, "end": e, "name": ".", "padj": padj,
                                 "strand": strand, "reads": rng.integers(1, 100, n), "pval": padj / 3,
                                 "mu_0": 0.1, "pi_0": 0.1, "mu_1": 1.0, "pi_1": 0.1, "var_1": 0.1, "ler_1": 0,
                                 "ler_2": 0, "ler_3": 0, "non_zeros": 3, "summit": s + (e - s) // 2,
                                 "summit_val": rng.integers(1, 50, n)}).loc[:, ce.COMMON_HEADER]
    res = []
    for mod in (ce, rce):
        with tempfile.TemporaryDirectory() as d:
            # candidate files (all tested peaks per strand), as written by the per-strand testing step
            for lab, df in dfs.items():
                fn = os.path.join(d, "s_%s.bed" % lab)
                df.sort_values(by=["chromosome", "start"]).to_csv(fn, sep="\t", index=False, header=False)
                mod.index_bed_file(fn)
            (b, dv, u), _ = mod.pair(os.path.join(d, "s"), {k: v.copy() for k, v in dfs.items()}, 0.1, stringent,
                                     min_len_opposite_peaks=min_len, div_size_min=div_min)
            res.append([file_bytes(x) for x in (b, dv, u)])
    same(res[0], res[1])


# ------------------------------------------------------------------ IO
@st.composite
def random_reads(draw):
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.integers(1, 800))
    reads = []
    contigs = [("chr1", 20_000), ("chr2", 5_000), ("ctgA", 3_000)]
    for i in range(n):
        c, L = contigs[int(rng.integers(0, 3))]
        ln = int(rng.integers(10, 60))
        pos = int(rng.integers(0, L - ln - 5))
        cig = rng.choice(["%dM" % ln, "3S%dM" % (ln - 3), "%dM2D%dM" % (ln // 2, ln - ln // 2),
                          "%dM4S" % (ln - 4)])
        r1 = bool(rng.random() < 0.5)
        reads.append(dict(contig=c, pos=pos, cigar=str(cig), reverse=bool(rng.random() < 0.5),
                          mapq=int(rng.choice([0, 10, 29, 30, 31, 60])), paired=True, read1=r1, read2=not r1,
                          secondary=bool(rng.random() < 0.05)))
    return contigs, reads


@SLOW
@given(random_reads(), st.sampled_from(["R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3", "PROseq", "CoPRO"]),
       st.booleans(), st.sampled_from(["", "chr"]))
def test_get_read_signal(rr, lib, rc, prefix):
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        a = io.get_read_signal(bam, lib, prefix, os.path.join(d, "a"), "p", reverse_complement=rc)
        b = rio.get_read_signal(bam, lib, prefix, os.path.join(d, "b"), "p", reverse_complement=rc)
        same(a[2], b[2])
        for k in (0, 1):
            assert set(a[k]) == set(b[k])
            for c in a[k]:
                same(np.load(a[k][c]), np.load(b[k][c]))


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([5, 100, 300, 70000]))
def test_get_coverage_bw(seed, vmax):
    rng = np.random.default_rng(seed)
    chroms = [("chr1", 5000), ("chr2", 2000)]
    pv = {c: np.where(rng.random(L) < 0.05, rng.integers(1, vmax, L), 0).astype(float) for c, L in chroms}
    mv = {c: -np.where(rng.random(L) < 0.05, rng.integers(1, vmax, L), 0).astype(float) for c, L in chroms}
    with tempfile.TemporaryDirectory() as d:
        p = make_bigwig(os.path.join(d, "p.bw"), chroms, pv)
        m = make_bigwig(os.path.join(d, "m.bw"), chroms, mv)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        a = io.get_coverage_bw(p, m, "", os.path.join(d, "a"), "q")
        b = rio.get_coverage_bw(p, m, "", os.path.join(d, "b"), "q")
        same(a[2], b[2])
        for k in (0, 1):
            for c in a[k]:
                same(np.load(a[k][c]), np.load(b[k][c]))
