"""Differential tests for the chunked / pooled window-evaluation engine (evaluate_strand_jobs):
in-memory SubpeakIndex vs pysam.TabixFile, forced phase-1 block splitting and phase-2 gap-based chunking
under a process pool, and multi-job (several samples x strands) scheduling. Outputs are compared
bitwise against pints_ref."""
import os
import tempfile

import numpy as np
import pysam
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints.stats_engine as se
import pints_ref.calling_engine as rce
import pints_ref.stats_engine as rse
from helpers import synth_coverage
from test_differential import same, file_bytes, N

FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))
SLOW = settings(max_examples=max(3, N // 3), deadline=None, suppress_health_check=list(HealthCheck))


def _globals(iqr="bgIQR"):
    ce.stat_tester, rce.stat_tester = se.ZIP(), rse.ZIP()
    ce.iqr_obj, rce.iqr_obj = getattr(se, iqr)(), getattr(rse, iqr)()


# ------------------------------------------------------------------ SubpeakIndex == tabix
@FAST
@given(st.integers(0, 2 ** 31 - 1), st.integers(0, 60))
def test_subpeak_index_matches_tabix(seed, n):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(50_000, n, replace=False))
    lens = rng.integers(1, rng.choice([5, 300, 20_000]), n)
    lines = ["chrZ\t%d\t%d\tp%d\t%f\t0.1\t+\t0.2\t3\t%d\t7\n" % (s, s + l, i, rng.random(), s)
             for i, (s, l) in enumerate(zip(starts, lens))]
    idx = ce.SubpeakIndex("chrZ", lines)
    assert idx.is_supported()
    with tempfile.TemporaryDirectory() as d:
        bed = os.path.join(d, "x.bed")
        with open(bed, "w") as fh:
            fh.writelines(lines)
        if n == 0:
            return  # tabix cannot index an empty file; empty index returns nothing
        ce.index_bed_file(bed)
        tbx = pysam.TabixFile(bed + ".gz", index=bed + ".gz.csi")
        queries = [(int(a), int(a) + int(w)) for a, w in zip(rng.integers(0, 75_000, 40), rng.integers(0, 12_000, 40))]
        queries += [(0, 0), (0, 1), (int(starts[0]), int(starts[0])), (int(starts[-1]), int(starts[-1]) + 1)]
        for qs, qe in queries:
            ref = [tuple(r) for r in tbx.fetch("chrZ", qs, qe, parser=pysam.asTuple())]
            got = [tuple(f[:-1] + [f[-1].rstrip("\n")]) for f in idx.fetch("chrZ", qs, qe, parser=pysam.asTuple())]
            assert ref == got, (qs, qe)
        with pytest.raises(ValueError):
            idx.fetch("chrZ", -1, 10)
        with pytest.raises(ValueError):
            idx.fetch("chrOther", 0, 10)


def test_subpeak_index_rejects_ties_and_unsorted():
    mk = lambda pairs: ["chrZ\t%d\t%d\tp\t1.0\t0\t+\t0\t1\t%d\t1\n" % (a, b, a) for a, b in pairs]
    assert ce.SubpeakIndex("chrZ", mk([(1, 5), (6, 9)])).is_supported()
    assert not ce.SubpeakIndex("chrZ", mk([(1, 5), (1, 9)])).is_supported()   # tie: tabix order undefined
    assert not ce.SubpeakIndex("chrZ", mk([(6, 9), (1, 5)])).is_supported()   # unsorted
    assert not ce.SubpeakIndex("chrZ", mk([(1, 1)])).is_supported()           # zero length


# ------------------------------------------------------------------ array-backed chunking == line parsing
def _old_segment_chunks(lines, target):
    """`_segment_chunks` before it used the index's arrays (perf/subpeak-mem 49b9dcb)."""
    if target <= 0 or len(lines) <= target:
        return [(0, len(lines))]
    chunks, lo, max_end = [], 0, None
    for i, ln in enumerate(lines):
        f = ln.split("\t", 3)
        s, e = int(f[1]), int(f[2])
        if max_end is not None and s - max_end >= ce._SEGMENT_GAP and i - lo >= target:
            chunks.append((lo, i))
            lo = i
        max_end = e if max_end is None or e > max_end else max_end
    chunks.append((lo, len(lines)))
    return chunks


def _old_context_for(lines, lo, hi):
    """`_context_for` before it used the index's arrays: a linear scan with the same overlap rule."""
    st = [int(l.split("\t")[1]) for l in lines]
    en = [int(l.split("\t")[2]) for l in lines]
    q0 = max(min(st[lo:hi]) - ce._FLANK_MAX_HALF - 2, 0)
    q1 = max(en[:hi]) + ce._FLANK_MAX_HALF + 2
    return [lines[i] for i in range(len(lines)) if st[i] < q1 and en[i] > q0]


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.integers(1, 80), st.integers(0, 12))
def test_array_chunking_matches_line_parsing(seed, n, target):
    rng = np.random.default_rng(seed)
    gaps = rng.choice([1, 50, 5_000, ce._SEGMENT_GAP - 1, ce._SEGMENT_GAP, ce._SEGMENT_GAP + 1, 40_000], n)
    starts = np.cumsum(gaps)
    lens = rng.integers(1, rng.choice([5, 300, 20_000]), n)  # long records can bridge later gaps
    lines = ["chrZ\t%d\t%d\tp%d\t1.0\t0\t+\t0\t1\t%d\t1\n" % (s, s + l, i, s)
             for i, (s, l) in enumerate(zip(starts, lens))]
    bounds = (starts.astype(np.int64), (starts + lens).astype(np.int64))
    for idx in (ce.SubpeakIndex("chrZ", lines), ce.SubpeakIndex("chrZ", lines, *bounds)):
        assert idx.is_supported()
        chunks = ce._segment_chunks(lines, target, idx)
        assert chunks == _old_segment_chunks(lines, target) == ce._segment_chunks(lines, target)
        for lo, hi in chunks:
            ctx, (cs, cn) = ce._context_for(idx, lines, lo, hi)
            assert list(ctx) == _old_context_for(lines, lo, hi)
            assert cs.tolist() == [int(l.split("\t")[1]) for l in ctx]
            assert cn.tolist() == [int(l.split("\t")[2]) for l in ctx]
        for qs, qe in zip(rng.integers(0, int(starts[-1]) + 30_000, 20), rng.integers(0, 30_000, 20)):
            qs, qe = int(qs), int(qs + qe)
            assert idx.overlapping(qs, qe).tolist() == list(idx.indices(qs, qe))


# ------------------------------------------------------------------ forced chunking under a pool
def _clustered_cov(rng, n_clusters=7, dtype=np.int16):
    """Clusters of signal separated by gaps that are sometimes >= _SEGMENT_GAP and sometimes not."""
    parts = []
    for k in range(n_clusters):
        parts.append(synth_coverage(rng, length=int(rng.integers(4_000, 12_000)), n_peaks=int(rng.integers(3, 15)),
                                    dtype=np.int64, edge_peaks=False))
        gap = int(rng.choice([2_000, 9_000, 14_999, 15_000, 16_000, 30_000]))
        parts.append(np.zeros(gap, dtype=np.int64))
    return np.concatenate(parts).astype(dtype)


def _covs(seed):
    rng = np.random.default_rng(seed)
    covs = {"chr%d" % (k + 1): _clustered_cov(rng, n_clusters=int(rng.integers(3, 9))) for k in range(3)}
    covs["chrE"] = np.zeros(4_000, dtype=np.int8)
    return covs


def _write(d, covs):
    files = {}
    for c, v in covs.items():
        files[c] = os.path.join(d, "%s.npy" % c)
        np.save(files[c], v)
    return files


def _spd(mod, d, files, strand):
    lab = "pl" if strand == "+" else "mn"
    return {c: mod.unified_element_definition([files], c, strand, os.path.join(d, "s_%s.bed" % lab)) for c in files}


def _run_pss(mod, d, covs, strand, kw):
    files = _write(d, covs)
    spd = _spd(mod, d, files, strand)
    lab = "pl" if strand == "+" else "mn"
    return file_bytes(mod.peaks_single_strand(files, os.path.join(d, "s_1_%s.bed" % lab), spd, strand, 0.1, **kw))


@pytest.fixture
def tiny_chunks(monkeypatch):
    monkeypatch.setattr(ce, "_MIN_P1_BLOCK", 3)
    monkeypatch.setattr(ce, "_MIN_P2_CHUNK", 2)


@SLOW
@given(seed=st.integers(0, 2 ** 31 - 1), strand=st.sampled_from(["+", "-"]), threads=st.sampled_from([2, 5]),
       sens=st.booleans(), iqr=st.sampled_from(["bgIQR", "pkIQR"]))
def test_forced_chunking_pool_equals_ref(seed, strand, threads, sens, iqr):
    _globals(iqr)
    old = ce._MIN_P1_BLOCK, ce._MIN_P2_CHUNK
    ce._MIN_P1_BLOCK, ce._MIN_P2_CHUNK = 3, 2
    try:
        covs = _covs(seed)
        kw = dict(sensitive=sens, thread_n=threads)
        with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
            same(_run_pss(ce, da, covs, strand, kw), _run_pss(rce, db, covs, strand, dict(kw, thread_n=1)))
    finally:
        ce._MIN_P1_BLOCK, ce._MIN_P2_CHUNK = old


def test_forced_chunking_actually_splits(tiny_chunks):
    """Guard against the chunked path silently degenerating to one chunk per chromosome."""
    _globals()
    covs = _covs(7)
    with tempfile.TemporaryDirectory() as d:
        files = _write(d, covs)
        spd = _spd(ce, d, files, "+")
        n_multi = 0
        for c in files:
            if spd[c] is None:
                continue
            ctx = ce._subpeak_phase1(files[c], os.path.join(d, "s_1_pl.bed"), "+", c, spd[c])
            chunks = ce._segment_chunks(ctx["lines"], 2)
            n_multi += len(chunks) > 1
            # chunks tile the lines and are only cut at gaps >= _SEGMENT_GAP
            assert chunks[0][0] == 0 and chunks[-1][1] == len(ctx["lines"])
            for (a, b), (b2, _) in zip(chunks, chunks[1:]):
                assert b == b2
                max_end = max(int(l.split("\t")[2]) for l in ctx["lines"][:b])
                assert int(ctx["lines"][b].split("\t")[1]) - max_end >= ce._SEGMENT_GAP
        assert n_multi >= 1


def test_multi_job_scheduler_equals_ref(tiny_chunks):
    """Several samples x strands evaluated in one pool == per-chromosome reference evaluation."""
    _globals()
    with tempfile.TemporaryDirectory() as d:
        jobs, refs = [], []
        for s, seed in enumerate((3, 11)):
            covs = _covs(seed)
            sd = os.path.join(d, "s%d" % s)
            os.makedirs(sd)
            files = _write(sd, covs)
            for strand in ("+", "-"):
                spd = _spd(ce, sd, files, strand)
                out = os.path.join(sd, "o_%s.bed" % ("pl" if strand == "+" else "mn"))
                jobs.append((files, out, spd, strand))
        with ce._make_pool(4) as pool:
            got = ce.evaluate_strand_jobs(jobs, 0.1, pool=pool, thread_n=4)
        for (files, out, spd, strand), dfs in zip(jobs, got):
            exp = []
            rd = out.replace(".bed", "_ref.bed")
            for c, f in files.items():
                if spd[c] is None:
                    continue
                exp.append(rce.check_window_chromosome(f, rd, strand, c, spd[c], 0.1))
            assert len(exp) == len(dfs)
            for a, b in zip(exp, dfs):
                assert a.equals(b) and list(a.dtypes) == list(b.dtypes)


def test_tabix_fallback_path_equals_ref(monkeypatch, tiny_chunks):
    """If the in-memory index is rejected, the engine must fall back to tabix without chunking."""
    _globals()
    monkeypatch.setattr(ce.SubpeakIndex, "is_supported", lambda self: False)
    covs = _covs(5)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        same(_run_pss(ce, da, covs, "+", dict(thread_n=3)), _run_pss(rce, db, covs, "+", dict(thread_n=1)))


# ------------------------------------------------------------------ subpeak beds kept in memory
def _subpeak_files(d, prefix):
    return sorted(f for f in os.listdir(d) if f.startswith(prefix + "_subpeaks_"))


@pytest.mark.parametrize("threads", [1, 3])
def test_supported_subpeak_beds_are_not_written(tiny_chunks, threads):
    """Phase 1 writes, compresses and indexes a subpeak bed only for the tabix fallback."""
    _globals()
    covs = _covs(7)
    with tempfile.TemporaryDirectory() as d:
        files = _write(d, covs)
        spd = _spd(ce, d, files, "+")
        out = os.path.join(d, "o_pl.bed")
        if threads == 1:
            ce.evaluate_strand_jobs([(files, out, spd, "+")], 0.1, pool=None, thread_n=1)
        else:
            with ce._make_pool(threads) as pool:
                ce.evaluate_strand_jobs([(files, out, spd, "+")], 0.1, pool=pool, thread_n=threads)
        assert _subpeak_files(d, "o_pl") == []


@pytest.mark.parametrize("threads", [1, 3])
def test_unsupported_subpeak_beds_are_still_indexed(monkeypatch, tiny_chunks, threads):
    _globals()
    monkeypatch.setattr(ce.SubpeakIndex, "is_supported", lambda self: False)
    covs = _covs(5)
    with tempfile.TemporaryDirectory() as d:
        files = _write(d, covs)
        spd = _spd(ce, d, files, "+")
        out = os.path.join(d, "o_pl.bed")
        if threads == 1:
            ce.evaluate_strand_jobs([(files, out, spd, "+")], 0.1, pool=None, thread_n=1)
        else:
            with ce._make_pool(threads) as pool:
                ce.evaluate_strand_jobs([(files, out, spd, "+")], 0.1, pool=pool, thread_n=threads)
        n = sum(v is not None for v in spd.values())
        got = _subpeak_files(d, "o_pl")
        assert len(got) == 2 * n and all(f.endswith((".bed.gz", ".bed.gz.csi")) for f in got)


@pytest.mark.parametrize("seed", [3, 5, 11])
def test_plain_element_definition_equals_compressed(seed):
    """compress=False leaves the same lines uncompressed, and phase 1 reads either form identically."""
    _globals()
    covs = _covs(seed)
    with tempfile.TemporaryDirectory() as d:
        files = _write(d, covs)
        for c in files:
            gz = ce.unified_element_definition([files], c, "+", os.path.join(d, "g_pl.bed"))
            plain = ce.unified_element_definition([files], c, "+", os.path.join(d, "p_pl.bed"), compress=False)
            assert (gz is None) == (plain is None)
            if gz is None:
                continue
            assert plain.endswith(".bed") and not os.path.exists(plain + ".gz")
            assert ce._read_subpeak_input(plain) == ce._read_subpeak_input(gz)
            with open(plain, "rb") as fh:
                assert fh.read() == file_bytes(gz)
            a = ce._subpeak_phase1(files[c], os.path.join(d, "a_pl.bed"), "+", c, gz)
            b = ce._subpeak_phase1(files[c], os.path.join(d, "b_pl.bed"), "+", c, plain)
            assert a["lines"] == b["lines"]
            assert all(a[k] == b[k] for k in ("bkg_mu_threshold", "empirical_true_peak_threshold", "enable_eler"))
