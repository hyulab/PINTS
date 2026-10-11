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
