"""Differential tests for the in-memory coverage store: tracks built by each producer hold exactly what the
reference's .npy dumps hold (values, dtypes, chromosome order, read totals), and no dumps are written."""
import os
import tempfile

import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.io_engine as io
import pints_ref.io_engine as rio
from helpers import make_bam
from test_differential import same, N
from test_differential_parallel import mixed_reads

SLOW = settings(max_examples=max(3, N // 2), deadline=None, suppress_health_check=list(HealthCheck))
LIBS = ["R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3", "PROcap", "PROseq", "CoPRO", "mNETseq"]


def _assert_tracks_equal_dumps(tracks, dumps):
    assert list(tracks) == list(dumps)  # same chromosomes, same (header) order
    for c in dumps:
        ref = np.load(dumps[c])
        t = tracks[c]
        assert t.length == ref.shape[0] and t.dtype == ref.dtype, (c, t, ref.dtype)
        assert np.array_equal(t.dense(), ref), c


def _npys(d):
    return [f for f in os.listdir(d) if f.endswith(".npy")]


@SLOW
@given(mixed_reads(), st.sampled_from(LIBS), st.booleans(), st.sampled_from([1, 3]),
       st.sampled_from([(), ("ctg",)]), st.booleans())
def test_bam_tracks_equal_reference_dumps(rr, lib, rc, threads, filters, fast):
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        pl, mn, total, cache = io._read_signal(bam, lib, "", os.path.join(d, "a"), "p", reverse_complement=rc,
                                               thread_n=threads, filters=filters, fast_bam=fast, as_tracks=True)
        b = rio.get_read_signal(bam, lib, "", os.path.join(d, "b"), "p", reverse_complement=rc, filters=filters)
        same(total, b[2])
        _assert_tracks_equal_dumps(pl, b[0])
        _assert_tracks_equal_dumps(mn, b[1])
        assert cache == [] and _npys(os.path.join(d, "a")) == []


@SLOW
@given(mixed_reads(), st.sampled_from(["R_5", "R1_3", "PROseq"]), st.sampled_from([2, 8]),
       st.sampled_from([300, 4000]), st.integers(0, 2 ** 31 - 1))
def test_bam_region_tracks_equal_reference_dumps(rr, lib, threads, span, seed):
    """Contigs split into regions whose 5' ends spill into neighbouring regions."""
    contigs, reads = rr
    rng = np.random.default_rng(seed)
    for c, L in contigs[:1]:
        for b in range(span, L, span):
            for pos in (b - 25, b - 1, b, b - 30):
                reads.append(dict(contig=c, pos=pos, cigar="30M", reverse=bool(rng.random() < 0.5), mapq=60))
    old = io.BAM_SPLIT_MIN_SPAN
    io.BAM_SPLIT_MIN_SPAN = span
    try:
        with tempfile.TemporaryDirectory() as d:
            bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
            os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
            pl, mn, total, _ = io._read_signal(bam, lib, "", os.path.join(d, "a"), "p", thread_n=threads,
                                               as_tracks=True)
            b = rio.get_read_signal(bam, lib, "", os.path.join(d, "b"), "p")
            same(total, b[2])
            _assert_tracks_equal_dumps(pl, b[0])
            _assert_tracks_equal_dumps(mn, b[1])
            assert _npys(os.path.join(d, "a")) == []
    finally:
        io.BAM_SPLIT_MIN_SPAN = old


@SLOW
@given(mixed_reads(), st.sampled_from(["PROcap", "R2_3"]))
def test_bam_tracks_from_the_npy_cache(rr, lib):
    """Dumps left in the output directory are reused (as tracks) and reported for deletion."""
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        out = os.path.join(d, "a"); os.makedirs(out)
        dumps = io.get_read_signal(bam, lib, "", out, "p")
        before = sorted(_npys(out))
        pl, mn, total, cache = io._read_signal(bam, lib, "", out, "p", as_tracks=True)
        ref = rio.get_read_signal(bam, lib, "", out, "p")  # the reference also reads the cache
        same(total, ref[2])
        _assert_tracks_equal_dumps(pl, dumps[0])
        _assert_tracks_equal_dumps(mn, dumps[1])
        assert sorted(cache) == sorted(list(dumps[0].values()) + list(dumps[1].values()))
        assert sorted(_npys(out)) == before


# ------------------------------------------------------------------ bigWig
def _runs_bigwig(path, chroms, rng, vmax):
    """bedGraph-style bigWig with runs of several lengths, negative, fractional (some cast to 0) and huge values.
    (No inf: pyBigWig's header() crashes on it, in the reference too.)"""
    import pyBigWig
    bw = pyBigWig.open(path, "w")
    bw.addHeader(list(chroms))
    for name, L in chroms:
        pos, starts, ends, vals = 0, [], [], []
        while True:
            pos += int(rng.integers(0, 60))
            ln = int(rng.choice([1, 1, 1, 2, 7, 150]))
            if pos + ln > L:
                break
            v = float(rng.choice([rng.integers(1, vmax), -rng.integers(1, vmax), rng.uniform(-3, 3), 0.4, 3e9]))
            starts.append(pos); ends.append(pos + ln); vals.append(v)
            pos += ln
        if starts:
            bw.addEntries([name] * len(starts), starts, ends=ends, values=vals)
    bw.close()
    return path


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([3, 100, 30000, 3_000_000]), st.sampled_from([1, 3]),
       st.sampled_from(["", "chr"]), st.sampled_from([64, 1 << 22]), st.booleans(), st.booleans())
def test_bigwig_tracks_equal_reference_dumps(seed, vmax, threads, prefix_filter, iv_block, mismatch, dense):
    rng = np.random.default_rng(seed)
    chroms = [("chr1", 9000), ("chr2", 2000), ("ctgX", 777), ("chrM", 16)]
    mn_chroms = chroms + [("chrU", 3001)] if mismatch else chroms  # only in one strand: dropped, counts adjusted
    old_iv = io._BW_IV_BLOCK
    io._BW_IV_BLOCK = iv_block
    old_runs = io._bw_track_runs
    if dense:  # intervals unusable: the dense values() fallback
        io._bw_track_runs = lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory() as d:
            p = _runs_bigwig(os.path.join(d, "p.bw"), chroms, rng, vmax)
            m = _runs_bigwig(os.path.join(d, "m.bw"), mn_chroms, rng, vmax)
            os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
            with np.errstate(invalid="ignore"):
                pl, mn, total, cache = io._coverage_bw(p, m, prefix_filter, os.path.join(d, "a"), "q",
                                                       threads=threads, as_tracks=True)
                b = rio.get_coverage_bw(p, m, prefix_filter, os.path.join(d, "b"), "q")
            same(total, b[2])
            _assert_tracks_equal_dumps(pl, b[0])
            _assert_tracks_equal_dumps(mn, b[1])
            assert cache == [] and _npys(os.path.join(d, "a")) == []
            for t in list(pl.values()) + list(mn.values()):
                assert (t.values != 0).all()
    finally:
        io._BW_IV_BLOCK = old_iv
        io._bw_track_runs = old_runs


def test_bigwig_tracks_without_numpy_and_from_cache(monkeypatch):
    import pyBigWig
    rng = np.random.default_rng(5)
    chroms = [("chr1", 3000), ("chr2", 500)]
    with tempfile.TemporaryDirectory() as d:
        p = _runs_bigwig(os.path.join(d, "p.bw"), chroms, rng, 300)
        m = _runs_bigwig(os.path.join(d, "m.bw"), chroms, rng, 300)
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        os.makedirs(a); os.makedirs(b)
        with np.errstate(invalid="ignore"):
            ref = rio.get_coverage_bw(p, m, "", b, "q")
            for numpy_flag in (0, 1):
                monkeypatch.setattr(pyBigWig, "numpy", numpy_flag)
                pl, mn, total, cache = io._coverage_bw(p, m, "", a, "q", as_tracks=True)
                same(total, ref[2])
                _assert_tracks_equal_dumps(pl, ref[0]); _assert_tracks_equal_dumps(mn, ref[1])
                assert cache == [] and _npys(a) == []
            dumps = io.get_coverage_bw(p, m, "", a, "q")  # leaves dumps: the cache
            for numpy_flag, threads in ((0, 1), (1, 1), (1, 3)):
                monkeypatch.setattr(pyBigWig, "numpy", numpy_flag)
                pl, mn, total, cache = io._coverage_bw(p, m, "", a, "q", threads=threads, as_tracks=True)
                same(total, ref[2])
                _assert_tracks_equal_dumps(pl, ref[0]); _assert_tracks_equal_dumps(mn, ref[1])
                assert sorted(cache) == sorted(list(dumps[0].values()) + list(dumps[1].values()))


# ------------------------------------------------------------------ replicate merge
from pints import coverage_store as cs


def _runs_of(arr):
    """Runs-form track of a dense array (maximal runs of equal non-zero value)."""
    nz = arr != 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], arr.astype(np.int64), [0]))) != 0)
    st_, en, va = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if nz[a]:
            st_.append(a); en.append(b); va.append(arr[a])
    return cs.CoverageTrack(arr.shape[0], arr.dtype, st_, va, en)


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([2, 3]), st.booleans(), st.sampled_from([3, 300, 40_000]))
def test_merge_replicate_tracks_equals_file_merge(seed, n_reps, runs, vmax):
    rng = np.random.default_rng(seed)
    chroms = {"chr1": 5000, "chr2": 3000, "chrZ": 40, "chrE": 700, "chrR": 900}
    with tempfile.TemporaryDirectory() as d:
        def sample(tag, rep):
            paths, tracks = {}, {}
            for c, L in chroms.items():
                if c == "chrR" and rep == 1:
                    continue  # not shared by every replicate: left out of the merge
                fn = os.path.join(d, "%s_%s.npy" % (tag, c))
                if c == "chrE":  # registered empty track (BAM contig without reads)
                    paths[c] = io.save_zero_track(fn, L)
                    tracks[c] = cs.zero_track(L)
                    continue
                dt = rng.choice([np.int8, np.int16, np.int32]) if vmax < 100 else np.int32
                v = np.where(rng.random(L) < 0.08, rng.integers(1, vmax, L), 0)
                if c == "chrZ" and rep == 0:
                    v[:] = 0  # empty, but not registered as a zero track (bigWig dumps)
                if runs:
                    v = np.repeat(v[::5], 5)[:L]
                v = v.astype(dt)
                np.save(fn, v)
                paths[c] = fn
                tracks[c] = _runs_of(v) if runs else cs.track_from_dense(v)
            return paths, tracks
        pls = [sample("p%d" % i, i) for i in range(n_reps)]
        mns = [sample("m%d" % i, i) for i in range(n_reps)]
        for sub in ("a", "b"):
            os.makedirs(os.path.join(d, sub))
        got = io.merge_replicate_tracks([t for _, t in pls], [t for _, t in mns])
        a = io.merge_replicates_bw([p for p, _ in pls], [p for p, _ in mns], os.path.join(d, "a"), "m")
        b = rio.merge_replicates_bw([p for p, _ in pls], [p for p, _ in mns], os.path.join(d, "b"), "m")
        for exp in (a, b):
            same(got[2], exp[2])
            same(got[3], exp[3])
        same(got[2], a[2])
        for k in (0, 1):
            _assert_tracks_equal_dumps(got[k], a[k])
            assert [io.is_zero_track(got[k][c]) for c in got[k]] == [io.is_zero_track(a[k][c]) for c in a[k]]
