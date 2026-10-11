"""Differential tests for Gate-1 code paths that have no direct counterpart in the reference:
vectorised window sums / region merge, contig-parallel BAM parsing (incl. fallback), and the
all-zero ("sparse") coverage-track shortcuts. Everything is compared bitwise against pints_ref."""
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from pybedtools import BedTool

import pints.calling_engine as ce
import pints.io_engine as io
import pints_ref.calling_engine as rce
import pints_ref.io_engine as rio
from helpers import make_bam, synth_coverage
from test_differential import same, file_bytes, N

FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))
SLOW = settings(max_examples=max(3, N // 2), deadline=None, suppress_health_check=list(HealthCheck))


# ------------------------------------------------------------------ window sums / regions
@FAST
@given(st.integers(0, 2 ** 31 - 1), st.integers(1, 5000), st.sampled_from([np.int8, np.int16, np.int32, np.int64]),
       st.sampled_from([(100, 100), (4, 2), (7, 7), (10, 3), (5, 9)]))
def test_window_sums_equal_sliding_window(seed, length, dt, ws):
    w, s = ws
    rng = np.random.default_rng(seed)
    cov = np.where(rng.random(length) < 0.1, rng.integers(-3, 50, length), 0).astype(dt)
    ref = list(rce.sliding_window(cov, window_size=w, step_size=s))
    sums, starts = ce.window_sums(cov, window_size=w, step_size=s)
    assert len(ref) == len(sums)
    for (v, (a, b)), v2, a2 in zip(ref, sums, starts):
        assert type(v) == type(v2) and v == v2 and a == a2 and b == a2 + w


def _ref_regions(cov, chrom="chrZ"):
    bins = [(chrom, c[0], c[1], ".", int(v), "+") for v, c in rce.sliding_window(cov) if v > 0]
    if not bins:
        return []
    txt = "\n".join("\t".join(map(str, b)) for b in bins) + "\n"
    return [(r.start, r.end) for r in BedTool(txt, from_string=True).merge(c=(4, 5, 6),
                                                                           o=("distinct", "sum", "distinct"))]


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.integers(1, 30000), st.floats(0.0005, 0.2))
def test_nonzero_window_regions_equal_bedtools_merge(seed, length, dens):
    rng = np.random.default_rng(seed)
    cov = np.where(rng.random(length) < dens, rng.integers(1, 20, length), 0).astype(np.int32)
    assert ce.nonzero_window_regions(cov) == _ref_regions(cov)


# ------------------------------------------------------------------ BAM parsing
@st.composite
def mixed_reads(draw):
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    contigs = [("chr1", 20_000), ("chrEmpty", 7_000), ("chr2", 5_000), ("ctgA", 3_000), ("chrM", 500)]
    used = [c for c in contigs if c[0] != "chrEmpty"]
    reads = []
    for i in range(draw(st.integers(1, 800))):
        c, L = used[int(rng.integers(0, len(used)))]
        ln = int(rng.integers(10, 60))
        pos = int(rng.integers(0, L - ln - 5))
        cig = rng.choice(["%dM" % ln, "3S%dM" % (ln - 3), "%dM2D%dM" % (ln // 2, ln - ln // 2), "%dM4S" % (ln - 4)])
        kind = rng.random()
        paired = kind < 0.7
        r1 = bool(rng.random() < 0.5)
        reads.append(dict(contig=c, pos=pos, cigar=str(cig), reverse=bool(rng.random() < 0.5),
                          mapq=int(rng.choice([0, 10, 29, 30, 31, 60, 255])), paired=paired,
                          read1=paired and r1, read2=paired and not r1,
                          secondary=bool(rng.random() < 0.05), supplementary=bool(rng.random() < 0.03)))
    for i in range(int(rng.integers(0, 5))):  # unplaced unmapped reads at the end of the file
        reads.append(dict(contig=None, unmapped=True, mapq=0))
    return contigs, reads


def _load_all(res):
    return [{c: np.load(p) for c, p in d.items()} for d in res[:2]], res[2]


@SLOW
@given(mixed_reads(), st.sampled_from(["R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3", "PROcap", "PROseq", "CoPRO",
                                       "mNETseq"]),
       st.booleans(), st.sampled_from([1, 3]), st.sampled_from([(), ("ctg",)]))
def test_get_read_signal_contig_parallel(rr, lib, rc, threads, filters):
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        a = io.get_read_signal(bam, lib, "", os.path.join(d, "a"), "p", reverse_complement=rc, thread_n=threads,
                               filters=filters)
        b = rio.get_read_signal(bam, lib, "", os.path.join(d, "b"), "p", reverse_complement=rc, filters=filters)
        assert list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])  # same keys, same (header) order
        la, ra = _load_all(a)
        lb, rb = _load_all(b)
        same(ra, rb)
        same(la, lb)
        # zero-track registry must only contain genuinely empty tracks
        for d_ in a[:2]:
            for c, p in d_.items():
                if io.is_zero_track(p):
                    assert not np.load(p).any()
        assert io.is_zero_track(a[0]["chrEmpty"]) and io.is_zero_track(a[1]["chrEmpty"])


def test_get_read_signal_without_index_falls_back():
    rng = np.random.default_rng(7)
    contigs = [("chr1", 5000), ("chr2", 3000)]
    reads = [dict(contig="chr%d" % (1 + i % 2), pos=int(rng.integers(0, 2900)), cigar="30M",
                  reverse=bool(i % 3 == 0), mapq=60) for i in range(300)]
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        os.remove(bam + ".bai")
        assert io.parse_bam_by_contig(bam, contigs, 30, "R", "5", False, d, "z") is None
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        a = io.get_read_signal(bam, "PROcap", "", os.path.join(d, "a"), "p", thread_n=2)
        b = rio.get_read_signal(bam, "PROcap", "", os.path.join(d, "b"), "p")
        same(_load_all(a), _load_all(b))


# ------------------------------------------------------------------ zero-track shortcuts
def test_zero_track_file_is_valid_npy():
    with tempfile.TemporaryDirectory() as d:
        p = io.save_zero_track(os.path.join(d, "z"), 123_457, np.int8)
        np.save(os.path.join(d, "r"), np.zeros(123_457, dtype=np.int8))
        assert open(p, "rb").read() == open(os.path.join(d, "r.npy"), "rb").read()
        assert io.is_zero_track(p) and not io.is_zero_track(os.path.join(d, "r.npy"))


def test_unified_element_definition_skips_zero_tracks():
    with tempfile.TemporaryDirectory() as d:
        z1 = io.save_zero_track(os.path.join(d, "z1"), 50_000)
        z2 = io.save_zero_track(os.path.join(d, "z2"), 50_000)
        cov = {"chrZ": z1}
        a = ce.unified_element_definition([cov, {"chrZ": z2}], "chrZ", "+", os.path.join(d, "a.bed"))
        b = rce.unified_element_definition([cov, {"chrZ": z2}], "chrZ", "+", os.path.join(d, "b.bed"))
        assert a is None and b is None
        # mixed zero / non-zero replicates must still be merged
        rng = np.random.default_rng(3)
        np.save(os.path.join(d, "nz.npy"), synth_coverage(rng, length=50_000, n_peaks=10, dtype=np.int16))
        reps = [{"chrZ": z1}, {"chrZ": os.path.join(d, "nz.npy")}]
        a = ce.unified_element_definition(reps, "chrZ", "-", os.path.join(d, "c.bed"))
        b = rce.unified_element_definition(reps, "chrZ", "-", os.path.join(d, "e.bed"))
        assert file_bytes(a) == file_bytes(b)


@pytest.mark.parametrize("sf", [0.5, 1.0, 3.7])
def test_normalize_using_input_zero_assay(sf):
    import logging
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as d:
        z = io.save_zero_track(os.path.join(d, "assay_pl_chrA"), 10_000)
        nz = os.path.join(d, "assay_pl_chrB.npy")
        np.save(nz, rng.integers(0, 5, 10_000).astype(np.int8))
        inp = {}
        for c in ("chrA", "chrB"):
            inp[c] = os.path.join(d, "in_%s.npy" % c)
            np.save(inp[c], rng.integers(0, 3, 10_000).astype(np.int8))
        assay = {"chrA": z, "chrB": nz}
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        log = logging.getLogger("t")
        a = io.normalize_using_input(assay, assay, inp, inp, sf, os.path.join(d, "a"), "n", log)
        b = rio.normalize_using_input(assay, assay, inp, inp, sf, os.path.join(d, "b"), "n", log)
        for x, y in zip(a, b):
            assert list(x) == list(y)
            for c in x:
                same(np.load(x[c]), np.load(y[c]))


def test_merge_replicates_zero_tracks():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as d:
        pls, mns = [], []
        for r in range(2):
            pl, mn = {}, {}
            for c in ("chrA", "chrB"):
                if c == "chrA":
                    pl[c] = io.save_zero_track(os.path.join(d, "r%d_pl_%s" % (r, c)), 8_000)
                    mn[c] = io.save_zero_track(os.path.join(d, "r%d_mn_%s" % (r, c)), 8_000)
                else:
                    pl[c] = os.path.join(d, "r%d_pl_%s.npy" % (r, c))
                    mn[c] = os.path.join(d, "r%d_mn_%s.npy" % (r, c))
                    np.save(pl[c], rng.integers(0, 4, 8_000).astype(np.int8))
                    np.save(mn[c], rng.integers(0, 4, 8_000).astype(np.int8))
            pls.append(pl); mns.append(mn)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        a = io.merge_replicates_bw(pls, mns, os.path.join(d, "a"), "m")
        b = rio.merge_replicates_bw(pls, mns, os.path.join(d, "b"), "m")
        assert a[2] == b[2] and a[3] == b[3]
        for x, y in zip(a[:2], b[:2]):
            assert set(x) == set(y)
            for c in x:
                same(np.load(x[c]), np.load(y[c]))


# ------------------------------------------------------------------ parallel block-wise bigWig loader
from helpers import make_bigwig


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([3, 100, 30000, 3_000_000]), st.sampled_from([7, 333, 4096]),
       st.sampled_from(["", "chr"]), st.sampled_from([2, 5]), st.booleans())
def test_get_coverage_bw_parallel(seed, vmax, block, prefix_filter, threads, fractional):
    rng = np.random.default_rng(seed)
    chroms = [("chr1", 9000), ("chr2", 2000), ("ctgX", 777), ("chrM", 16)]
    if rng.random() < 0.5:
        chroms.append(("chrU", 12001))

    def vals(L, sign):
        v = np.where(rng.random(L) < 0.05, rng.integers(1, vmax, L), 0).astype(float)
        if fractional:
            v = v * rng.uniform(0.1, 1.3, L)
        return sign * v

    pv = {c: vals(L, 1) for c, L in chroms}
    mv = {c: vals(L, -1) for c, L in chroms}
    old = io._BW_BLOCK
    io._BW_BLOCK = block
    try:
        with tempfile.TemporaryDirectory() as d:
            p = make_bigwig(os.path.join(d, "p.bw"), chroms, pv)
            m = make_bigwig(os.path.join(d, "m.bw"), chroms, mv)
            os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
            a = io.get_coverage_bw(p, m, prefix_filter, os.path.join(d, "a"), "q", threads=threads)
            b = rio.get_coverage_bw(p, m, prefix_filter, os.path.join(d, "b"), "q")
            assert type(a[2]) == type(b[2]) and a[2] == b[2]
            for k in (0, 1):
                assert list(a[k]) == list(b[k])
                for c in a[k]:
                    assert os.path.basename(a[k][c]) == os.path.basename(b[k][c])
                    assert file_bytes(a[k][c]) == file_bytes(b[k][c])
            # second call re-uses the existing dumps (exists() branch)
            a2 = io.get_coverage_bw(p, m, prefix_filter, os.path.join(d, "a"), "q", threads=threads)
            assert type(a2[2]) == type(b[2]) and a2[2] == b[2]
    finally:
        io._BW_BLOCK = old


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from(["bedgraph", "fixed", "var"]),
       st.sampled_from([np.int8, np.int16, np.int32, np.int64]))
def test_bw_sparse_track_equals_dense(seed, mode, dtype):
    """Interval-based track == dense values() path for bedGraph / fixedStep(span) / variableStep(span) bigWigs."""
    import pyBigWig
    rng = np.random.default_rng(seed)
    L = int(rng.integers(50, 5000))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.bw")
        bw = pyBigWig.open(path, "w")
        bw.addHeader([("chrA", L), ("chrB", 10)])
        pos, n = 0, 0
        starts, ends, vals = [], [], []
        while pos < L - 1:
            pos += int(rng.integers(0, 40))
            ln = int(rng.integers(1, 9))
            if pos + ln > L:
                break
            v = float(rng.choice([rng.uniform(-120, 120), rng.integers(-100, 100), 0.5, -0.5, np.inf, -np.inf]))
            starts.append(pos); ends.append(pos + ln); vals.append(v)
            pos += ln
        if starts:
            if mode == "bedgraph":
                bw.addEntries(["chrA"] * len(starts), starts, ends=ends, values=vals)
            elif mode == "fixed":
                span = 3
                k = L // span
                fv = [float(x) for x in rng.choice([0.0, 1.0, -2.5, 7.0], size=k)]
                bw.addEntries("chrA", 0, values=fv, span=span, step=span)
            else:
                span = 2
                p = sorted(set(int(x) // 2 * 2 for x in rng.choice(L - 2, size=min(40, L // 4), replace=False)))
                bw.addEntries("chrA", p, values=[float(x) for x in rng.uniform(-9, 9, len(p))], span=span)
        bw.close()
        bw = pyBigWig.open(path)
        for chrom, csize in bw.chroms().items():
            sp = io._bw_sparse_track(bw, chrom, csize, dtype)
            with np.errstate(invalid="ignore"):
                dense = np.nan_to_num(bw.values(chrom, 0, csize, numpy=True))
                dense[dense < 0] *= -1
                dense = dense.astype(dtype, copy=False)
            assert sp is not None and sp.dtype == dense.dtype and np.array_equal(sp, dense)
        bw.close()


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 3]), st.sampled_from([0.37, 1.0, 2.659, 0.0]))
def test_parallel_merge_and_normalize(seed, threads, sf):
    """Pooled merge_replicates_bw / normalize_using_input == reference (files, dict order, read counts)."""
    import logging
    rng = np.random.default_rng(seed)
    chroms = {"chr1": 5000, "chr2": 3000, "chrZ": 40, "chrE": 700}
    with tempfile.TemporaryDirectory() as d:
        def dump(tag, empty=()):
            out = {}
            for c, L in chroms.items():
                v = np.zeros(L, dtype=np.int16) if c in empty else \
                    np.where(rng.random(L) < 0.1, rng.integers(1, 300, L), 0).astype(np.int16)
                fn = os.path.join(d, "%s_%s.npy" % (tag, c))
                np.save(fn, v)
                out[c] = fn
            return out
        pls = [dump("p%d" % i, empty=("chrE",)) for i in range(2)]
        mns = [dump("m%d" % i) for i in range(2)]
        for sub in ("a", "b"):
            os.makedirs(os.path.join(d, sub))
        a = io.merge_replicates_bw(pls, mns, os.path.join(d, "a"), "m", threads=threads)
        b = rio.merge_replicates_bw(pls, mns, os.path.join(d, "b"), "m")
        assert a[2] == b[2] and a[3] == b[3] and type(a[2]) == type(b[2])
        for k in (0, 1):
            assert list(a[k]) == list(b[k])
            for c in a[k]:
                assert file_bytes(a[k][c]) == file_bytes(b[k][c])
        ipl, imn = dump("ip"), dump("im")
        del imn["chrZ"]  # missing control chromosome branch
        lg = logging.getLogger("t")
        x = io.normalize_using_input(pls[0], mns[0], ipl, imn, sf, os.path.join(d, "a"), "n", lg, threads=threads)
        y = rio.normalize_using_input(pls[0], mns[0], ipl, imn, sf, os.path.join(d, "b"), "n", lg)
        for k in (0, 1):
            assert list(x[k]) == list(y[k])
            for c in x[k]:
                assert file_bytes(x[k][c]) == file_bytes(y[k][c])
