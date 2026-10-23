"""Differential tests for the block-wise (memory-bounded) code paths: element definition, bigWig loading and
input normalisation. Each block-wise path is compared bitwise with the whole-array code it replaces (and with
the frozen reference `pints_ref`), using tiny blocks so that block boundaries are exercised densely."""
import logging
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints.io_engine as io
import pints_ref.calling_engine as rce
import pints_ref.io_engine as rio
from helpers import synth_coverage
from test_differential import file_bytes, N

FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))
SLOW = settings(max_examples=max(3, N // 2), deadline=None, suppress_health_check=list(HealthCheck))
DTYPES = [np.int8, np.int16, np.int32, np.int64]


# ------------------------------------------------------------------ npy helpers
@pytest.mark.parametrize("dt", DTYPES + [np.float64])
@pytest.mark.parametrize("length", [1, 99, 100, 101, 12_345])
def test_npy_create_header_matches_np_save(dt, length):
    rng = np.random.default_rng(length)
    v = rng.integers(0, 100, length).astype(dt)
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "r.npy")
        np.save(ref, v)
        new = os.path.join(d, "n.npy")
        lay = io._npy_create(new, dt, length)
        with open(new, "r+b") as fh:
            for a in range(0, length, 37):  # block-wise writes
                b = min(length, a + 37)
                fh.seek(lay[0] + a * lay[1].itemsize)
                fh.write(v[a:b].view(np.uint8))
        assert open(new, "rb").read() == open(ref, "rb").read()
        assert io.npy_layout(ref) == lay
        with open(ref, "rb") as fh:
            for a, b in [(0, length), (0, 0), (length // 3, length), (length - 1, length)]:
                assert np.array_equal(io.npy_read(fh, lay, a, b), v[a:b])


def test_npy_layout_rejects_non_plain_arrays():
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "o.npy"), np.array([1, None], dtype=object))
        np.save(os.path.join(d, "m.npy"), np.zeros((3, 4), dtype=np.int16))
        assert io.npy_layout(os.path.join(d, "o.npy")) is None
        assert io.npy_layout(os.path.join(d, "m.npy")) is None
        assert io.npy_layout(os.path.join(d, "missing.npy")) is None


# ------------------------------------------------------------------ element definition
def _whole_array_ued(reps, chrom, out):
    """The pre-block-wise implementation (block path disabled)."""
    saved = ce._blockwise_element_definition
    ce._blockwise_element_definition = lambda *a, **k: False
    try:
        return ce.unified_element_definition(reps, chrom, "+", out)
    finally:
        ce._blockwise_element_definition = saved


def _ued_case(rng, length, n_reps, dens, dtypes, zero_rep):
    reps = []
    for k in range(n_reps):
        dt = dtypes[k % len(dtypes)]
        if zero_rep and k == 0:
            reps.append(None)
            continue
        if rng.random() < 0.5:
            v = synth_coverage(rng, length=max(length, 500), n_peaks=int(rng.integers(1, 30)), dtype=np.int64)[:length]
        else:  # long runs of non-zero windows (regions straddling many blocks)
            v = np.where(rng.random(length) < dens, rng.integers(1, 40, length), 0)
        hi = np.iinfo(dt).max
        reps.append(np.clip(v, 0, hi).astype(dt))
    return reps


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 50, 99, 100, 101, 250, 999, 1000, 1001, 7_777, 30_000]),
       st.integers(1, 3), st.floats(0.001, 0.9), st.sampled_from([100, 200, 300, 1000, 4000, 10 ** 6]),
       st.lists(st.sampled_from(DTYPES), min_size=1, max_size=3), st.booleans())
def test_blockwise_element_definition_equals_whole_array(seed, length, n_reps, dens, block, dtypes, zero_rep):
    rng = np.random.default_rng(seed)
    reps = _ued_case(rng, length, n_reps, dens, dtypes, zero_rep)
    old = ce.NPY_BLOCK
    ce.NPY_BLOCK = block
    try:
        with tempfile.TemporaryDirectory() as d:
            cov = []
            for k, v in enumerate(reps):
                p = os.path.join(d, "r%d" % k)
                p = io.save_zero_track(p, length) if v is None else (np.save(p, v) or p + ".npy")
                cov.append({"chrQ": p})
            a = ce.unified_element_definition(cov, "chrQ", "+", os.path.join(d, "a.bed"))
            b = _whole_array_ued(cov, "chrQ", os.path.join(d, "b.bed"))
            assert (a is None) == (b is None)
            assert file_bytes(a) == file_bytes(b)
            # the file is created exactly when the whole-array code creates it
            assert os.path.exists(os.path.join(d, "a_subpeaks_chrQ.bed")) == \
                os.path.exists(os.path.join(d, "b_subpeaks_chrQ.bed"))
    finally:
        ce.NPY_BLOCK = old


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1_000, 9_000, 25_001]), st.sampled_from([100, 700, 3000]))
def test_blockwise_element_definition_equals_reference(seed, length, block):
    rng = np.random.default_rng(seed)
    reps = [synth_coverage(rng, length=length, n_peaks=20, dtype=np.int16),
            np.where(rng.random(length) < 0.3, rng.integers(1, 9, length), 0).astype(np.int8)]
    old = ce.NPY_BLOCK
    ce.NPY_BLOCK = block
    try:
        with tempfile.TemporaryDirectory() as d:
            cov = []
            for k, v in enumerate(reps):
                np.save(os.path.join(d, "r%d.npy" % k), v)
                cov.append({"chrQ": os.path.join(d, "r%d.npy" % k)})
            a = ce.unified_element_definition(cov, "chrQ", "+", os.path.join(d, "a.bed"))
            b = rce.unified_element_definition(cov, "chrQ", "+", os.path.join(d, "b.bed"))
            assert file_bytes(a) == file_bytes(b)
    finally:
        ce.NPY_BLOCK = old


def test_blockwise_element_definition_int32_wraparound():
    """Merging into int32 wraps exactly as the whole-array `+=` does."""
    L = 2000
    v = np.zeros(L, dtype=np.int32)
    v[300:340] = 2 ** 31 - 5
    v[1500:1510] = 7
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "a.npy"), v)
        np.save(os.path.join(d, "b.npy"), v)
        cov = [{"c": os.path.join(d, "a.npy")}, {"c": os.path.join(d, "b.npy")}]
        old = ce.NPY_BLOCK
        ce.NPY_BLOCK = 300
        try:
            x = ce.unified_element_definition(cov, "c", "+", os.path.join(d, "x.bed"))
        finally:
            ce.NPY_BLOCK = old
        y = _whole_array_ued(cov, "c", os.path.join(d, "y.bed"))
        assert file_bytes(x) == file_bytes(y)


@pytest.mark.parametrize("case", ["float", "length_mismatch", "object", "empty"])
def test_blockwise_element_definition_fallbacks(case):
    """Inputs outside the block-wise contract use the original code (same result / same exception type)."""
    with tempfile.TemporaryDirectory() as d:
        a = np.where(np.arange(5000) % 97 < 3, 5, 0).astype(np.int16)
        np.save(os.path.join(d, "a.npy"), a)
        if case == "float":
            np.save(os.path.join(d, "b.npy"), a.astype(np.float64))
        elif case == "length_mismatch":
            np.save(os.path.join(d, "b.npy"), a[:4000])
        elif case == "object":
            np.save(os.path.join(d, "b.npy"), a.astype(object), allow_pickle=True)
        else:
            np.save(os.path.join(d, "a.npy"), a[:0])
            np.save(os.path.join(d, "b.npy"), a[:0])
        cov = [{"c": os.path.join(d, "a.npy")}, {"c": os.path.join(d, "b.npy")}]
        assert ce._blockwise_element_definition(cov, "c", os.path.join(d, "q.bed"), 1, 1.0, 3) is False

        def run(fn):
            try:
                return ("ok", file_bytes(fn()))
            except Exception as e:  # noqa: BLE001
                return ("err", type(e))

        assert run(lambda: ce.unified_element_definition(cov, "c", "+", os.path.join(d, "x.bed"))) == \
            run(lambda: _whole_array_ued(cov, "c", os.path.join(d, "y.bed")))


# ------------------------------------------------------------------ bigWig loader
class _FakeBW:
    """Minimal pyBigWig stand-in: intervals(chrom, a, b) returns overlapping intervals in stored order."""

    def __init__(self, iv):
        self.iv = iv

    def intervals(self, chrom, a=None, b=None):
        if a is None:
            return tuple(self.iv)
        return tuple(x for x in self.iv if x[1] > a and x[0] < b)


def _dense_from_intervals(iv, csize, dt):
    v = np.full(csize, np.nan)
    for s, e, x in iv:
        v[s:e] = x
    with np.errstate(invalid="ignore"):
        v = np.nan_to_num(v)
        v[v < 0] *= -1
        return v.astype(dt, copy=False)


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from(DTYPES), st.sampled_from([1, 7, 64, 1000, 1 << 22]),
       st.integers(1, 3000), st.booleans())
def test_bw_blockwise_sparse_equals_dense(seed, dt, block, csize, shuffled):
    rng = np.random.default_rng(seed)
    iv, pos = [], 0
    while pos < csize:
        pos += int(rng.integers(0, 50))
        ln = int(rng.integers(1, 400 if rng.random() < 0.2 else 4))
        if pos + ln > csize:
            break
        iv.append((pos, pos + ln, float(rng.choice([rng.uniform(-100, 100), 0.0, -0.5, np.nan, np.inf]))))
        pos += ln
    if shuffled:  # disjoint but stored unsorted: per-block order is what matters
        iv = [iv[i] for i in rng.permutation(len(iv))]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npy")
        with np.errstate(invalid="ignore"):
            s = io._bw_blockwise_sparse(_FakeBW(iv), "c", csize, dt, p, block=block)
            ref = _dense_from_intervals(iv, csize, dt)
        if s is None:
            assert shuffled  # only an unsorted block may be rejected
            return
        np.save(os.path.join(d, "r.npy"), ref)
        assert open(p, "rb").read() == open(os.path.join(d, "r.npy"), "rb").read()
        assert type(s) == type(np.sum(ref)) and s == np.sum(ref)


@pytest.mark.parametrize("iv", [
    [(0, 10, 1.0), (5, 12, 2.0)],          # overlap inside a block
    [(0, 10, 1.0), (9, 30, 2.0)],          # overlap at a block edge
    [(3.5, 10, 1.0)],                      # non-integer coordinate
    [(0, 10, 1.0), (20, 60, 1.0)],         # beyond the chromosome end
    [(-2, 5, 1.0)],                        # negative start
])
def test_bw_blockwise_sparse_rejects_bad_intervals(iv):
    with tempfile.TemporaryDirectory() as d:
        assert io._bw_blockwise_sparse(_FakeBW(iv), "c", 50, np.int16, os.path.join(d, "x.npy"), block=8) is None


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 5, 333]))
def test_bw_chrom_worker_blockwise_equals_reference(seed, block):
    """Real bigWig files (bedGraph spans crossing blocks) through the parallel loader vs pints_ref."""
    import pyBigWig
    rng = np.random.default_rng(seed)
    chroms = [("chr1", 4000), ("chr2", 900), ("chrE", 300)]
    old = io._BW_IV_BLOCK
    io._BW_IV_BLOCK = block
    try:
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for tag, sign in (("p", 1), ("m", -1)):
                path = os.path.join(d, tag + ".bw")
                bw = pyBigWig.open(path, "w")
                bw.addHeader(chroms)
                for c, L in chroms:
                    if c == "chrE":
                        continue
                    s, e, v, pos = [], [], [], 0
                    while True:
                        pos += int(rng.integers(0, 30))
                        ln = int(rng.integers(1, 120))
                        if pos + ln > L:
                            break
                        s.append(pos); e.append(pos + ln); v.append(sign * float(rng.uniform(0, 300)))
                        pos += ln
                    if s:
                        bw.addEntries([c] * len(s), s, ends=e, values=v)
                bw.close()
                paths.append(path)
            os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
            a = io.get_coverage_bw(paths[0], paths[1], "", os.path.join(d, "a"), "q", threads=3)
            b = rio.get_coverage_bw(paths[0], paths[1], "", os.path.join(d, "b"), "q")
            assert type(a[2]) == type(b[2]) and a[2] == b[2]
            for k in (0, 1):
                assert list(a[k]) == list(b[k])
                for c in a[k]:
                    assert file_bytes(a[k][c]) == file_bytes(b[k][c])
    finally:
        io._BW_IV_BLOCK = old


# ------------------------------------------------------------------ input normalisation
@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 2, 100, 1000, 7, 4096]),
       st.sampled_from([0.0, 0.37, 1.0, 2.659, -1.0, float("nan"), float("inf")]),
       st.sampled_from(DTYPES), st.sampled_from([None, 126, 127, 128, 32766, 32767, 32768, 2 ** 31]),
       st.integers(1, 5000))
def test_blockwise_normalize_equals_whole_array(seed, block, sf, dt, peak, L):
    rng = np.random.default_rng(seed)
    hi = int(np.iinfo(dt).max)
    exp = np.where(rng.random(L) < 0.2, rng.integers(0, min(hi, 300) + 1, L), 0).astype(np.int64)
    if peak is not None and peak <= hi:
        exp[int(rng.integers(0, L))] = peak  # dtype boundary of the output
    exp = exp.astype(dt)
    inp = np.where(rng.random(L) < 0.2, rng.integers(0, min(hi, 50) + 1, L), 0).astype(dt)
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "e.npy"), exp)
        np.save(os.path.join(d, "i.npy"), inp)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        fa = os.path.join(d, "a", "n_pl_c.npy")
        with np.errstate(invalid="ignore", over="ignore"):
            ra = io._normalize_chrom_worker(os.path.join(d, "e.npy"), os.path.join(d, "i.npy"), sf, fa, block=block)
            rb = rio.normalize_using_input({"c": os.path.join(d, "e.npy")}, {}, {"c": os.path.join(d, "i.npy")},
                                           {}, sf, os.path.join(d, "b"), "n", logging.getLogger("t"))[0]
        assert (ra is None) == ("c" not in rb)
        if ra is not None:
            assert file_bytes(ra) == file_bytes(rb["c"])


def test_blockwise_normalize_shape_mismatch_falls_back():
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "e.npy"), np.arange(10, dtype=np.int16))
        np.save(os.path.join(d, "i.npy"), np.arange(1, dtype=np.int16))  # broadcasts in the original code
        r = io._normalize_chrom_worker(os.path.join(d, "e.npy"), os.path.join(d, "i.npy"), 2.0,
                                       os.path.join(d, "o.npy"), block=3)
        ref = np.arange(10, dtype=np.int16) - np.arange(1, dtype=np.int16) * 2.0
        ref[ref < 0] = 0
        np.save(os.path.join(d, "r.npy"), ref.astype(np.int8))
        assert open(r, "rb").read() == open(os.path.join(d, "r.npy"), "rb").read()


# ------------------------------------------------------------------ replicate merging
@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 2, 100, 1000, 7, 4096]),
       st.lists(st.sampled_from(DTYPES), min_size=2, max_size=3),
       st.sampled_from([None, 63, 64, 127, 128, 16383, 32767, 32768, 2 ** 31 - 1, 2 ** 31]),
       st.integers(1, 5000))
def test_blockwise_merge_equals_reference(seed, block, dts, peak, L):
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as d:
        pls, mns = [], []
        for i, dt in enumerate(dts):
            hi = int(np.iinfo(dt).max)
            for tag, out in (("p", pls), ("m", mns)):
                v = np.where(rng.random(L) < 0.2, rng.integers(0, min(hi, 300) + 1, L), 0).astype(np.int64)
                if peak is not None and peak <= hi and tag == "p":
                    v[int(rng.integers(0, L))] = peak  # output dtype boundary, int32 wrap-around for 2**31
                fn = os.path.join(d, "%s%d.npy" % (tag, i))
                np.save(fn, v.astype(dt))
                out.append({"c": fn})
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        fa = [os.path.join(d, "a", "m_%s_c.npy" % s) for s in ("pl", "mn")]
        with np.errstate(over="ignore"):
            sa = io._merge_rep_chrom_worker([x["c"] for x in pls], [x["c"] for x in mns], fa[0], fa[1], block=block)
            rb = rio.merge_replicates_bw(pls, mns, os.path.join(d, "b"), "m")
        assert sa == (rb[2], rb[3]) and type(sa[0]) == type(rb[2]) and type(sa[1]) == type(rb[3])
        assert file_bytes(fa[0]) == file_bytes(rb[0]["c"]) and file_bytes(fa[1]) == file_bytes(rb[1]["c"])


def test_blockwise_merge_serial_path_equals_reference():
    rng = np.random.default_rng(7)
    chroms = {"chr1": 5000, "chr2": 3001, "chrZ": 1}
    with tempfile.TemporaryDirectory() as d:
        pls, mns = [], []
        for i in range(3):
            for tag, out in (("p", pls), ("m", mns)):
                dd = {}
                for c, L in chroms.items():
                    fn = os.path.join(d, "%s%d_%s.npy" % (tag, i, c))
                    np.save(fn, rng.integers(0, 200, L).astype(np.int16))
                    dd[c] = fn
                out.append(dd)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        old = io.NPY_BLOCK
        io.NPY_BLOCK = 333
        try:
            a = io.merge_replicates_bw(pls, mns, os.path.join(d, "a"), "m", threads=1)
        finally:
            io.NPY_BLOCK = old
        b = rio.merge_replicates_bw(pls, mns, os.path.join(d, "b"), "m")
        assert a[2] == b[2] and a[3] == b[3] and type(a[2]) == type(b[2])
        for k in (0, 1):
            assert list(a[k]) == list(b[k])
            for c in a[k]:
                assert file_bytes(a[k][c]) == file_bytes(b[k][c])


def test_blockwise_merge_length_mismatch_falls_back():
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "a.npy"), np.arange(10, dtype=np.int16))
        np.save(os.path.join(d, "b.npy"), np.arange(1, dtype=np.int16))  # broadcasts in the original code
        assert io._merge_rep_strand([os.path.join(d, "a.npy"), os.path.join(d, "b.npy")],
                                    os.path.join(d, "o.npy"), block=3) is None
        s = io._merge_rep_chrom_worker([os.path.join(d, "a.npy"), os.path.join(d, "b.npy")],
                                       [os.path.join(d, "a.npy"), os.path.join(d, "a.npy")],
                                       os.path.join(d, "p.npy"), os.path.join(d, "q.npy"), block=3)
        ref = np.arange(10, dtype=np.int16).astype(np.int32) + np.arange(1, dtype=np.int16)
        assert s[0] == ref.sum() and s[1] == 2 * np.arange(10).sum()
        np.save(os.path.join(d, "r.npy"), ref.astype(np.int8))
        assert open(os.path.join(d, "p.npy"), "rb").read() == open(os.path.join(d, "r.npy"), "rb").read()


@pytest.mark.parametrize("values, expect", [((100, 20), np.int8),    # fits the inputs' dtype: one pass
                                            ((100, 28), np.int16)])  # exceeds it: rewritten as int16
def test_blockwise_merge_dtype_guess(values, expect):
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, v in enumerate(values):
            a = np.zeros(1000, dtype=np.int8)
            a[500] = v
            files.append(os.path.join(d, "r%d.npy" % i))
            np.save(files[-1], a)
        s = io._merge_rep_strand(files, os.path.join(d, "o.npy"), block=64)
        ref = np.load(files[0]).astype(np.int32) + np.load(files[1])
        np.save(os.path.join(d, "ref.npy"), ref.astype(expect))
        assert s == ref.sum() and np.load(os.path.join(d, "o.npy")).dtype == expect
        assert file_bytes(os.path.join(d, "o.npy")) == file_bytes(os.path.join(d, "ref.npy"))
