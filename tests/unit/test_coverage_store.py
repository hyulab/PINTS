"""CoverageTrack against the dense array it stands for: windows, fill/clear, sum, max, .npy round trips."""
import os
import pickle
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from pints import coverage_store as cs
from pints.io_engine import NP_DT_NAME

PROPS = settings(max_examples=200, deadline=None, suppress_health_check=list(HealthCheck))


@st.composite
def dense_tracks(draw):
    """A dense integer array (mostly zeros, with points and runs) and its track in points or runs form."""
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.sampled_from([0, 1, 7, 100, 2_500, 40_000]))
    dtype = draw(st.sampled_from(list(NP_DT_NAME) + [np.uint32]))
    arr = np.zeros(n, dtype=dtype)
    if n:
        hi = min(int(np.iinfo(dtype).max), 1000)
        k = int(rng.integers(0, max(1, n // 20) + 1))
        arr[rng.integers(0, n, k)] = rng.integers(1, hi + 1, k)
        for _ in range(int(rng.integers(0, 4))):  # a few longer constant runs, as bigWig intervals give
            a = int(rng.integers(0, n))
            arr[a:a + int(rng.integers(1, 300))] = rng.integers(1, hi + 1)
        if draw(st.booleans()):
            arr[:] = np.where(rng.random(n) < 0.002, 0, arr)  # holes inside runs
    runs = draw(st.booleans())
    if runs:
        nz = arr != 0
        edges = np.flatnonzero(np.diff(np.concatenate(([0], arr, [0])).astype(np.int64)) != 0)
        # split into maximal runs of equal non-zero value
        starts, ends, vals = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            if nz[a]:
                starts.append(a); ends.append(b); vals.append(arr[a])
        tr = cs.CoverageTrack(n, dtype, starts, vals, ends)
    else:
        tr = cs.track_from_dense(arr)
    return arr, tr


def _bounds(rng, n):
    pts = [0, n, -1, -n - 5, n + 10, 2_500, -2_500] + [int(x) for x in rng.integers(-n - 10, n + 10, 12)]
    return [(a, b) for a in pts for b in pts[:6]] + list(zip(pts, pts[::-1]))


@PROPS
@given(dense_tracks(), st.integers(0, 2 ** 31 - 1))
def test_window_matches_dense_slice(t, seed):
    arr, tr = t
    assert tr.length == arr.shape[0] and tr.dtype == arr.dtype
    np.testing.assert_array_equal(tr.dense(), arr)
    assert tr.dense().dtype == arr.dtype
    for a, b in _bounds(np.random.default_rng(seed), arr.shape[0]):
        w = tr.window(a, b)
        assert w.dtype == arr.dtype and np.array_equal(w, arr[a:b]), (a, b)


@PROPS
@given(dense_tracks(), st.integers(0, 2 ** 31 - 1))
def test_fill_and_clear_in_a_full_length_buffer(t, seed):
    arr, tr = t
    buf = tr.empty_buffer()
    rng = np.random.default_rng(seed)
    for a, b in _bounds(rng, arr.shape[0])[:10]:
        tr.fill(buf, a, b)
        lo, hi = max(a, 0), min(max(b, 0), arr.shape[0])
        exp = np.zeros_like(arr)
        if hi > lo:
            exp[lo:hi] = arr[lo:hi]
        assert np.array_equal(buf, exp), (a, b)
        tr.clear(buf, a, b)
        assert not buf.any()


@PROPS
@given(dense_tracks())
def test_reductions_match_dense(t):
    arr, tr = t
    s = tr.sum()
    assert s == arr.sum() and s.dtype == arr.sum().dtype
    assert tr.is_zero == (not arr.any())
    if arr.shape[0]:
        m = tr.max()
        assert m == arr.max() and m.dtype == arr.max().dtype
    else:
        with pytest.raises(ValueError):
            tr.max()
    st_, en, va = tr.nonzero()
    exp = np.zeros_like(arr)
    for a, b, v in zip(st_, en, va):
        exp[a:b] = v
    assert np.array_equal(exp, arr) and (va != 0).all()


@PROPS
@given(dense_tracks())
def test_npy_round_trip_is_np_save(t):
    arr, tr = t
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.npy"), os.path.join(d, "b.npy")
        np.save(a, arr)
        cs.track_to_npy(tr, b)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()
        back = cs.track_from_npy(a)
        assert back.dtype == arr.dtype and np.array_equal(back.dense(), arr)


def test_registry_keys_pickle_small_and_tracks_refuse():
    tr = cs.track_from_dense(np.array([0, 3, 0, 1], dtype=np.int16))
    key = cs.register(0, "pl", "chr1", tr)
    try:
        assert cs.is_key(key) and cs.lookup(key) is tr
        k2 = pickle.loads(pickle.dumps(key))
        assert k2 == key and cs.is_key(k2) and cs.lookup(k2) is tr
        with pytest.raises(TypeError):
            pickle.dumps(tr)
    finally:
        cs.release([key])
    with pytest.raises(KeyError):
        cs.lookup(key)


def test_float_tracks_are_rejected():
    with pytest.raises(TypeError):
        cs.CoverageTrack(4, np.float64, [1], [2.0])
