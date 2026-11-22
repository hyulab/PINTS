"""Differential tests for the exact fast ZIP EM path (Numba replica of numpy's pairwise summation), compared
bitwise against pints_ref, including its fallback when Numba is unavailable."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.stats_engine as se
import pints_ref.stats_engine as rse
from test_differential import N

FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))


def _same_fit(x, y):
    return len(x) == len(y) and all(type(p) == type(q) and (p == q or (p != p and q != q)) for p, q in zip(x, y))


@FAST
@given(st.integers(0, 2 ** 31 - 1),
       st.sampled_from([3, 6, 8, 9, 20, 127, 128, 129, 1000, 5000, 8191, 8192, 8193, 10001, 20000]),
       st.sampled_from([np.int8, np.int16, np.int32, np.int64, np.uint16, np.uint32, np.float32, np.float64]),
       st.floats(0, 1), st.floats(-2, 2), st.booleans())
def test_zip_fit_fast_path_bit_identical(seed, length, dt, pz, loglam, negate):
    rng = np.random.default_rng(seed)
    w = rng.poisson(10 ** loglam, length)
    w[rng.random(length) < pz] = 0
    if negate and np.dtype(dt).kind != "u":
        w = -w
    w = w.astype(dt)
    assert _same_fit(se.ZIP().fit(w), rse.ZIP().fit(w))
    assert _same_fit(se.ZIP(init_mu=1.0, init_pi=0.5).fit(w), rse.ZIP(init_mu=1.0, init_pi=0.5).fit(w))


def test_zip_fit_fast_path_is_used(monkeypatch):
    """Guard: integer windows must take the fast path (and still match the original loop)."""
    calls = []
    orig = se.ZIP._fit_fast
    monkeypatch.setattr(se.ZIP, "_fit_fast", lambda *a: calls.append(1) or orig(*a))
    w = np.array([0, 0, 3, 0, 1, 0, 0, 7, 0, 2] * 50, dtype=np.int32)
    assert _same_fit(se.ZIP().fit(w), rse.ZIP().fit(w)) and calls


@pytest.mark.parametrize("pz", [0.0, 0.999])
def test_zip_fit_fast_path_falls_back_consistently(monkeypatch, pz):
    rng = np.random.default_rng(1)
    w = rng.poisson(3, 4000)
    w[rng.random(4000) < pz] = 0
    monkeypatch.setattr(se, "_HAS_NUMBA", False)
    slow = se.ZIP().fit(w)
    monkeypatch.setattr(se, "_HAS_NUMBA", True)
    assert _same_fit(slow, se.ZIP().fit(w)) and _same_fit(slow, rse.ZIP().fit(w))


