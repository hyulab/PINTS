"""adjust_pvalues reproduces statsmodels' multipletests bit for bit for the methods pints_caller offers."""
import numpy as np
import pandas as pd
import pytest

from pints.stats_engine import adjust_pvalues

multitest = pytest.importorskip("statsmodels.stats.multitest")
METHODS = ("fdr_bh", "bonferroni", "fdr_tsbh", "fdr_tsbky")


def _cases():
    rng = np.random.default_rng(7)
    signal = rng.beta(0.05, 5, 400)
    yield "uniform", rng.random(1000)
    yield "mixture", np.concatenate((signal, rng.random(3000)))
    yield "ties", np.round(rng.random(2000), 2)
    yield "zeros and ones", np.array([0., 1., 0., 0.5, 1., 1e-300, 0.])
    yield "all significant", np.full(50, 1e-12)
    yield "none significant", rng.uniform(0.5, 1, 200)
    yield "single", np.array([0.03])
    yield "nan", np.array([0.01, np.nan, 0.2, 0.04])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("alpha", (0.05, 0.1, 0.5))
def test_matches_statsmodels(method, alpha):
    for name, p in _cases():
        expected = multitest.multipletests(p, alpha=alpha, method=method)[1]
        got = adjust_pvalues(p, alpha=alpha, method=method)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected, err_msg=name)  # NaN == NaN here; values compared exactly


def test_series_input_keeps_positions():
    p = pd.Series([0.04, 0.001, 0.3, 0.02], index=[10, 3, 7, 1])
    expected = multitest.multipletests(p, alpha=0.1, method="fdr_bh")[1]
    np.testing.assert_array_equal(adjust_pvalues(p, 0.1), expected)


@pytest.mark.parametrize("method", METHODS)
def test_no_pvalues_raise_zero_division_like_statsmodels(method):
    # independent_filtering falls back to Bonferroni on this error
    with pytest.raises(ZeroDivisionError):
        multitest.multipletests(np.array([]), method=method)
    with pytest.raises(ZeroDivisionError):
        adjust_pvalues(np.array([]), method=method)


def test_unknown_method():
    with pytest.raises(ValueError):
        adjust_pvalues(np.array([0.1]), method="holm")
