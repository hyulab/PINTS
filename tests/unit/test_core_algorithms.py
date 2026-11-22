"""Characterisation tests: interval utilities, segment joining, statistics (hand-computed expectations)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from pints.calling_engine import cut_peaks, merge_intervals, quasi_max_score_segment, sliding_window
from pints.stats_engine import IQR, ZIP, Poisson, get_elbow, independent_filtering


def _tuples(x):
    return [tuple(int(v) for v in t) for t in x]


# ------------------------------------------------------------ merge_intervals
@pytest.mark.parametrize("intervals,distance,expected", [
    ([(6, 8), (1, 9), (2, 4), (4, 7)], 0, [(1, 9)]),        # containment case from the docstring
    ([(1, 3), (5, 7)], 1, [(1, 3), (5, 7)]),               # gap 2 > distance 1
    ([(1, 3), (5, 7)], 2, [(1, 7)]),                       # gap == distance -> merge
    ([(1, 3), (3, 5)], 0, [(1, 5)]),                       # book-ended
    ([(5, 7)], 0, [(5, 7)]),
    ([], 0, []),
])
def test_merge_intervals(intervals, distance, expected):
    assert _tuples(merge_intervals(list(intervals), distance)) == expected


def test_merge_intervals_negative_distance_exits():
    with pytest.raises(SystemExit):
        merge_intervals([(1, 2)], -1)


# ------------------------------------------------------------ sliding_window
def test_sliding_window_sums_and_coords():
    cov = np.arange(250)
    got = [(int(w), c) for w, c in sliding_window(cov)]
    assert got == [(4950, (0, 100)), (14950, (100, 200))]


def test_sliding_window_custom_step():
    cov = np.ones(10, dtype=np.int32)
    got = [(int(w), c) for w, c in sliding_window(cov, window_size=4, step_size=2)]
    assert got == [(4, (0, 4)), (4, (2, 6)), (4, (4, 8)), (4, (6, 10))]


@pytest.mark.parametrize("kw", [dict(step_size=0)])
def test_sliding_window_invalid(kw):
    with pytest.raises(ValueError):
        list(sliding_window(np.ones(10), **kw))


def test_sliding_window_empty():
    with pytest.raises(ValueError):
        list(sliding_window(np.array([])))


def test_sliding_window_shorter_than_window_yields_nothing():
    assert list(sliding_window(np.ones(50))) == []


# ------------------------------------------------------------ segment joining
def test_quasi_max_score_segment_merges_both_directions():
    f, r = quasi_max_score_segment([(0, 4, 8, 2.0), (6, 10, 8, 2.0)], 0.5, 3, 30)
    assert f == [(0, 10, 16, 1.6), (6, 10, 8, 2.0)]
    assert r == [(0, 10, 16, 1.6), (0, 4, 8, 2.0)]


def test_quasi_max_score_segment_respects_max_distance_and_trigger():
    c = [(0, 4, 8, 2.0), (50, 54, 8, 2.0)]
    f, r = quasi_max_score_segment(c, 0.0, 3, 30)   # gap 46 > 30 -> no merge
    assert f == c and r == c[::-1]
    c2 = [(0, 2, 8, 4.0), (3, 7, 8, 2.0)]           # first seed narrower than ce_trigger
    f2, _ = quasi_max_score_segment(c2, 0.0, 3, 30)
    assert f2[0] == c2[0]


def test_quasi_max_score_segment_empty():
    assert quasi_max_score_segment([], 0.3, 3, 30) == ([], [])


def test_cut_peaks_donor_tolerance():
    w = np.array([0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 2, 6, 2, 0])
    assert _tuples(cut_peaks(w, 0.3, 3)) == [(0, 13)]
    assert _tuples(cut_peaks(w, 1.0, 3)) == [(0, 4), (9, 13)]


def test_cut_peaks_no_peak():
    assert cut_peaks(np.zeros(20, dtype=int), 0.3, 3) == []


# ------------------------------------------------------------ ZIP / Poisson
def test_zip_moment_estimators_overdispersed():
    lamb, pi = ZIP.zip_moment_estimators(np.array([0, 0, 0, 4]))
    assert lamb == 3.0 and pi == pytest.approx(2 / 3, abs=0)


def test_zip_moment_estimators_underdispersed():
    lamb, pi = ZIP.zip_moment_estimators(np.array([2, 2, 2, 2]))
    assert lamb == 2.0 and pi == 0


def test_zip_log_likelihood_core_hand():
    pi, lamb = 0.5, 1.0
    p0 = pi + (1 - pi) * math.exp(-lamb)
    p2 = (1 - pi) * math.exp(-lamb) / 2
    got = ZIP._log_likelihood_core(np.array([0, 2]), np.array([3, 1]), pi, lamb)
    assert got == pytest.approx(3 * math.log(p0) + math.log(p2), rel=1e-12)


def test_zip_log_likelihood_floor():
    # pmf below machine epsilon is floored at eps
    got = ZIP._log_likelihood_core(np.array([200]), np.array([1]), 0.0, 1.0)
    assert got == pytest.approx(math.log(np.finfo(float).eps))


def test_zip_fit_all_zero():
    assert ZIP().fit(np.zeros(10, dtype=np.int32)) == (0, 0, 0, 0, True)


def test_zip_fit_no_zero():
    mu, var, pi, ll, conv = ZIP().fit(np.array([1, 2, 3]))
    assert (mu, var, pi, conv) == (2.0, 2.0, 0, True) and np.isnan(ll)


def test_zip_fit_recovers_parameters():
    rng = np.random.default_rng(299)
    n, mu, pi = 5000, 2.5, 0.55
    x = (rng.random(n) > pi) * rng.poisson(mu, n)
    mu_hat, var_hat, pi_hat, ll, conv = ZIP().fit(x)
    assert conv and mu_hat == var_hat
    assert mu_hat == pytest.approx(mu, abs=0.1) and pi_hat == pytest.approx(pi, abs=0.03)


def test_zip_fit_max_iter_flag():
    z = ZIP(max_iter=0, stop_diff=0)
    _, _, _, _, conv = z.fit(np.array([0, 0, 0, 0, 5, 1, 0, 0, 3]))
    assert conv is False


def test_zip_lrt_identical_groups_not_significant():
    rng = np.random.default_rng(1)
    x = (rng.random(400) > 0.6) * rng.poisson(3, 400)
    stat, p = ZIP().lrt(x, x.copy())
    assert stat == pytest.approx(0, abs=1e-6) and p == pytest.approx(1, abs=1e-6)


def test_zip_lrt_different_groups_significant():
    rng = np.random.default_rng(2)
    a = (rng.random(300) > 0.3) * rng.poisson(8, 300)
    b = (rng.random(3000) > 0.9) * rng.poisson(1, 3000)
    stat, p = ZIP().lrt(a, b)
    assert stat > 100 and p < 1e-20


def test_poisson_model_can_be_instantiated():
    # Poisson implements `lrt` since the bugfix PR (before, `--model Poisson` crashed with TypeError)
    statistic, p_value = Poisson().lrt(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert statistic == 0. and p_value == 1.


def test_poisson_model_methods():
    assert Poisson.fit(None, np.array([1, 2, 3])) == (2.0, 2.0, 0, None, True)
    assert Poisson.sf(None, 5, None, None, 2.0, None, None) == poisson.sf(5, 2.0)


def test_zip_sf_is_poisson_sf_of_mu():
    # documents current behaviour: the ZIP survival function is the Poisson one at mu
    assert ZIP().sf(5.7, 1, 0.3, 2.0, 2.0, 0.1) == poisson.sf(5.7, 2.0)


# ------------------------------------------------------------ IQR / elbow
def test_iqr_thresholds():
    assert IQR.get_outlier_threshold([1, 2, 3, 4, 5], 0) == (0.0, 6.0)
    assert IQR.get_outlier_threshold([1, 2, 3, 4, 5], 1) == (None, 6.0)
    assert IQR.get_outlier_threshold([1, 2, 3, 4, 5], -1) == (0.0, None)


def test_get_elbow():
    x, y = get_elbow(np.arange(5), np.array([0, 0, 0, 0, 4.0]))
    assert (x, y) == (3, 0.0)


def test_get_elbow_degenerate():
    assert all(np.isnan(v) for v in get_elbow([1, 2], [1, 2]))
    assert all(np.isnan(v) for v in get_elbow([1, 2, 3], [1, 2]))


def test_get_elbow_tie_takes_last():
    x, _ = get_elbow(np.arange(5), np.array([0, 1, 1, 1, 0.0]))
    assert x == 3


# ------------------------------------------------------------ independent filtering
def _if_df(rng, n=400):
    reads = rng.integers(1, 200, n)
    # low-count rows are uniform noise; high-count rows carry signal
    p = np.where(reads > 100, rng.uniform(0, 1e-3, n), rng.uniform(0, 1, n))
    return pd.DataFrame({"chromosome": "chr1", "start": np.arange(n) * 100, "end": np.arange(n) * 100 + 50,
                         "name": ".", "pval": p, "strand": "+", "reads": reads})


def test_independent_filtering_improves_or_keeps_rejections(rng, logger):
    from statsmodels.stats.multitest import multipletests
    df = _if_df(rng)
    out = independent_filtering(df, fdr_target=0.1, logger=logger)
    direct = (multipletests(df.pval, method="fdr_bh")[1] < 0.1).sum()
    assert (out.padj < 0.1).sum() >= direct
    assert list(out.columns[:len(df.columns)]) == list(df.columns)
    assert len(out) == len(df)


def test_independent_filtering_filtered_rows_keep_direct_padj(rng, logger):
    from statsmodels.stats.multitest import multipletests
    df = _if_df(rng)
    out = independent_filtering(df, fdr_target=0.1, logger=logger)
    direct = multipletests(df.pval, method="fdr_bh")[1]
    # rows below the selected read threshold keep their unfiltered BH value
    thr = out.loc[out.padj.values != direct, "reads"].min()
    low = df.reads <= df.reads.min()
    assert np.array_equal(out.loc[low, "padj"].values, direct[low.values]) or thr is np.nan
