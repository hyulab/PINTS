#!/usr/bin/env python
# coding=utf-8
# Created by: Li Yao (ly349@cornell.edu)
# Created on: 2019-08-07
#
# PINTS: Peak Identifier for Nascent Transcript Starts
# Copyright (C) 2019-2025 Yu Lab.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
import sys

try:
    import pysam
    import numpy as np
    from scipy.stats import poisson, chi2
    from scipy.special import gamma, digamma, polygamma, gammaln, psi, factorial, xlogy
    from scipy.optimize import fmin_l_bfgs_b as BFGS
    from abc import ABC, abstractmethod
    from collections import defaultdict
except ImportError as e:
    missing_package = str(e).replace("No module named '", "").replace("'", "")
    sys.exit("Please install %s first!" % missing_package)

__CACHE_KEY_FORMAT__ = "%d-%d"


def pval_dist(pval_list, logger, output_diagnostics=True, output_to=None):
    """
    Plot pval distribution (histogram)

    Parameters
    ----------
    pval_list : array-like
        p-values
    logger : Python Logger Objects
        Logger for recording errors / info
    output_diagnostics : bool
        Whether to generate the histogram
    output_to : str or None
        Path for the output file

    Returns
    -------

    """
    pval_list_filtered = pval_list[np.logical_and(pval_list >= 0.0, pval_list <= 1)]
    if output_diagnostics and output_to is not None:
        if len(pval_list_filtered) < 20:
            logger.error("Cannot get enough p-values for binning")
            return
        import matplotlib.pyplot as plt
        plt.hist(pval_list_filtered, bins=20, range=(0, 1), density=True, color="powderblue", )
        plt.xlabel("$p$-value")
        plt.ylabel("Density")
        plt.xlim((0, 1))
        plt.tight_layout()
        plt.savefig(output_to, transparent=True, bbox_inches="tight")
        plt.close()
        logger.info("Diagnostic plot for p-values was wrote to %s" % output_to)


def _poisson_pmf(k, mu):
    """Bit-exact replica of ``scipy.stats.poisson.pmf(k, mu)`` for integer ``k`` >= 0 and scalar ``mu``.

    scipy's generic rv_discrete.pmf wrapper (argument parsing, support checks, argsreduce, place) costs
    ~50 us per call; the arithmetic below is the identical sequence of ufunc calls
    (k is promoted to int64 exactly as ``asarray(k) - asarray(loc)`` does in scipy).
    Falls back to scipy for anything outside that contract.
    """
    k = np.asarray(k)
    if k.ndim != 1 or k.dtype.kind not in "iu" or np.ndim(mu) != 0:
        return poisson.pmf(k, mu)
    k = k - np.asarray(0)
    if not (mu >= 0):
        return np.full(k.shape, np.nan)
    if k.shape[0] == 0 or k.min() < 0:
        return poisson.pmf(k, mu)
    mu_b = np.broadcast_to(np.asarray(mu), k.shape).ravel()
    return np.clip(np.exp(xlogy(k, mu_b) - gammaln(k + 1) - mu_b), 0, 1)


# ---------------------------------------------------------------------------------------------------
# Exact EM helpers. numpy reduces a contiguous float64 array in buffered blocks of 8192 elements, each
# block summed pairwise (8 accumulators, blocks <= 128 unrolled, split point n/2 rounded down to a
# multiple of 8), starting from 0.0. `_masked_sum` reproduces that summation tree for an array that is
# `a` where mask is True and `b` elsewhere, without materialising it, so the result is bit-identical to
# ``np.where(mask, a, b).sum()``.
# ---------------------------------------------------------------------------------------------------
try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _pw_masked(mask, lo, n, a, b):
        if n < 8:
            res = 0.0
            for i in range(lo, lo + n):
                res += a if mask[i] else b
            return res
        elif n <= 128:
            r0 = a if mask[lo] else b
            r1 = a if mask[lo + 1] else b
            r2 = a if mask[lo + 2] else b
            r3 = a if mask[lo + 3] else b
            r4 = a if mask[lo + 4] else b
            r5 = a if mask[lo + 5] else b
            r6 = a if mask[lo + 6] else b
            r7 = a if mask[lo + 7] else b
            i = 8
            m = n - (n % 8)
            while i < m:
                r0 += a if mask[lo + i] else b
                r1 += a if mask[lo + i + 1] else b
                r2 += a if mask[lo + i + 2] else b
                r3 += a if mask[lo + i + 3] else b
                r4 += a if mask[lo + i + 4] else b
                r5 += a if mask[lo + i + 5] else b
                r6 += a if mask[lo + i + 6] else b
                r7 += a if mask[lo + i + 7] else b
                i += 8
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n:
                res += a if mask[lo + i] else b
                i += 1
            return res
        else:
            n2 = n // 2
            n2 -= n2 % 8
            return _pw_masked(mask, lo, n2, a, b) + _pw_masked(mask, lo + n2, n - n2, a, b)

    @_nb.njit(cache=True)
    def _masked_sum(mask, a, b):
        res = 0.0
        n = mask.shape[0]
        lo = 0
        while lo < n:
            m = min(8192, n - lo)
            res += _pw_masked(mask, lo, m, a, b)
            lo += m
        return res

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


_EPS = np.finfo(float).eps


class _LLCache:
    """Per-fit invariants of the ZIP log-likelihood over the unique counts of a window."""
    __slots__ = ("k", "logfact", "counts", "zero_first")

    def __init__(self, u_ele, c_ele):
        self.k = u_ele - np.asarray(0)   # same promotion as scipy's ``asarray(k) - asarray(loc)``
        self.logfact = gammaln(self.k + 1)
        self.counts = c_ele
        self.zero_first = u_ele[0] == 0  # u_ele is sorted and unique

    @staticmethod
    def supported(u_ele):
        return u_ele.ndim == 1 and u_ele.dtype.kind in "iu" and u_ele.shape[0] > 0 and u_ele[0] >= 0

    def log_likelihood(self, pi, lamb):
        """Bit-identical to ``ZIP._log_likelihood_core(u_ele, c_ele, pi, lamb)`` for lamb >= 0."""
        u_pmf = (1 - pi) * np.minimum(np.exp(xlogy(self.k, lamb) - self.logfact - lamb), 1.)
        if self.zero_first:
            u_pmf[0] += pi
        return (np.log(np.maximum(u_pmf, _EPS)) * self.counts).sum()


class StatModel(ABC):
    def __init__(self, init_mu=None, init_variance=None, init_pi=None, max_iter=1000, stop_diff=0.0001, debug=False,
                 output_to=""):
        self.init_mu = init_mu
        self.init_variance = init_variance
        self.init_pi = init_pi
        self.max_iter = max_iter
        self.stop_diff = stop_diff
        self.debug = debug
        self.output_to = output_to
        self.infinitesimal = np.finfo(float).eps
        super().__init__()

    @abstractmethod
    def fit(self, window):
        pass

    @abstractmethod
    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        pass

    @abstractmethod
    def lrt(self, group_1_obs, group_2_obs):
        pass


class Poisson(StatModel):
    def fit(self, window):
        """

        Returns
        -------
        mu : float

        var : float

        pi : None

        llc : float

        convergence : bool

        """
        m = np.mean(window)
        return m, m, 0, None, True

    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        """

        Parameters
        ----------
        peak_mu
        peak_var
        peak_pi
        le_mu
        le_var
        le_pi

        Returns
        -------

        """
        from scipy.stats import poisson
        return poisson.sf(peak_mu, le_mu)

    @staticmethod
    def _max_log_likelihood(obs):
        obs = np.asarray(obs, dtype=float)
        if obs.size == 0:
            return 0.
        m = obs.mean()
        if m <= 0:
            return 0.
        return float(np.sum(poisson.logpmf(obs, m)))

    def lrt(self, group_1_obs, group_2_obs):
        """
        Likelihood ratio test: separate Poisson rates vs one shared rate (chi-square, 1 df).

        Returns
        -------
        statistic : float
        p_value : float
        """
        g1 = np.asarray(group_1_obs)
        g2 = np.asarray(group_2_obs)
        statistic = 2 * (Poisson._max_log_likelihood(g1) + Poisson._max_log_likelihood(g2)
                         - Poisson._max_log_likelihood(np.concatenate((g1, g2))))
        statistic = max(statistic, 0.)
        return statistic, chi2.sf(statistic, df=1)


class ZIP(StatModel):
    @staticmethod
    def zip_moment_estimators(windows):
        """
        Moments estimators of ZIP
        :param windows: array-like
        :return: Corrected MMEs (lambda, pi)
        """
        s2 = windows.var()
        m = windows.mean()
        m2 = m ** 2
        if m >= s2:
            pi_mo = 0
            lamb_mo = m
        else:
            lamb_mo = (s2 + m2) / m - 1
            pi_mo = (s2 - m) / (s2 + m2 - m)

        return lamb_mo, pi_mo

    @staticmethod
    def _log_likelihood_core(count, count_numbers, pi, lamb):
        smallest = np.finfo(float).eps
        # estimate log likelihood
        # compute the Poisson PMF scaled by the non-zero-inflation component (i.e., the "Poisson part").
        u_pmf = (1 - pi) * _poisson_pmf(count, lamb)
        # handle the zero-inflation part correctly
        u_pmf[np.where(count == 0)] += pi
        u_pmf[u_pmf < smallest] = smallest
        likelihood = (np.log(u_pmf) * count_numbers).sum()
        return likelihood

    @staticmethod
    def log_likelihood(observations, pi, lamb):
        """
        Compute the log-likelihood of a zero-inflated Poisson (ZIP) model
        given observed count data.

        Parameters
        ----------
        observations : array-like
            A 1D array or list of observed count data (non-negative integers).
        pi : float
            The zero-inflation probability (0 <= pi <= 1), representing the probability
            of structural zeros in the ZIP model.
        lamb : float
            The Poisson mean (lambda > 0) for the count-generating part of the model.

        Returns
        -------
        float
            The total log-likelihood of the observed data under the specified zero-inflated Poisson model.

        Notes
        -----
            The ZIP model assumes that the probability mass function is:
                P(X=0) = pi + (1 - pi) * exp(-lambda)
                P(X=k) = (1 - pi) * Poisson(lambda) for k > 0

        """
        # extract unique observed values (e.g., 0, 1, 2, ...) and their frequencies
        u_ele, c_ele = np.unique(observations, return_counts=True)
        # estimate log likelihood
        return ZIP._log_likelihood_core(u_ele, c_ele, pi, lamb)

    def fit(self, window):
        """
        EM for Zero-inflated Poisson

        Parameters
        ----------
        window :
            array-like

        Returns
        -------
        mu : float

        var : float
            placeholder, the value is equal to mu
        pi : float

        llc : float

        convergence : bool

        """
        zero_elements = window == 0
        zs = zero_elements.sum()

        # if there's no non-zero observation
        if zs == window.shape[0]:
            return 0, 0, 0, 0, True

        if self.init_mu is None or self.init_pi is None:
            init_lamb, init_pi = ZIP.zip_moment_estimators(windows=window)
        else:
            init_lamb = self.init_mu
            init_pi = 0.8
        lamb = init_lamb
        pi = init_pi
        n_iter = 0
        hat_z = np.zeros(len(window))

        if zs == 0:
            m = window.mean()
            return m, m, 0, np.nan, True

        I = len(window)
        prev_likelihood = 0.
        likelihoods = []
        u_ele, c_ele = np.unique(window, return_counts=True)
        if _HAS_NUMBA and not self.debug and window.ndim == 1:
            fast = self._fit_fast(window, zero_elements, u_ele, c_ele, I, lamb, pi)
            if fast is not None:
                return fast
        while True:
            # expectation
            hat_z[zero_elements] = pi / (pi + np.exp(-lamb) * (1 - pi))
            # maximization
            pi = hat_z.sum() / I
            indicator = 1 - hat_z
            lamb = (indicator * window).sum() / indicator.sum()
            # estimate likelihood
            likelihood = ZIP._log_likelihood_core(u_ele, c_ele, pi, lamb)
            likelihoods.append(likelihood)
            if n_iter > 0:
                if abs(likelihood - prev_likelihood) < self.stop_diff:
                    if self.debug and self.output_to != "":
                        import matplotlib.pyplot as plt
                        plt.plot(likelihoods)
                        plt.ylabel('Log-likelihood')
                        plt.xlabel('Iteration')
                        plt.tight_layout()
                        plt.savefig(self.output_to, bbox_inches="tight", transparent=True)
                    return lamb, lamb, pi, likelihood, True
                if n_iter > self.max_iter:
                    return lamb, lamb, pi, likelihood, False
            prev_likelihood = likelihood
            n_iter += 1

    def _fit_fast(self, window, zero_elements, u_ele, c_ele, I, lamb, pi):
        """
        Same EM iterations as `fit`, without materialising hat_z / indicator arrays.

        hat_z is v on zero positions and 0 elsewhere, indicator = 1 - hat_z, so their sums are masked
        sums with numpy's summation tree, and for 0 <= v <= 1 (indicator * window) equals the float64
        cast of window bit for bit. Returns None (caller runs the original loop from the start) whenever
        a precondition does not hold, so results are always identical to the original implementation.
        """
        sum_w = window.astype(np.float64).sum()
        if not np.isfinite(sum_w) or not _LLCache.supported(u_ele):
            return None
        llc = _LLCache(u_ele, c_ele)
        mask = np.ascontiguousarray(zero_elements)
        n_iter = 0
        prev_likelihood = 0.
        while True:
            v = pi / (pi + np.exp(-lamb) * (1 - pi))
            if not (0. <= v <= 1.):
                return None
            pi = np.float64(_masked_sum(mask, float(v), 0.)) / I        # == hat_z.sum() / I
            lamb = sum_w / np.float64(_masked_sum(mask, float(1 - v), 1.))  # == (ind * w).sum() / ind.sum()
            if not (lamb >= 0):
                return None
            likelihood = llc.log_likelihood(pi, lamb)
            if n_iter > 0:
                if abs(likelihood - prev_likelihood) < self.stop_diff:
                    return lamb, lamb, pi, likelihood, True
                if n_iter > self.max_iter:
                    return lamb, lamb, pi, likelihood, False
            prev_likelihood = likelihood
            n_iter += 1

    def lrt(self, group_1_obs, group_2_obs):
        """
        Perform a likelihood ratio test (LRT) to compare two zero-inflated Poisson populations.

        This method tests the null hypothesis that both groups share the same
        ZIP parameters (lambda and pi) versus the alternative that each group
        has its own parameters.

        Parameters
        ----------
        group_1_obs : array-like
            Observed count data for group 1 (non-negative integers).
        group_2_obs : array-like
            Observed count data for group 2 (non-negative integers).

        Returns
        -------
        statistic : float
            The LRT statistic, computed as:
                W = 2 * (logL_group1 + logL_group2 - logL_pooled)
        p_value : float
            The p-value from the chi-square distribution with 2 degrees of freedom.
        """
        # convert input to NumPy arrays for consistent numerical operations
        group_1_obs = np.asarray(group_1_obs)
        group_2_obs = np.asarray(group_2_obs)

        # estimate ZIP parameters for each group using self.fit()
        lamb1, _, pi1, _, _ = self.fit(group_1_obs)
        lamb2, _, pi2, _, _ = self.fit(group_2_obs)

        # compute log-likelihoods under the unrestricted model (each group has its own params)
        loglik1 = ZIP.log_likelihood(group_1_obs, pi1, lamb1)
        loglik2 = ZIP.log_likelihood(group_2_obs, pi2, lamb2)

        # combine both groups and fit a common ZIP model (under H0)
        merged_obs = np.concatenate((group_1_obs, group_2_obs))
        lamb0, _, pi0, _, _ = self.fit(np.asarray(merged_obs))

        # compute log-likelihood under the null hypothesis (shared parameters)
        loglik_null = ZIP.log_likelihood(merged_obs, pi0, lamb0)

        # likelihood ratio test statistic
        statistic = 2 * (loglik1 + loglik2 - loglik_null)

        # compute p-value from chi-square distribution with 2 degrees of freedom
        p_value = chi2.sf(statistic, df=2)
        return statistic, p_value

    @staticmethod
    def zip_cdf(x, pi, lambda_):
        """

        Parameters
        ----------
        x
        pi
        lambda_

        Returns
        -------

        """
        assert x >= 0, "zip_cdf, x should > 0 (x=%f)" % x
        p = pi + (1 - pi) * poisson.cdf(x, lambda_)
        if p > 1:
            p = 1
        elif p < 0:
            p = 0
        return p

    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        from scipy.stats import poisson
        return poisson.sf(peak_mu, le_mu)


class IQR(ABC):
    @staticmethod
    def get_outlier_threshold(data, direction=-1):
        """
        Get outlier threshold

        Parameters
        ----------
        data :
        direction : int
            Which direction of outlier threshold to be returned. 1 for upper, 0 for both, -1 for lower
        Returns
        -------
        lower_bound : int or None
            Lower threshold for outlier detection
        upper_bound : int or None
            Upper threshold for outlier detection
        """
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        scale = 1.
        if direction == 1:
            return None, q3 + scale * iqr
        elif direction == -1:
            return q1 - scale * iqr, None
        else:
            return q1 - scale * iqr, q3 + scale * iqr

    @staticmethod
    def atom_ler(bed_handler, chromosome, query_start, query_end, queried_peaks,
                 small_window_threshold=4, peak_in_bg_threshold=1, offset=0):
        """
        Atom operation for local env refinement

        Parameters
        ----------
        bed_handler : pysam.TabixFile
            pysam.TabixFile handler to the bed file which stores information about all peaks
        chromosome : str
            Name of the chromosome / contig
        query_start : int
            0-based start coordinate
        query_end : int
            0-based end coordinate
        queried_peaks :

        small_window_threshold : int
            Candidate peaks with lengths shorter than this value will be skipped
        peak_in_bg_threshold : float
            Candidate peaks with density higher than this value will be removed from the local environment
        offset : int


        Returns
        -------
        se_coords : list of tuples
            List of peak coordinates (relative, tuple)
        re_coords : list of tuples
            List of peak coordinates (absolute, tuple)
        densities : List of floats
            List of peak densities
        """
        se_coords = []
        re_coords = []
        densities = []

        query_start = query_start if query_start >= 0 else 0
        for sub_peak in bed_handler.fetch(chromosome, query_start, query_end, parser=pysam.asTuple()):
            sp_start = int(sub_peak[1])
            sp_start = sp_start if sp_start >= 0 else 0
            sp_end = int(sub_peak[2])
            k = "{l}-{d}".format(l=sp_end - sp_start, d=sub_peak[4])
            if sp_start < query_start:
                sp_start = query_start
            if sp_end > query_end:
                sp_end = query_end
            # if sp_end - sp_start < small_window_threshold:
            #     continue
            peak_dens = float(sub_peak[4])
            queried_peaks[k] += 1

            if peak_dens >= peak_in_bg_threshold:
                a = sp_start - query_start
                b = sp_end - query_start
                se_coords.append((a + offset, b + offset))
                re_coords.append((sp_start, sp_end))
                densities.append(peak_dens)

        return se_coords, re_coords, densities

    @staticmethod
    @abstractmethod
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left,
                                  query_end_left, query_start_right, query_end_right,
                                  small_window_threshold, peak_in_bg_threshold,
                                  coverage_info, fdr_target, cache, disable_ler=False,
                                  enable_eler=True, peak_threshold=None):
        """
        LER-based local environment refinement

        Parameters
        ----------
        stat_tester :

        bed_handler : pysam.TabixFile
            pysam.TabixFile handler to the bed file which stores information about all peaks
        chromosome : str
            Name of the chromosome / contig
        query_start_left : int
            0-based start coordinate for left-side local environment
        query_end_left : int
            0-based end coordinate for left-side local environment
        query_start_right : int
            0-based start coordinate for right-side local environment
        query_end_right : int
            0-based end coordinate for right-side local environment
        small_window_threshold : int
            Candidate peaks with lengths shorter than this value will be skipped
        peak_in_bg_threshold : float
            Candidate peaks with density higher than this value will be removed from the local environment
        coverage_info : np.array
            array which stores coverage info
        fdr_target : float
            fdr target
        cache : dict
            Cache for LER
        disable_ler : bool
            Set it to True to disable LER
        enable_eler : bool
            Set it to False to disable empirical LER
        peak_threshold : None or numeric
            Only applicable to pkIQR, the `min(outlier_t, peak_threshold)`
            will be used as the final `outlier_t`

        Returns
        -------
        local_env : array-like
            Corrected local environment
        n_real_peaks_corrected : int
            Number of real peaks detected in local environment
        """
        pass


class bgIQR(IQR):
    @staticmethod
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left,
                                  query_end_left, query_start_right, query_end_right,
                                  small_window_threshold, peak_in_bg_threshold,
                                  coverage_info, fdr_target, cache, disable_ler=False,
                                  enable_eler=True, peak_threshold=None):
        ler_count = 0
        bg_mus = []
        local_env_left = coverage_info[query_start_left:query_end_left]
        local_env_right = coverage_info[query_start_right:query_end_right]
        local_cache = defaultdict(int)
        se_l, re_l, dens_l = bgIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome,
                                            query_start=query_start_left, query_end=query_end_left,
                                            queried_peaks=local_cache, small_window_threshold=small_window_threshold,
                                            peak_in_bg_threshold=peak_in_bg_threshold)
        se_r, re_r, dens_r = bgIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome,
                                            query_start=query_start_right, query_end=query_end_right,
                                            queried_peaks=local_cache, small_window_threshold=small_window_threshold,
                                            peak_in_bg_threshold=peak_in_bg_threshold)
        coord_offset = len(local_env_left)
        new_local_env = np.concatenate((local_env_left, local_env_right), axis=None)

        if disable_ler:
            return new_local_env, ler_count

        uncertain_se_l = []
        uncertain_re_l = []
        uncertain_se_r = []
        uncertain_re_r = []
        all_dens = []

        for k, (b, s) in enumerate(re_l):
            cache_key = __CACHE_KEY_FORMAT__ % (b, s)
            if cache_key in cache:
                if cache[cache_key] == 1:
                    new_local_env[se_l[k][0]:se_l[k][1]] = -1
                    ler_count += 1
            else:
                uncertain_re_l.append(re_l[k])
                uncertain_se_l.append(se_l[k])
                all_dens.append(dens_l[k])
        for k, (b, s) in enumerate(re_r):
            cache_key = __CACHE_KEY_FORMAT__ % (b, s)
            if cache_key in cache:
                if cache[cache_key] == 1:
                    new_local_env[se_r[k][0] + coord_offset:se_r[k][1] + coord_offset] = -1
                    ler_count += 1
            else:
                uncertain_re_r.append(re_r[k])
                uncertain_se_r.append(se_r[k])
                all_dens.append(dens_r[k])

        se_coords = []
        se_coords.extend(uncertain_se_l)
        se_coords.extend([(b + coord_offset, s + coord_offset) for b, s in uncertain_se_r])

        re_coords = []
        re_coords.extend(uncertain_re_l)
        re_coords.extend(uncertain_re_r)

        n_candidate = len(uncertain_se_l) + len(uncertain_se_r)
        if n_candidate > 3:
            for b, s in uncertain_se_l:
                le = np.copy(local_env_left)
                le[b:s] = -1
                background_window = np.concatenate((le, local_env_right), axis=None)

                mu_, var_, pi_, llc_, _ = stat_tester.fit(background_window[background_window >= 0])
                bg_mus.append(mu_)

            for b, s in uncertain_se_r:
                le = np.copy(local_env_right)
                le[b:s] = -1
                background_window = np.concatenate((le, local_env_left), axis=None)

                mu_, var_, pi_, llc_, _ = stat_tester.fit(background_window[background_window >= 0])
                bg_mus.append(mu_)

            outlier_t, _ = bgIQR.get_outlier_threshold(bg_mus)
            for k, v in enumerate(bg_mus):
                cache_key = __CACHE_KEY_FORMAT__ % (re_coords[k][0], re_coords[k][1])
                if v < outlier_t or (
                        enable_eler and all_dens[k] > peak_threshold):
                    ler_count += 1
                    new_local_env[se_coords[k][0]:se_coords[k][1]] = -1
                    cache[cache_key] = 1
                else:
                    cache[cache_key] = 0
            return new_local_env[new_local_env >= 0], ler_count
        elif n_candidate > 0:
            uncertain_re_l.extend(uncertain_re_r)

            for k, (b, s) in enumerate(uncertain_re_l):
                cache_key = __CACHE_KEY_FORMAT__ % (re_coords[k][0], re_coords[k][1])
                background_window = np.concatenate((coverage_info[b - 2500:b],
                                                    coverage_info[s:s + 2500]),
                                                   axis=None)
                # mask original peak in question
                # this is important to assays which have clean bg/sequenced shallowly
                if query_start_right < b:
                    rs = query_end_left - (b - 2500)
                    re = query_start_right - (b - 2500)
                    background_window[rs:re] = -1
                elif s < query_end_left:
                    rs = query_end_left - (s - 2500)
                    re = query_start_right - (s - 2500)
                    background_window[rs:re] = -1
                background_window = background_window[background_window >= 0]
                mu_pk, var_pk, pi_pk, llc_pk, _ = stat_tester.fit(coverage_info[b:s])
                mu_bg, var_bg, pi_bg, llc_bg, _ = stat_tester.fit(background_window)

                p_val_formal = stat_tester.sf(mu_pk, var_pk, pi_pk, mu_bg, var_bg, pi_bg)
                if p_val_formal < fdr_target or (
                        enable_eler and mu_pk > peak_threshold):
                    new_local_env[se_coords[k][0]:se_coords[k][1]] = -1
                    ler_count += 1
                    cache[cache_key] = 1
                else:
                    cache[cache_key] = 0

            return new_local_env[new_local_env >= 0], ler_count
        else:
            return new_local_env[new_local_env >= 0], ler_count


class pkIQR(IQR):
    @staticmethod
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left, 
                                  query_end_left, query_start_right, query_end_right, 
                                  small_window_threshold, peak_in_bg_threshold, 
                                  coverage_info, fdr_target, cache, disable_ler=False, 
                                  enable_eler=True, peak_threshold=5.):
        ler_count = 0
        local_env_left = coverage_info[query_start_left:query_end_left]
        local_env_right = coverage_info[query_start_right:query_end_right]
        coord_offset = len(local_env_left)
        local_cache = defaultdict(int)
        se_l, re_l, dens_l = pkIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome,
                                            query_start=query_start_left,
                                            query_end=query_end_left, queried_peaks=local_cache,
                                            small_window_threshold=small_window_threshold,
                                            peak_in_bg_threshold=peak_in_bg_threshold)
        se_r, re_r, dens_r = pkIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome,
                                            query_start=query_start_right,
                                            query_end=query_end_right, queried_peaks=local_cache,
                                            small_window_threshold=small_window_threshold,
                                            peak_in_bg_threshold=peak_in_bg_threshold, offset=coord_offset)

        new_local_env = np.concatenate((local_env_left, local_env_right), axis=None)

        if disable_ler:
            return new_local_env, ler_count

        all_dens = dens_l + dens_r
        if len(all_dens) >= 3:
            _, outlier_t = pkIQR.get_outlier_threshold(all_dens, 1)
            if outlier_t > peak_threshold:
                outlier_t = peak_threshold
            for i, se in enumerate(se_l):
                if dens_l[i] >= outlier_t:
                    new_local_env[se[0]:se[1]] = -1
                    ler_count += 1
            for i, se in enumerate(se_r):
                if dens_r[i] >= outlier_t:
                    new_local_env[se[0]:se[1]] = -1
                    ler_count += 1
        elif len(all_dens) > 0:
            re_l.extend(re_r)
            se_l.extend(se_r)
            for k, (b, s) in enumerate(re_l):
                background_window = np.concatenate((coverage_info[b - 2500:b], coverage_info[s:s + 2500]),
                                                   axis=None)
                # mask original peak in question
                # this is important to assays which have clean bg/sequenced shallowly
                if query_start_right < b:
                    rs = query_end_left - (b - 2500)
                    re = query_start_right - (b - 2500)
                    background_window[rs:re] = -1
                elif s < query_end_left:
                    rs = query_end_left - (s - 2500)
                    re = query_start_right - (s - 2500)
                    background_window[rs:re] = -1
                background_window = background_window[background_window >= 0]
                mu_pk, var_pk, pi_pk, llc_pk, _ = stat_tester.fit(coverage_info[b:s])
                mu_bg, var_bg, pi_bg, llc_bg, _ = stat_tester.fit(background_window)

                p_val_formal = stat_tester.sf(mu_pk, var_pk, pi_pk, mu_bg, var_bg, pi_bg)
                if p_val_formal < fdr_target or (
                        enable_eler and mu_pk > peak_threshold):
                    new_local_env[se_l[k][0]:se_l[k][1]] = -1
                    ler_count += 1
        return new_local_env[new_local_env >= 0], ler_count


def independent_filtering(df, fdr_target=0.1, output_to=None, logger=None, **kwargs):
    """
    Independent filtering

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of peak candidates (bed-like format)
    fdr_target : float
        FDR target. Default 0.1.
    output_to : str or None, optional
        Path for the output
    logger : None or a logger
        A logger to write logs
    **kwargs :
        Keyword arguments, adjust_method (str, default fdr_bh)
                           and output_diagnostics (bool, default True) is effective
    Returns
    -------
    final_df : pd.DataFrame
        DataFrame with adjusted p-values

    Refs
    ----
    Bourgon, R., Gentleman, R. & Huber, W.
         Independent filtering increases detection power for high-throughput experiments.
         PNAS 107, 9546–9551 (2010).
    """
    from statsmodels.stats.multitest import multipletests
    from pints.io_engine import log_assert
    import pandas as pd

    adjust_method = kwargs.get("adjust_method", "fdr_bh")
    log_assert(isinstance(df, pd.DataFrame), "df needs to be pd.DataFrame", logger)
    log_assert("reads" in df.columns, "Please provide read counts (reads) in df", logger)
    quantiles_tested = []
    windows_remain = []
    windows_rejected = []
    adjusted_df = None
    select_probe = None
    read_counts_threshold = 0
    try:
        direct_padj = multipletests(df["pval"], alpha=fdr_target, method=adjust_method)[1]
    except ZeroDivisionError:
        direct_padj = df["pval"] * df.shape[0]
    # Every quantile threshold defines a subset; identical thresholds give identical subsets, so the
    # multiple-testing correction is computed once per distinct threshold. Only the selected subset is
    # materialised as a DataFrame (same construction as before, so the result is identical).
    def _materialise(probe, read_out):
        filtered = df.loc[probe, :].copy()
        filtered["padj"] = read_out
        filtered["name"] = "."
        return filtered.loc[:, ("chromosome", "start", "end", "name", "padj", "strand", "reads")]

    reads_col = df["reads"]
    pval_col = df["pval"]
    per_threshold = {}
    selected = None
    for quantile in np.arange(0, 1, 0.005):
        threshold = reads_col.quantile(quantile)
        hit = per_threshold.get(threshold) if threshold == threshold else None
        if hit is None:
            tmp_probe = reads_col > threshold
            sub_p = pval_col.loc[tmp_probe]
            try:
                read_out = multipletests(sub_p, alpha=fdr_target, method=adjust_method)[1]
            except ZeroDivisionError:
                # if we cannot run BH, then we use Bonferroni correction
                read_out = sub_p * sub_p.shape[0]
            hit = (tmp_probe, read_out, int(sub_p.shape[0]), sum(read_out < fdr_target))
            if threshold == threshold:
                per_threshold[threshold] = hit
        tmp_probe, read_out, n_remain, n_rejected = hit
        quantiles_tested.append(quantile)
        windows_remain.append(n_remain)
        windows_rejected.append(n_rejected)

        if windows_rejected[-1] >= max(windows_rejected):
            selected = (tmp_probe, read_out)
            select_probe = tmp_probe
            read_counts_threshold = threshold
    if selected is not None:
        adjusted_df = _materialise(*selected)

    optimized_threshold = np.argmax(windows_rejected)
    logger.info("Optimized filters from independent filtering: percentile %f, rejection %d" % (
        quantiles_tested[optimized_threshold], windows_rejected[optimized_threshold]))

    final_df = df.copy()
    final_df["padj"] = direct_padj
    final_df.loc[select_probe, "padj"] = adjusted_df["padj"]
    quantiles_tested_arr = np.asarray(quantiles_tested)
    windows_rejected_arr = np.asarray(windows_rejected)
    if kwargs.get("output_diagnostics", False) and output_to is not None:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(5.5, 5.5))
        ax2 = ax1.twinx()
        ax1.scatter(quantiles_tested, windows_remain, facecolors="none", edgecolors="#0571b0",
                    label="Windows remain")
        ax2.scatter(quantiles_tested, windows_rejected, facecolors="none", edgecolors="#ca0020", label="Rejections")
        ax2.annotate("Read counts: %d" % read_counts_threshold,
                     xy=(quantiles_tested_arr[np.argmax(windows_rejected_arr)], np.max(windows_rejected_arr)))
        ax1.set_xlabel("Quantile of filter")
        ax1.set_ylabel("Windows remain", color="#0571b0")
        ax1.tick_params(axis='y', labelcolor="#0571b0")
        ax2.set_ylabel("Rejections", color="#ca0020")
        ax2.tick_params(axis='y', labelcolor="#ca0020")
        plt.tight_layout()
        plt.savefig(output_to, transparent=True, bbox_inches="tight")
        plt.close()
        logger.info("Diagnostic plot for independent filtering was written to %s" % output_to)
    return final_df


def get_elbow(X, Y):
    """
    Find the elbow points by finding the point that's the furthest
    from the extreme line, which passes the two boundary points (edge points)
    ref: DOI: 10.1007/BF01195985

    Parameters
    ----------
    X : np.array

    Y : np.array

    Returns
    -------
    elbow_x : float or np.nan
        X coordinate of the elbow/knee point. If the alg fails to find the point, it returns np.nan
    elbow_y : float or np.nan
        Y coordinate of the elbow/knee point. If the alg fails to find the point, it returns np.nan
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape[0] < 3 or X.shape[0] != Y.shape[0]:
        return np.nan, np.nan

    sort_index = X.argsort()
    X_s = X[sort_index]
    Y_s = Y[sort_index]

    p1 = np.array([X_s[0], Y_s[0]])
    p2 = np.array([X_s[-1], Y_s[-1]])
    P3 = np.column_stack((X_s, Y_s))

    # z component of the cross product (np.cross no longer takes 2-D vectors since NumPy 2.5); same arithmetic
    d1, d3 = p2 - p1, P3 - p1
    D = np.abs((d1[0] * d3[:, 1] - d1[1] * d3[:, 0]) / np.linalg.norm(d1))
    tmp = np.where(D == np.nanmax(D))[0]
    if len(tmp) > 0:
        midx = tmp[-1]
        X_e = X_s[midx]
        Y_e = Y_s[midx]
    else:
        X_e = np.nan
        Y_e = np.nan
    return X_e, Y_e
