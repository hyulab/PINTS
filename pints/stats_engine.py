#!/usr/bin/env python
# coding=utf-8
# Created by: Li Yao (ly349@cornell.edu)
# Created on: 2019-08-07
#
# PINTS: Peak Identifier for Nascent Transcripts Sequencing
# Copyright (C) 2019 Li Yao at the Yu Lab
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
import warnings

try:
    import pysam
    import numpy as np
    from scipy.stats import poisson, binom_test, fisher_exact, nbinom
    from scipy.special import gamma, digamma, polygamma, gammaln, psi, factorial
    from scipy.optimize import fmin_l_bfgs_b as BFGS
    from scipy.optimize import newton
    from abc import ABC, abstractmethod
    from collections import defaultdict
except ImportError as e:
    missing_package = str(e).replace("No module named '", "").replace("'", "")
    sys.exit("Please install %s first!" % missing_package)


def prop_test(pi_0, l_0, pi_1, l_1, empirical_threshold=5, alternative="greater"):
    """
    survivors = np.array([[1781, total1 - 1781], [1443, total2 - 47]])
    proportions_ztest
    In the two sample test, smaller means that the alternative hypothesis is
    p1 < p2 and larger means p1 > p2 where p1 is the proportion of the first sample and p2 of the second one
    :return:
    """
    count_0 = int(pi_0 * l_0)
    total_0 = l_0
    count_1 = int(pi_1 * l_1)
    total_1 = l_1
    # if no zero
    if count_0 == 0 or count_1 == 0:
        return 10e-16
    try:
        while count_0 < empirical_threshold:
            count_0 = int(pi_0 * l_0 * 10)
            total_0 *= 10
        while count_1 < empirical_threshold:
            count_1 = int(pi_1 * l_1 * 10)
            total_1 *= 10
        _, pval = fisher_exact(np.array([[count_0, count_1], [total_0 - count_0, total_1 - count_1]]),
                               alternative=alternative)
    except Exception as e:
        print(e, count_0, total_0, count_1, total_1)
        pval = 1
    return pval


def get_rank(input_array):
    """
    Get the rank of an array

    Parameters
    ----------
    input_array : np.Array

    Returns
    ranks : np.Array
        Rank of the element in input_array
    -------

    """
    temp = input_array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(input_array))
    return ranks


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
        if self.init_mu is None or self.init_pi is None:
            init_lamb, init_pi = ZIP.zip_moment_estimators(windows=window)
        else:
            init_lamb = self.init_mu
            # init_pi = self.init_pi
            init_pi = 0.8
        lamb = init_lamb
        pi = init_pi
        n_iter = 0
        hat_z = np.zeros(len(window))
        zero_elements = window == 0
        smallest = np.finfo(float).eps

        if zero_elements.sum() == 0:
            m = window.mean()
            return m, m, 0, np.nan, True

        I = len(window)
        prev_likelihood = 0.
        likelihoods = []
        u_ele, c_ele = np.unique(window, return_counts=True)
        while True:
            # expectation
            hat_z[zero_elements] = pi / (pi + np.exp(-lamb) * (1 - pi))
            # maximization
            pi = hat_z.sum() / I
            indicator = 1 - hat_z
            lamb = (indicator * window).sum() / indicator.sum()
            # estimate likelihood
            u_pmf = (1 - pi) * poisson.pmf(u_ele, lamb)
            u_pmf[np.where(u_ele == 0)] += pi
            u_pmf[u_pmf < smallest] = smallest
            likelihood = (np.log(u_pmf) * c_ele).sum()
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
        assert x >= 0, "zip_cdf, x should > 0"
        p = pi + (1 - pi) * poisson.cdf(x, lambda_)
        if p > 1:
            p = 1
        elif p < 0:
            p = 0
        return p

    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        # return 1 - ZIP.zip_cdf(peak_mu, peak_pi, le_mu)
        from scipy.stats import poisson
        return poisson.sf(peak_mu, le_mu)


class NegativeBinomial(StatModel):
    def fit(self, window):
        """
        EM for Negative Binomial

        Parameters
        ----------
        window

        Returns
        -------
        r : float

        p : float

        pi : None

        llc : None

        convergence : bool

        """

        def log_likelihood(params, *args):
            r, p = params
            X = args[0]
            N = X.size

            # MLE estimate based on the formula on Wikipedia:
            # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
            result = np.sum(gammaln(X + r)) \
                     - np.sum(np.log(factorial(X))) \
                     - N * (gammaln(r)) \
                     + N * r * np.log(p) \
                     + np.sum(X * np.log(1 - (p if p < 1 else 1 - self.infinitesimal)))

            return -result

        if self.init_mu is None or self.init_variance is None:
            # reasonable initial values (from fitdistr function in R)
            m = np.mean(window)
            v = np.var(window)
            size = (m ** 2) / (v - m) if v > m else 10

            # convert mu/size parameterization to prob/size
            p0 = size / ((size + m) if size + m != 0 else 1)
            r0 = size
            # initial_params = np.array([r0, p0])
        else:
            size = self.init_mu ** 2 / (self.init_variance - self.init_mu) if self.init_variance > self.init_mu else 10
            p0 = size / ((size + self.init_mu) if size + self.init_mu != 0 else 1)
            r0 = size
        initial_params = np.array([r0, p0])
        try:
            bounds = [(self.infinitesimal, None), (self.infinitesimal, 1)]
            optimres = BFGS(log_likelihood,
                            x0=initial_params,
                            # fprime=log_likelihood_deriv,
                            args=(window,),
                            approx_grad=1,
                            bounds=bounds)

            params = optimres[0]
            convergent_flag = True if optimres[2]["warnflag"] == 0 else False
        except:
            # print("Failed to converge.")
            return r0, p0, None, None, False
        return params[0], params[1], None, None, convergent_flag

    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        return nbinom.sf(peak_mu * (1 - peak_var) / peak_var, le_mu, le_var)


class ZINB(StatModel):
    @staticmethod
    def fit_nb(X, Z, pi, initial_params=None):
        infinitesimal = np.finfo(float).eps

        def log_likelihood(params, *args):
            mu, k = params
            X = args[0]
            Z = args[1]
            pi = args[2]
            muk = mu + k
            muk = muk if muk > 0 else infinitesimal
            pi = pi if pi > 0 else infinitesimal
            k = k if k > 0 else infinitesimal
            mu = mu if mu > 0 else infinitesimal
            one_minus_pi = 1 - pi
            one_minus_pi = one_minus_pi if one_minus_pi > 0 else infinitesimal
            # log_muk = np.log(muk if muk > 0 else infinitesimal)
            # N = len(X)
            is_zeros = X == 0
            is_zeros = is_zeros.astype(int)
            # incomplete log-likelihood function
            """
            result = np.sum(is_zeros * np.log(pi + (1 - pi) * ((k / (muk)) ** k))) + np.sum((1 - is_zeros) * (
                    np.log(1 - pi) + gammaln(X + k) - gammaln(X + 1) - gammaln(k) + k * np.log(k) - k * np.log(
                muk) + X * np.log(mu if mu > 0 else infinitesimal) - X * np.log(muk)))
            """
            # complete log-likelihood function
            result = 0
            try:
                result = np.sum(Z * np.log(pi)) + np.sum((1 - Z) * (
                        np.log(one_minus_pi) + gammaln(X + k) - gammaln(X + 1) - gammaln(k) + k * np.log(
                    k) - k * np.log(
                    muk) + X * np.log(mu) - X * np.log(muk)))
            except Exception:
                print("pi", pi, np.log(pi))
                print("1-pi", np.log(1 - pi))
                print(gammaln(k))
                print(np.log(k))
                print(np.log(mu))
                print(pi, mu, k, muk)
            return -result

        if initial_params is None:
            # reasonable initial values (from fitdistr function in R)
            m = np.mean(X)
            v = np.var(X)
            size = (m ** 2) / (v - m) if v > m else 10

            # convert mu/size parameters to mu/k
            p_0 = size / ((size + m) if size + m != 0 else 1)
            mu_0 = size * (1 - p_0) / p_0
            k_0 = 1 / size
            initial_params = np.array([mu_0, k_0])

        bounds = [(infinitesimal, None), (infinitesimal, None)]
        optimres = BFGS(log_likelihood,
                        x0=initial_params,
                        # fprime=log_likelihood_deriv,
                        args=(X, Z, pi),
                        approx_grad=1,
                        bounds=bounds)
        params = optimres[0]
        return params[0], params[1]

    def fit(self, window):
        infinitesimal = np.finfo(float).eps
        if self.init_mu is None or self.init_k is None or self.init_pi is None:
            not_zero = window != 0
            nn_zero = sum(not_zero)
            init_pi = (window.shape[0] - not_zero.sum()) / window.shape[0]
            if nn_zero > 0:
                init_mu = np.mean(window[not_zero])
                s2 = np.var(window[not_zero])
                size = init_mu ** 2 / (s2 - init_mu + 0.0001)
                size = size if size > 0 else 0.0001
                init_k = 1 / size
            else:
                init_mu = 0
                init_k = 1
        else:
            init_mu = self.init_mu
            init_k = self.init_k
            init_pi = self.init_pi
        mu = init_mu
        k = init_k
        pi = init_pi
        mu_pre = mu
        k_pre = k
        pi_pre = pi
        n_iter = 0
        hat_z = np.zeros(len(window))
        zero_elements = window == 0
        n = len(window)
        prev_likelihood = 0.
        likelihoods = []

        while True:
            # expectation
            # in case of overflow
            # nb_term = k_pre / (mu_pre + k_pre)
            # if nb_term < 1 and k_pre >
            hat_z[zero_elements] = pi_pre / (pi_pre + (1 - pi_pre) * ((k_pre / (mu_pre + k_pre)) ** k_pre))
            # maximization
            pi = hat_z.sum() / n
            # mu & k
            mu, k = ZINB.fit_nb(window, hat_z, pi_pre, [mu_pre, k_pre])
            # estimate likelihood
            pos_pmf = nbinom.pmf(window, 1 / k, 1 / (1 + k * mu))
            pos_pmf[pos_pmf == 0] = infinitesimal
            likelihood = np.sum(zero_elements * np.log(pi if pi > 0 else infinitesimal)) + np.log(pos_pmf).sum()
            likelihoods.append(likelihood)

            if n_iter > 0:
                if abs(likelihood - prev_likelihood) < self.stop_diff or abs(mu - mu_pre) < self.stop_diff or abs(
                        k - k_pre) < self.stop_diff or abs(pi - pi_pre) < self.stop_diff:
                    if self.debug and self.output_to != "":
                        import matplotlib.pyplot as plt
                        plt.plot(likelihoods)
                        plt.ylabel('Log-likelihood')
                        plt.xlabel('Iteration')
                        plt.tight_layout()
                        plt.savefig(self.output_to, bbox_inches="tight", transparent=True)
                    return mu, k, pi, likelihood, True
                if n_iter > self.max_iter:
                    return init_mu, init_k, init_pi, likelihood, False
            prev_likelihood = likelihood
            mu_pre = mu
            k_pre = k
            pi_pre = pi
            n_iter += 1

    def sf(self, peak_mu, peak_var, peak_pi, le_mu, le_var, le_pi):
        pass


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
            if sp_end - sp_start < small_window_threshold:
                continue
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
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left, query_end_left,
                                  query_start_right, query_end_right, small_window_threshold, peak_in_bg_threshold,
                                  coverage_info, fdr_target, cache, disable_ler=False):
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
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left, query_end_left,
                                  query_start_right, query_end_right, small_window_threshold, peak_in_bg_threshold,
                                  coverage_info, fdr_target, cache, disable_ler=False):
        ler_count = 0
        bg_mus = []
        local_env_left = coverage_info[query_start_left:query_end_left]
        local_env_right = coverage_info[query_start_right:query_end_right]
        local_cache = defaultdict(int)
        se_l, re_l, _ = bgIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome, query_start=query_start_left,
                                       query_end=query_end_left, queried_peaks=local_cache,
                                       small_window_threshold=small_window_threshold,
                                       peak_in_bg_threshold=peak_in_bg_threshold)
        se_r, re_r, _ = bgIQR.atom_ler(bed_handler=bed_handler, chromosome=chromosome, query_start=query_start_right,
                                       query_end=query_end_right, queried_peaks=local_cache,
                                       small_window_threshold=small_window_threshold,
                                       peak_in_bg_threshold=peak_in_bg_threshold)
        coord_offset = len(local_env_left)
        new_local_env = np.concatenate((local_env_left, local_env_right), axis=None)

        if disable_ler:
            return new_local_env, ler_count

        uncertain_se_l = []
        uncertain_re_l = []
        uncertain_se_r = []
        uncertain_re_r = []

        for k, (b, s) in enumerate(re_l):
            cache_key = "%d-%d" % (b, s)
            if cache_key in cache:
                if cache[cache_key] == 1:
                    new_local_env[se_l[k][0]:se_l[k][1]] = -1
                    ler_count += 1
            else:
                uncertain_re_l.append(re_l[k])
                uncertain_se_l.append(se_l[k])
        for k, (b, s) in enumerate(re_r):
            cache_key = "%d-%d" % (b, s)
            if cache_key in cache:
                if cache[cache_key] == 1:
                    new_local_env[se_r[k][0] + coord_offset:se_r[k][1] + coord_offset] = -1
                    ler_count += 1
            else:
                uncertain_re_r.append(re_r[k])
                uncertain_se_r.append(se_r[k])

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
                cache_key = "%d-%d" % (re_coords[k][0], re_coords[k][1])
                if v < outlier_t:
                    ler_count += 1
                    new_local_env[se_coords[k][0]:se_coords[k][1]] = -1
                    cache[cache_key] = 1
                else:
                    cache[cache_key] = 0
            return new_local_env[new_local_env >= 0], ler_count
        elif n_candidate > 0:
            uncertain_re_l.extend(uncertain_re_r)

            for k, (b, s) in enumerate(uncertain_re_l):
                cache_key = "%d-%d" % (re_coords[k][0], re_coords[k][1])
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
                if p_val_formal < fdr_target:
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
    def remove_peaks_in_local_env(stat_tester, bed_handler, chromosome, query_start_left, query_end_left,
                                  query_start_right, query_end_right, small_window_threshold, peak_in_bg_threshold,
                                  coverage_info, fdr_target, cache, disable_ler=False):
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
        all_dens = dens_l + dens_r
        if len(all_dens) >= 3:
            _, outlier_t = pkIQR.get_outlier_threshold(all_dens, 1)
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
                if p_val_formal < fdr_target:
                    new_local_env[se_l[k][0]:se_l[k][1]] = -1
                    ler_count += 1
        return new_local_env[new_local_env >= 0], ler_count


def independent_filtering(df, output_to=None, logger=None, **kwargs):
    """
    Independent filtering

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of peak candidates (bed-like format)
    output_to : str or None, optional
        Path for the output
    logger : None or a logger
        A logger to write logs
    **kwargs :
        Keyword arguments, fdr_target (float, default 0.1),
                           adjust_method (str, default fdr_bh),
                           ind_filter_granularity (float, default 0.005)
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

    log_assert(isinstance(df, pd.DataFrame), "df needs to be pd.DataFrame", logger)
    log_assert("reads" in df.columns, "Please provide read counts (reads) in df", logger)
    quantiles_tested = []
    windows_remain = []
    windows_rejected = []
    adjusted_df = None
    select_probe = None
    read_counts_threshold = 0
    for quantile in np.arange(0, 1, kwargs["ind_filter_granularity"]):
        threshold = df.reads.quantile(quantile)
        tmp_probe = df["reads"] > threshold
        filtered_df = df.loc[tmp_probe, :].copy()
        try:
            mult_res = multipletests(filtered_df["pval"], alpha=kwargs["fdr_target"], method=kwargs["adjust_method"])
            read_out = mult_res[1]
        except ZeroDivisionError:
            # if we cannot run BH, then we use Bonferroni correction
            read_out = filtered_df["pval"] * filtered_df.shape[0]
        filtered_df["padj"] = read_out
        # filtered_df.drop(columns="pval", inplace=True)
        filtered_df["name"] = "."
        filtered_df = filtered_df.loc[:, ("chromosome", "start", "end", "name", "padj", "strand", "reads")]
        quantiles_tested.append(quantile)
        windows_remain.append(filtered_df.shape[0])
        windows_rejected.append(sum(read_out < kwargs["fdr_target"]))

        if len(windows_rejected) > 0:
            if windows_rejected[-1] >= max(windows_rejected):
                adjusted_df = filtered_df
                select_probe = tmp_probe
                read_counts_threshold = threshold
        else:
            adjusted_df = filtered_df
            select_probe = tmp_probe
            read_counts_threshold = threshold

    optimized_threshold = np.argmax(windows_rejected)
    logger.info("Optimized filters from independent filtering: percentile %f, rejection %d" % (
        quantiles_tested[optimized_threshold], windows_rejected[optimized_threshold]))

    final_df = df.copy()
    final_df["padj"] = final_df["pval"]
    final_df.loc[select_probe, "padj"] = adjusted_df["padj"]
    quantiles_tested_arr = np.asarray(quantiles_tested)
    windows_rejected_arr = np.asarray(windows_rejected)
    if kwargs["output_diagnostics"] and output_to is not None:
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


if __name__ == "__main__":
    warnings.filterwarnings("error")
    np.random.seed(299)
    n = 1000
    theta = 2.5  # Poisson rate
    pi = 0.55  # probability of extra-zeros (pi = 1-psi)
    mu = 4.48
    k = 0.25

    # Simulate some data
    z = ZIP(debug=True)
    counts = np.array([(np.random.random() > pi) *
                       np.random.poisson(theta) for i in range(n)])
    print(z.fit(counts))
