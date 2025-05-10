#!/usr/bin/env python
# coding=utf-8

#  PINTS: Peak Identifier for Nascent Transcript Starts
#  Copyright (c) 2019-2025 Li Yao at the Yu Lab.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

import datetime
import gzip
import logging
import os
import sys
from collections import namedtuple
from glob import glob
from multiprocessing import Pool

logging.basicConfig(format="%(name)s - %(asctime)s - %(levelname)s: %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S",
                    level=logging.INFO,
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("PINTS - Caller")

try:
    import numpy as np
    import pandas as pd
    import pysam
    import pyBigWig
    import pybedtools
    from scipy.stats import poisson
    from scipy.signal import find_peaks, peak_widths
    from scipy.ndimage import gaussian_filter1d
    from pybedtools import BedTool
    from .stats_engine import Poisson, ZIP, pval_dist, bgIQR, pkIQR, \
        independent_filtering, get_elbow
    from .io_engine import get_read_signal, get_coverage_bw, log_assert, normalize_using_input, \
        index_bed_file, peak_bed_to_gtf, merge_replicates_bw
    from pints import __version__
except ImportError as e:
    missing_package = str(e).replace("No module named '", "").replace("'", "")
    logger.error("Please install %s first!" % missing_package)
    sys.exit(-1)

housekeeping_files = []
COMMON_HEADER = ('chromosome', 'start', 'end', 'name', 'padj', 'strand', 'reads',
                 'pval', 'mu_0', 'pi_0', 'mu_1', 'pi_1', 'var_1', 'ler_1', 'ler_2', 'ler_3',
                 'non_zeros', 'summit', 'summit_val')
__SUB_PEAK_TPL__ = "_subpeaks_%s.bed"
__SIG_PEAK_TPL__ = "%s_sig_%s.bed"
__STRICT_QC__ = False
stat_tester = None
iqr_obj = None


def check_update():
    """
    Check for updates

    Returns
    -------

    """
    logger.info("Checking for updates...")
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry
    import time
    requests.packages.urllib3.disable_warnings()
    retry_strategy = Retry(
        total=3,
        backoff_factor=3
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    with requests.Session() as http:
        http.mount("https://", adapter)

        try:
            req = http.get(url="https://pypi.org/pypi/pyPINTS/json", timeout=10)
            rj = req.json()
            remote_version = rj["info"]["version"]
            import pints
            if pints.__version__ == remote_version:
                logger.info("You are using the latest version of PINTS")
            else:
                logger.warning("!!!Your PINTS version is out-dated ({cv} vs. {rv})!!!".format(cv=pints.__version__,
                                                                                              rv=remote_version))
                logger.warning("Please run `pip install -U pyPINTS` to update!")
                time.sleep(10)
        except Exception as ex:
            logger.error(ex)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handler for exception

    Parameters
    ----------
    exc_type :
    exc_value :
    exc_traceback :

    Returns
    -------

    Refs
    ----
    https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python/16993115#16993115
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def run_command(cmd, repress_log=False):
    """
    Run command

    Parameters
    ----------
    cmd : str
        command
    repress_log : bool
        When it's set to False, if the command failed, the log will not be wrote to logger.

    Returns
    -------
    stdout : str
        Stdout output
    stderr : str
        Stderr output for the child process
    return_code : int
        Exit status of the child process
    """
    from subprocess import Popen, PIPE
    with Popen(cmd, shell=True, stderr=PIPE, stdout=PIPE) as p:
        stdout, stderr = p.communicate()
    stderr = stderr.decode("utf-8")
    stdout = stdout.decode("utf-8")
    if not repress_log and p.returncode != 0:
        logger.error("Failed to run command %s" % cmd)
    return stdout, stderr, p.returncode


def runtime_check(check_updates=True):
    """
    Runtime check, make sure all dependent tools are callable

    Parameters
    ----------
    check_updates : bool
        Set this to True to check whether the PINTS instance is up-to-date

    Returns
    -------

    """
    import shutil
    if sys.platform == "win32":
        logger.warning("No test had performed on Windows, so it might be buggy.")
    dependent_tools = ("bgzip", "tabix", "bedtools")
    for tool in dependent_tools:
        full_path = shutil.which(tool)
        if full_path is None:
            logger.error("Required tool %s is not callable" % tool)
            sys.exit(1)
    if check_updates:
        check_update()


def merge_intervals(intervals, distance=0):
    """
    Merge intervals

    Parameters
    ----------
    intervals : tuple/list
        List / tuple of interval tuples
    distance : int
        Maximum distance between features allowed for features to be merged.
        Default is 0. That is, overlapping and/or book-ended features are merged.

    Returns
    -------
    merged_intervals : list
        Tuple of merged intervals

    Refs
    ----
        https://www.geeksforgeeks.org/merging-intervals/
    """
    log_assert(distance >= 0, "distance need to be >= 0", logger)
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for t in s:
        if t[0] > s[m][1] + distance:
            m += 1
            s[m] = t[:2]
        else:
            # consider intervals
            # ((6, 8), (1, 9), (2, 4), (4, 7))
            # if we don't add an extra check
            # the final result will be (1, 8) instead of (1, 9)
            if s[m][1] <= t[1]:
                s[m] = [s[m][0], t[1]]
    return s[:m + 1]


def sliding_window(chromosome_coverage, window_size=100, step_size=100):
    """
    Generate sliding windows

    Parameters
    ----------
    chromosome_coverage : array-like
        0-based per base coverage array for a certain chromosome
    window_size : int
        Window size, by default, 100
    step_size : int
        Step size, by default, 100 (non-overlap)

    Yields
    ------
    window : int
        Read counts in this window
    start : int
        0-based start coordinate of this window
    end : int
        0-based end coordinate of this window
    """
    if step_size < 1:
        logger.error("step_size must >= 1")
        raise ValueError("step_size must >= 1")
    if len(chromosome_coverage) < 1:
        logger.error("chromosome_coverage must >= 1")
        raise ValueError("chromosome_coverage must >= 1")

    total_bins = np.floor(chromosome_coverage.shape[0] / step_size - window_size / step_size + 1).astype(
        int)
    start = 0
    end = window_size
    for _ in range(total_bins):
        window = np.sum(chromosome_coverage[start:end])
        yield window, (start, end)
        start += step_size
        end = start + window_size


def check_window(coord_start, coord_end, mu_peak, var_peak, pi_peak, chromosome_coverage, peak_in_bg_threshold,
                 mu_bkg_minimum, sp_bed_handler, chromosome_name, fdr_target, cache, small_window_threshold=5,
                 flanking=(10000, 5000, 1000), disable_ler=False, enable_eler=True, top_peak_mu=None):
    """
    Calculate p-value for a peak

    Parameters
    ----------
    coord_start : int
        0-based start coordinate
    coord_end : int
        0-based end coordinate
    mu_peak : float
        mu_mle of the peak
    var_peak : float or None
        var_mle of the peak, can be None if not evaluated
    pi_peak : float
        pi_mle of the peak
    chromosome_coverage : array-like
        0-based per base coverage array for a certain chromosome
    peak_in_bg_threshold : float
        Candidate peaks with density higher than this value will be removed from the local environment
    mu_bkg_minimum : float
        minimum mu for background
    sp_bed_handler : pysam.TabixFile
        pysam.TabixFile object for subpeak bed file
    chromosome_name : str
        name of the chromosome/contig to call peaks
    fdr_target : float
        FDR target
    cache : dict
        cache for IQR, leave it as it is
    small_window_threshold : int
        Candidate peaks with lengths shorter than this value will be skipped
    flanking : tuple
        Lengths of local environment that this function will check
    disable_ler : bool
        Disable local environment refinement, by default, False
    enable_eler : bool
        Set it to False to disable enhanced LER
    top_peak_mu : float
        Density threshold for peaks in local environments to be considered as potential true peaks

    Returns
    -------
    p_value : float
        p_value for the peak
    window_value: int
        read counts in this window
    mu_0 : float
        mu for local env
    pi_0: float
        pi for local env
    ler_counts : int
        # of local peaks masked by LER
    lrt_pval : float
        P-value from LRT test
    sig_dens : float
        Signal density
    """
    selected_window = chromosome_coverage[coord_start:coord_end]
    window_value = np.sum(selected_window)
    sig_dens = (selected_window > 0).sum() / (coord_end - coord_start)
    if coord_end - coord_start < small_window_threshold \
            or window_value == 0:
        return 1., window_value, 0., 0., (0, 0, 0), 1., sig_dens
    flanking = np.asarray(flanking, dtype=int) // 2
    chr_len = len(chromosome_coverage)
    mus = []
    variances = []
    pis = []
    ler_counts = []
    lrt_pvals = []
    for f in flanking:
        qsl = coord_start - f
        qel = coord_start - 1
        qsl = qsl if qsl >= 0 else 0
        qsr = coord_end + 1
        qer = coord_end + f
        bg, x = iqr_obj.remove_peaks_in_local_env(stat_tester=stat_tester, bed_handler=sp_bed_handler,
                                                  chromosome=chromosome_name, query_start_left=max(qsl, 0),
                                                  query_end_left=max(qel, 0), query_start_right=min(qsr, chr_len),
                                                  query_end_right=min(qer, chr_len),
                                                  small_window_threshold=small_window_threshold,
                                                  peak_in_bg_threshold=peak_in_bg_threshold,
                                                  coverage_info=chromosome_coverage,
                                                  fdr_target=fdr_target, cache=cache,
                                                  disable_ler=disable_ler, enable_eler=enable_eler,
                                                  peak_threshold=top_peak_mu)

        mu_, var_, pi_, _, _ = stat_tester.fit(bg)
        w, pval = stat_tester.lrt(selected_window, bg)
        lrt_pvals.append(pval)
        mus.append(mu_)
        variances.append(var_)
        pis.append(pi_)
        ler_counts.append(x)

    mu_0 = np.mean(mus)  # mus[index]
    var_0 = np.mean(variances)
    pi_0 = np.mean(pis)  # pis[index]
    if mu_bkg_minimum is not None and mu_0 < mu_bkg_minimum:
        mu_0 = mu_bkg_minimum

    pvalue = stat_tester.sf(mu_peak, var_peak, pi_peak, mu_0, var_0, pi_0)

    return pvalue, window_value, mu_0, pi_0, ler_counts, np.mean(lrt_pvals), sig_dens


def quasi_max_score_segment(candidates, donor_tolerance, ce_trigger, max_distance):
    """
    Max score segment algorithm to join adjacent sub peaks/seeds

    Parameters
    ----------
    candidates : list or tuple
        list of candidate peaks, each peak is also a list with 4 elements:
            start
            end
            read counts
            density
    donor_tolerance : float
        Donor tolerance in best score segments
    ce_trigger : int
        Trigger for receptor tolerance checking
    max_distance : int
        Max distance allowed to join two sub peaks/seeds

    Returns
    -------
    fwd_search :
        Merged peaks from forward search
    rev_search :
        Merged peaks from reverse search
    """
    fwd_search = []
    rev_search = []
    # forward search
    for k, c in enumerate(candidates):
        if k < len(candidates) - 1:
            new_total = c[2] + candidates[k + 1][2]
            new_density = new_total / (candidates[k + 1][1] - c[0])
            if new_density >= donor_tolerance * c[3]:
                distance_check = c[1] - c[0] < ce_trigger or candidates[k + 1][0] - c[1] > max_distance
                if distance_check:
                    fwd_search.append(c)
                    continue
                merged = (c[0], candidates[k + 1][1], new_total, new_density)
                fwd_search.append(merged)
            else:
                fwd_search.append(c)
        else:
            fwd_search.append(c)
    # reverse search
    for k in range(len(candidates) - 1, -1, -1):
        c = candidates[k]
        if k > 0:
            new_total = c[2] + candidates[k - 1][2]
            new_density = new_total / (c[1] - candidates[k - 1][0])
            if new_density >= donor_tolerance * c[3]:
                distance_check = c[1] - c[0] < ce_trigger or c[0] - candidates[k - 1][1] > max_distance
                if distance_check:
                    rev_search.append(c)
                    continue
                merged = (candidates[k - 1][0], c[1], new_total, new_density)
                rev_search.append(merged)
            else:
                rev_search.append(c)
        else:
            rev_search.append(c)
    return fwd_search, rev_search


def merge_covs(covs, chromosome_of_interest):
    """
    Merge coverage tracks

    Parameters
    ----------
    covs : list of dicts
        List of coverage dicts for each rep
    chromosome_of_interest : str
        Name of the chromosome/contig to be working on

    Returns
    -------
    merged_coverage : np.ndarray
        merged coverage tracks
    """
    merged_coverage = None
    for cd in covs:
        cov = np.load(cd[chromosome_of_interest])
        if merged_coverage is None:
            merged_coverage = np.zeros(cov.shape[0], dtype=np.int32)

        merged_coverage += cov
        del cov
    return merged_coverage


def cut_peaks_dry_run(annotation_gtf, pl_cov_files, mn_cov_files, tss_extension=200, highlight_chromosome="chr1",
                      output_diagnostics=False, save_to=None):
    """
    Select optimal alpha values to join sub peaks/seeds

    Parameters
    ----------
    annotation_gtf : str
        Gene annotation gtf file
    pl_cov_files : list of dicts
        List of coverage dicts for each rep (forward strand)
    mn_cov_files : list of dicts
        List of coverage dicts for each rep (reverse strand)
    tss_extension : int
        Number of bps to be extended from known TSSs
    highlight_chromosome : str
        Name of the chromosome/contig to be working on
    output_diagnostics : bool
        Write out diagnostics
    save_to : None or str
        Name of the output
    Returns
    -------
    selected_threshold : float
        Optimal threshold
    """
    from pints.io_engine import parse_gtf
    annotations = parse_gtf(annotation_gtf)
    transcripts_pc = annotations.loc[np.logical_and(np.logical_and(annotations.feature == "transcript",
                                                                   annotations.gene_type == "protein_coding"),
                                                    annotations.seqname == highlight_chromosome), :]
    log_assert(transcripts_pc.shape[0] > 0,
               "Cannot parse any annotations for protein-coding genes from provided annotations", logger)
    pct_bed = transcripts_pc.loc[:, ("seqname", "start", "end", "transcript_id", "gene_name", "strand", "gene_id")]
    pct_bed.start -= 1
    pct_tss = pct_bed.loc[:, ("seqname", "start", "end", "transcript_id", "gene_name", "strand")]
    pct_tss["start"] = pct_tss.apply(lambda x: x["start"] if x["strand"] == "+" else x["end"] - 1, axis=1)
    pct_tss["end"] = pct_tss.apply(lambda x: x["start"] + 1 if x["strand"] == "+" else x["end"], axis=1)
    pct_tss.drop_duplicates(subset=["seqname", "start", "end"], inplace=True)
    pct_tss_bed = BedTool.from_dataframe(pct_tss)
    pct_tss["start"] -= tss_extension
    pct_tss["end"] += tss_extension
    tss_window = BedTool.from_dataframe(pct_tss).sort().merge(s=True, c=(4, 5, 6),
                                                              o=("distinct", "distinct", "distinct"))
    logger.info("%d annotated TSSs loaded" % pct_tss.shape[0])
    pl_cov = merge_covs(pl_cov_files, highlight_chromosome)
    mn_cov = merge_covs(mn_cov_files, highlight_chromosome)

    def cp_atom(coverage_track, abs_start, abs_end, donor_tolerance, ce_trigger=3):
        starts = []
        ends = []
        sub_peaks = cut_peaks(coverage_track[abs_start:abs_end],
                              donor_tolerance=donor_tolerance,
                              ce_trigger=ce_trigger)
        for sp in sub_peaks:
            starts.append(sp[0] + abs_start)
            ends.append(sp[1] + abs_start)
        return starts, ends

    search_range = np.linspace(0, 1, 101)
    ambs = np.zeros(search_range.shape[0])
    median_sizes = np.zeros(search_range.shape[0])
    for i, dt in enumerate(search_range):
        predicted_peaks_dicts = []
        for window in tss_window:
            if window.strand == "+":
                cov_obj = pl_cov
            else:
                cov_obj = mn_cov
            if np.sum(cov_obj[window.start:window.stop]) > 0:
                ss, es = cp_atom(cov_obj, window.start, window.stop, donor_tolerance=dt)
                for sc, ec in zip(ss, es):
                    predicted_peaks_dicts.append({"seqname": highlight_chromosome,
                                                  "start": sc, "end": ec, "name": ".",
                                                  "score": ".", "strand": window.strand})
        predicted_peaks = BedTool.from_dataframe(pd.DataFrame(predicted_peaks_dicts))
        tmp_result = predicted_peaks.intersect(pct_tss_bed, c=True, s=True)
        tmp_df = tmp_result.to_dataframe(names=("seqname", "start", "end", "name", "score", "strand", "hits"))
        lens = tmp_df["end"] - tmp_df["start"]
        ambiguous_rate = tmp_df.loc[np.logical_and(lens > 2, tmp_df.hits > 1), :].shape[0] / \
                         tmp_df.loc[lens > 2, :].shape[0]

        median_sizes[i] = np.median(lens[lens >= 3])
        ambs[i] = ambiguous_rate

    smoothed_median_sizes = gaussian_filter1d(median_sizes, 1)
    smoothed_ambs = gaussian_filter1d(ambs, 1)
    size_knee_x, _ = get_elbow(search_range, smoothed_median_sizes)
    amb_knee_x, _ = get_elbow(search_range, smoothed_ambs)
    selected_threshold = max(size_knee_x, amb_knee_x)

    if output_diagnostics and save_to is not None:
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(2, 1)
        axs[0].plot(search_range, median_sizes)
        axs[0].axvline(size_knee_x)
        axs[0].set_ylabel("Median size of peaks")

        axs[1].plot(search_range, ambs, label="Ambiguous rate")
        axs[1].axvline(amb_knee_x)
        axs[1].set_ylabel("Ambiguous rate")
        plt.tight_layout()
        plt.savefig(save_to, transparent=True, bbox_inches="tight")
        plt.close()

    return selected_threshold


def cut_peaks(window, donor_tolerance, ce_trigger, join_distance=1, peak_rel_height=1.,
              max_distance=30):
    """
    Cut peaks from the given window

    Parameters
    ----------
    window : array-like
        Per base read counts / coverage
    donor_tolerance : float
        From sub peak seeking for merging, the new density should be larger than dt*prev_d
    ce_trigger : int
        Sub peak narrower than cet will trigger receptor tolerance check
    join_distance : int
        The maximum distance for two subpeaks to be joined.
    peak_rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if width is given.
    max_distance : int
        max distance between two subpeaks to be joined, by default, 20
    Returns
    -------
    merged_intervals : list
        List of tuples of merged intervals [(start_1, end_1), ... , (start_n, end_n)]
    """
    peaks, _ = find_peaks(window, rel_height=peak_rel_height)
    _, _, starts, ends = peak_widths(window, peaks, rel_height=peak_rel_height)
    intervals = []
    for k, start in enumerate(starts):
        intervals.append((int(start), int(ends[k])))
    mi = merge_intervals(intervals=intervals, distance=join_distance)
    candidates = []
    for m in mi:
        events = 0
        for i in range(m[0], m[1] + 1):
            events += window[i]
        candidates.append((m[0], m[1], events, events / (m[1] - m[0])))

    f, r = quasi_max_score_segment(candidates=candidates, donor_tolerance=donor_tolerance,
                                   ce_trigger=ce_trigger, max_distance=max_distance)
    f.extend(r)
    final = merge_intervals(f, distance=join_distance)
    return final


def check_window_chromosome(rc_file, output_file, strand_sign, chromosome_name, subpeak_file, fdr_target,
                            small_peak_threshold=5, min_mu_percent=0.1, disable_qc=False,
                            disable_ler=False, enable_eler=True, eler_min=1., sensitive=False,
                            fc_cutoff=1.5, init_dens_cutoff=0.25, init_height_cutoff=4):
    """
    Evaluate windows on a chromosome

    Parameters
    ----------
    rc_file : str
        Path to numpy saved read coverage info
    output_file : str
        Path to store outputs
    strand_sign : str
        Strand of windows
    chromosome_name : str
        Name of this chromosome
    subpeak_file : str
        File containing info about all subpeaks
    fdr_target : float
        fdr target
    small_peak_threshold : int
        Peaks shorter than this threshold will be evaluated by Poisson instead of ZIP, by default, 5
    min_mu_percent : float
        Local backgrounds smaller than this percentile among all peaks will be replaced. By default, 0.1.
    disable_qc : bool
        Disable QC warnings. By default, False.
    disable_ler : bool
        Disable LER. By default, False.
    enable_eler : bool
        Set it as False to disable enhanced LER
    eler_min : float
        Only consider peaks with density equal to or greater than this value when performing ELER calibration.
    sensitive : bool
        Use LRT for more sensitive peak testing
    fc_cutoff : float
        Fold-change cutoff to enable likelihood-ratio test
    init_dens_cutoff : float
        Peaks with fewer initiation locations than this fraction will not undergo LRT testing
    init_height_cutoff : float
        Peaks with initiation summit lower than this value will not undergo LRT testing

    Returns
    -------
    result_df : pd.DataFrame
        Window bed in dataframe
    """
    global housekeeping_files
    per_base_cov = np.load(rc_file, allow_pickle=True)
    subpeak_bed = output_file.replace(".bed", __SUB_PEAK_TPL__ % chromosome_name)
    if fc_cutoff > 0:
        real_fc_cutoff = 1. / fc_cutoff
    else:
        real_fc_cutoff = 1.
    bins = []
    all_peak_mus = []
    try:
        with open(subpeak_bed, "w") as spb_fh, gzip.open(subpeak_file, "rt") as peak_obj:
            for line in peak_obj:
                items = line.strip().split("\t")
                start = int(items[1])
                end = int(items[2])
                peak_region = per_base_cov[start:end]
                window_value = peak_region.sum()
                n_start_sizes = sum(peak_region > 0)  # if n start sizes is smaller than 3, then ZIP shouldn't be used
                peak_len = end - start
                if peak_len < small_peak_threshold \
                        or window_value == 0 or n_start_sizes <= 3:
                    mu_peak = window_value / peak_len
                    var_peak = np.var(np.log2(peak_region + np.finfo(float).eps))
                    pi_peak = 0
                else:
                    mu_peak, var_peak, pi_peak, _, _ = stat_tester.fit(peak_region)
                    all_peak_mus.append(mu_peak)
                x = np.argmax(peak_region)
                summit_coord = start + x
                non_zero_loci = (peak_region > 0).sum()

                spb_fh.write("%s\t%d\t%d\t%s\t%f\t%s\t%s\t%f\t%d\t%d\t%d\n" % (
                    chromosome_name, start, end, items[3], mu_peak, var_peak, strand_sign, pi_peak, non_zero_loci, 
                    summit_coord, peak_region[x]))

        index_bed_file(subpeak_bed, logger=logger)

        bed_handler = pysam.TabixFile(subpeak_bed + ".gz", index=subpeak_bed + ".gz.csi")
        if len(all_peak_mus) == 0:
            peak_threshold = 1
            bkg_mu_threshold = 0
            empirical_true_peak_threshold = 0
        else:
            peak_threshold = 1
            bkg_mu_threshold = np.quantile(all_peak_mus, min_mu_percent)
            search_grid = np.linspace(0, 1, 21)
            grid_quantiles = np.quantile(all_peak_mus, search_grid)
            probe = np.where(grid_quantiles > 0.5)[0]
            if probe.shape[0] > 0:
                suggest_val = search_grid[probe[0]]
            else:
                suggest_val = search_grid[-1]

            if not disable_qc:
                if bkg_mu_threshold < 0.5 and len(all_peak_mus) > 1000:
                    bkg_mu_threshold = np.quantile(all_peak_mus, suggest_val)
                    with open(subpeak_bed.replace(".bed", ".mmp"), "w") as fh:
                        fh.write(str(suggest_val))

            sorted_arr = np.sort(all_peak_mus)
            Y = np.log1p(sorted_arr[sorted_arr >= eler_min])
            X = np.log10(np.arange(Y.shape[0]) + 1)
            _, empirical_true_peak_threshold = get_elbow(X, Y)
            if np.isnan(empirical_true_peak_threshold):
                enable_eler = False
            else:
                empirical_true_peak_threshold = np.expm1(empirical_true_peak_threshold)
            logger.info("Minimum mu in local environment %f (%s)" % (bkg_mu_threshold, chromosome_name))

            if enable_eler:
                logger.info(
                    "Peaks with densities higher than {0} ({1}) will be considered as candidate peaks in local "
                    "background".format(
                        empirical_true_peak_threshold, chromosome_name)
                )
        global_cache = {}
        with gzip.open(subpeak_bed + ".gz", "rt") as peak_obj:
            for peak in peak_obj:
                candidate_peak = peak.split("\t")
                peak_start = int(candidate_peak[1])
                peak_end = int(candidate_peak[2])
                peak_id = candidate_peak[3]
                mu_peak = float(candidate_peak[4])
                peak_var = float(candidate_peak[5])
                pi_peak = float(candidate_peak[7])
                peak_non_zeros = int(candidate_peak[8])
                peak_summit = int(candidate_peak[9])
                peak_summit_val = int(candidate_peak[10])

                pval, wv, mu_bg, pi_bg, lerc, lrtp, sig_dens = check_window(
                    coord_start=peak_start, coord_end=peak_end, mu_peak=mu_peak,
                    var_peak=peak_var, pi_peak=pi_peak, chromosome_coverage=per_base_cov,
                    peak_in_bg_threshold=peak_threshold, mu_bkg_minimum=bkg_mu_threshold,
                    sp_bed_handler=bed_handler, chromosome_name=chromosome_name,
                    fdr_target=fdr_target, cache=global_cache, disable_ler=disable_ler,
                    enable_eler=enable_eler, top_peak_mu=empirical_true_peak_threshold)
                if wv > 0:
                    use_lrt = sensitive and pi_peak > 0. and pi_bg > 0. and (
                                mu_bg / mu_peak < real_fc_cutoff) and sig_dens > init_dens_cutoff and peak_summit_val > init_height_cutoff
                    bins.append(
                        (chromosome_name, peak_start, peak_end, peak_id, lrtp if use_lrt else pval,
                         wv, mu_bg, pi_bg, mu_peak, pi_peak, peak_var, peak_summit, peak_summit_val,
                         lerc[0], lerc[1], lerc[2], peak_non_zeros))
    except TypeError as ex:
        logger.error(str(chromosome_name) + "\t" + str(subpeak_file))
        logger.error(ex)
    result_df = pd.DataFrame(bins, columns=("chromosome", "start", "end", "name", "pval", "reads",
                                            "mu_0", "pi_0", "mu_1", "pi_1", "var_1", "summit", "summit_val",
                                            "ler_1", "ler_2", "ler_3", "non_zeros"))
    result_df["strand"] = strand_sign
    result_df = result_df.loc[:, ("chromosome", "start", "end", "name", "pval", "strand", "reads",
                                  "mu_0", "pi_0", "mu_1", "pi_1", "var_1",
                                  "summit", "summit_val", "ler_1", "ler_2", "ler_3", "non_zeros")]
    return result_df


def stratified_filtering(tmp_df, output_file, fdr_target, dry_run=False, **kwargs):
    """
    Stratified multiple testing correction

    Parameters
    ----------
    tmp_df : pd.DataFrame

    output_file : str
        Path of output files
    fdr_target : float
        FDR target
    dry_run : bool
        Set it to True to put placeholder instead of adjusted p-values
    **kwargs

    Returns
    -------
    result_df : str
        Path to a compressed and indexed bed file
    """
    if dry_run:
        result_df = tmp_df.copy()
        result_df["padj"] = result_df["pval"]
        result_df = result_df.loc[:, COMMON_HEADER]
    else:
        fn, ext = os.path.splitext(output_file)

        small_peak_threshold = kwargs.get("small_peak_threshold", 5)
        top_peak_threshold = kwargs.get("top_peak_threshold", 0.75)
        big_peaks_probe = tmp_df.end - tmp_df.start > small_peak_threshold
        small_peaks_probe = tmp_df.end - tmp_df.start <= small_peak_threshold
        lamb_global = tmp_df.loc[big_peaks_probe, "mu_1"].quantile(top_peak_threshold)
        lamb_global = lamb_global if lamb_global >= 1 else 1

        if kwargs.get("output_diagnostics", False):
            import matplotlib.pyplot as plt
            tx = np.arange(0, 1, 0.01)
            qs = [tmp_df.loc[big_peaks_probe, "mu_1"].quantile(x) for x in tx]
            plt.plot(tx, qs)
            plt.xlabel("Quantile")
            plt.ylabel("Peak density")
            plt.tight_layout()
            plt.savefig(fn + "_small_peak_threshold.pdf", transparent=True, bbox_inches="tight")
            plt.close()
        logger.info("Lambda for small peaks: %f" % lamb_global)

        small_pois_dict = {}

        def cached_pois(x):
            if x["end"] - x["start"] > small_peak_threshold:
                return x["pval"]
            expected_counts = lamb_global * (x["end"] - x["start"])
            k = "{reads}-{tr}".format(reads=x["reads"], tr=expected_counts)
            if k not in small_pois_dict:
                small_pois_dict[k] = poisson.sf(x["reads"], expected_counts)
            return small_pois_dict[k]

        is_disable_small = kwargs.get("disable_small", False)
        if is_disable_small:
            tmp_df = tmp_df.loc[big_peaks_probe, :]
            tmp_df_sm = None
        else:
            tmp_df["pval"] = tmp_df.apply(cached_pois, axis=1)

            tmp_df_sm = independent_filtering(tmp_df.loc[small_peaks_probe, :], fdr_target=fdr_target,
                                              output_to=fn + "_idpf_sm.pdf",
                                              logger=logger, **kwargs)
        if kwargs.get("output_diagnostics", False):
            pval_dist(tmp_df.loc[tmp_df["end"] - tmp_df["start"] > small_peak_threshold, "pval"],
                      logger=logger,
                      output_diagnostics=kwargs["output_diagnostics"],
                      output_to=fn + "_broad_pval_hist.pdf")
            pval_dist(tmp_df.loc[tmp_df["end"] - tmp_df["start"] <= small_peak_threshold, "pval"],
                      logger=logger,
                      output_diagnostics=kwargs["output_diagnostics"],
                      output_to=fn + "_narrow_peaks_pval_hist.pdf")
            pval_dist(tmp_df["pval"],
                      logger=logger,
                      output_diagnostics=kwargs["output_diagnostics"],
                      output_to=fn + "_pval_hist.pdf")

        # stratified independent filtering
        tmp_df_bg = independent_filtering(tmp_df.loc[tmp_df["end"] - tmp_df["start"] > small_peak_threshold,
                                          :], output_to=fn + "_idpf_bg.pdf", logger=logger, **kwargs)
        if tmp_df_sm is not None:
            result_df = pd.concat([tmp_df_sm, tmp_df_bg])
            n_sig_small = sum(tmp_df_sm.padj < fdr_target)
            n_sig_big = sum(tmp_df_bg.padj < fdr_target)
            if not kwargs.get("disable_qc", False):
                small_ratio = n_sig_small / (n_sig_big + 1)
                value_playgrounds = ()
                if small_ratio > 1:
                    value_playgrounds = (0.99, 0.95, 0.9)
                elif small_ratio >= 0.2:
                    value_playgrounds = (0.99, 0.95, 0.9, 0.85)
                elif small_ratio >= 0.18:
                    value_playgrounds = (0.99, 0.95, 0.9, 0.85, 0.8)

                if len(value_playgrounds) > 0:
                    tpt_suggestion = None
                    for vp in value_playgrounds:
                        if top_peak_threshold < vp:
                            tpt_suggestion = vp

                    if tpt_suggestion is None:
                        tpt_suggestion = 1
                    with open(fn + ".tpt", "w") as fh:
                        fh.write(str(tpt_suggestion))
        else:
            result_df = tmp_df_bg

        result_df = result_df.loc[:, COMMON_HEADER]
    result_df.sort_values(by=['chromosome', 'start'], inplace=True)
    result_df.to_csv(output_file, sep="\t", index=False, header=False)
    index_bed_file(output_file, logger=logger)
    return output_file + ".gz"


def peaks_single_strand(per_base_cov, output_file, shared_peak_definitions, strand_sign, fdr_target,
                        **kwargs):
    """
    Calling peaks on one strand

    Parameters
    ----------
    per_base_cov : dict
        Per base cov for available chromosomes
    output_file : str
        Path of output files
    shared_peak_definitions : dict
        Dictionary containing all subpeaks per chromosome
    strand_sign : str
        Strand sign for the data
    fdr_target : float
        FDR target
    **kwargs :

    Returns
    -------
    result_df : str
        Path to a compressed and indexed bed file
    """
    global housekeeping_files

    args = []
    for chrom, pbc_npy in per_base_cov.items():
        if shared_peak_definitions[chrom] is None:  # bypass chromosomes without signals
            continue
        sub_peaks_name = output_file.replace(".bed", __SUB_PEAK_TPL__ % chrom)
        merged_name = output_file.replace(".bed", "_%s_merged_windows.bed" % chrom)
        args.append((pbc_npy, output_file, strand_sign, chrom, shared_peak_definitions[chrom], fdr_target,
                     kwargs.get("small_peak_threshold", 5), kwargs.get("min_mu_percent", 0.1),
                     kwargs.get("disable_qc", False), kwargs.get("disable_ler", False),
                     kwargs.get("enable_eler", True), kwargs.get("eler_lower_bound", 1.),
                     kwargs.get("sensitive", False), kwargs.get("fc_cutoff", 1.5),
                     kwargs.get("init_dens_cutoff", 0.25), kwargs.get("init_height_cutoff", 4)))
        housekeeping_files.append(merged_name)
        housekeeping_files.append(sub_peaks_name + ".gz")
        housekeeping_files.append(sub_peaks_name + ".gz.csi")

    if kwargs.get("thread_n", 1) == 1:
        # for debugging
        sub_dfs = []
        for arg_i in args:
            sub_dfs.append(
                check_window_chromosome(*arg_i)
            )
    else:
        with Pool(kwargs.get("thread_n", 1)) as pool:
            sub_dfs = pool.starmap(check_window_chromosome, args)

    sub_dfs = [sdf for sdf in sub_dfs if sdf is not None]
    log_assert(len(sub_dfs) > 0, "No signal found across all chromosomes!", logger)
    tmp_df = pd.concat(sub_dfs)

    if kwargs.get("output_diagnostics", False):
        tmp_df.to_csv(output_file.replace(".bed", "_debug.csv"), index=False)

    return stratified_filtering(tmp_df=tmp_df, output_file=output_file, fdr_target=fdr_target, **kwargs)


def merge_opposite_peaks(sig_peak_bed, peak_candidate_bed, divergent_output_bed, bidirectional_output_bed,
                         singleton_bed, fdr_target, stringent_only=False, **kwargs):
    """
    Merge peaks on the opposite strand and generate divergent peak pairs

    Parameters
    ----------
    sig_peak_bed : str
        Path to bed file which contains significant peaks
    peak_candidate_bed : str
        Path to bed file which contains all candidate peaks on the opposite strand
    divergent_output_bed : str
        Path to output which stores divergent peaks
    bidirectional_output_bed : str
        Path to output which stores bidirectional peaks (divergent / convergent)
    singleton_bed : str
        Path to output which stores significant peaks which failed to pair
    fdr_target : float
        FDR target
    stringent_only : bool
        Set it to True if you only want to keep significant pairs (both peaks needs to be significant)

    **kwargs :
        close_threshold : int
            Distance threshold for two peaks (on opposite strands) to be merged
        min_len_opposite_peaks : int
            Minimum length requirement for peaks on the opposite strand to be paired,
            set it to 0 to lose this requirement
    Returns
    -------

    """
    tbx = pysam.TabixFile(peak_candidate_bed, index=peak_candidate_bed + ".csi")
    fh = open(sig_peak_bed, "r")
    div_fh = open(divergent_output_bed, "w")
    bid_fh = open(bidirectional_output_bed, "w")
    sfp_fh = open(singleton_bed, "w")  # singletons failed to pair
    close_threshold = kwargs.get("close_threshold", 300)
    min_len_opposite_peaks = kwargs.get("min_len_opposite_peaks", 0)
    for _, line in enumerate(fh):
        items = line.strip().split("\t")
        start = int(items[1])
        end = int(items[2])
        current_summit = int(items[-2])
        current_summit_val = int(items[-1])
        # allow overlapping
        if items[5] == "+":
            query_start = start - close_threshold
            query_start = query_start if query_start >= 0 else 0
            query_end = end
        else:
            query_start = start
            query_end = end + close_threshold

        opposite_start = np.nan
        opposite_end = np.nan
        opposite_pval = np.nan
        opposite_qval = np.nan
        opposite_sum = 0
        opposite_starts = []
        opposite_ends = []
        opposite_qvals = []
        opposite_pvals = []
        opposite_vals = []
        opposite_summits = []
        opposite_summit_vals = []
        # since windows on each strand have been merged,
        # so here I expect the following iter returns at
        # most two records
        try:
            query_start = query_start if query_start >= 0 else 0
            for hit in tbx.fetch(items[0], query_start, query_end, parser=pysam.asTuple()):
                hit_start = int(hit[1])
                hit_end = int(hit[2])
                hit_score = float(hit[4])
                hit_reads = float(hit[6])  # in case the read counts had been normed
                opposite_summit = int(hit[-2])
                opposite_summit_val = int(hit[-1])
                # filter peaks on the other strand which are shorter than a threshold
                if min_len_opposite_peaks > 0 and hit_end - hit_start < min_len_opposite_peaks:
                    continue
                if stringent_only and float(hit[7]) > fdr_target:
                    continue
                opposite_starts.append(hit_start)
                opposite_ends.append(hit_end)
                opposite_qvals.append(hit_score)
                opposite_pvals.append(float(hit[7]))
                opposite_vals.append(hit_reads)
                opposite_summits.append(opposite_summit)
                opposite_summit_vals.append(opposite_summit_val)
            if len(opposite_pvals) > 0:
                # if there are multiple significant peaks (more common in promoter regions)
                # pair peaks with the nearest, opposite, and significant peak
                probe_sig_peaks = np.where(np.array(opposite_pvals) < fdr_target)[0]
                if probe_sig_peaks.shape[0] > 1:
                    abs_distance = np.abs(np.array(opposite_summits) - current_summit)  # the abs dist between TSSs
                    index = probe_sig_peaks[  # select the abs dist for significant peaks
                        abs_distance[probe_sig_peaks].argmin()  # and further select peak with the closest one
                    ]
                else:
                    index = np.argmin(opposite_pvals)
                opposite_start = opposite_starts[index]
                opposite_end = opposite_ends[index]
                opposite_pval = opposite_pvals[index]
                opposite_qval = opposite_qvals[index]
                opposite_summit = int(opposite_summits[index])
                opposite_summit_val = int(opposite_summit_vals[index])
                opposite_sum = sum(opposite_vals[:index + 1])
        except ValueError as err:
            pass
        if opposite_start is np.nan:
            sfp_fh.write(line)
        else:
            items.extend((str(opposite_start), str(opposite_end), str(opposite_pval), str(opposite_sum)))
            coords = (int(items[1]), int(items[2]), opposite_start, opposite_end)

            if items[5] == "+":
                fwd_summit = current_summit
                rev_summit = opposite_summit
            else:
                fwd_summit = opposite_summit
                rev_summit = current_summit
                t = current_summit_val
                current_summit_val = opposite_summit_val
                opposite_summit_val = t

            tre_start = min(coords)
            tre_end = max(coords)
            if opposite_qval < fdr_target:
                pairing_confidence = "Stringent(qval)"
            elif opposite_pval < fdr_target:
                pairing_confidence = "Stringent(pval)"
            else:
                pairing_confidence = "Relaxed"
            if tre_end - tre_start > kwargs.get("div_size_min", 0):
                candidate_values = (items[0], str(tre_start), str(tre_end), ".",
                                    items[4], items[5], str(float(items[6]) + opposite_sum), items[1],
                                    items[2], str(opposite_start), str(opposite_end), str(fwd_summit),
                                    str(current_summit_val), str(rev_summit), str(opposite_summit_val),
                                    pairing_confidence + "\n")

                bid_fh.write("\t".join(candidate_values))
                if fwd_summit - rev_summit >= kwargs.get("summit_dist_min", 0):
                    div_fh.write("\t".join(candidate_values))
            else:
                sfp_fh.write(line)
    fh.close()
    bid_fh.close()
    div_fh.close()
    sfp_fh.close()


def housekeeping(pybedtools_prefix=""):
    """
    Delete intermediate files

    Returns
    -------

    """
    global housekeeping_files
    try:
        for hf in housekeeping_files:
            if os.path.exists(hf):
                os.remove(hf)
        if pybedtools_prefix != "":
            import shutil
            shutil.rmtree(pybedtools_prefix)
    except Exception as ex:
        logger.warning(str(ex))


def show_parameter_info(input_bam, **kwargs):
    """
    Show parameters

    Parameters
    ----------
    input_bam : str
        Path to the input
    kwargs

    Returns
    -------

    """
    _args = kwargs.copy()
    _args.pop("bam_file", None)
    bam_parser = _args.pop("bam_parser", None)
    input_pl_bw = _args.pop("bw_pl", None)
    input_mn_bw = _args.pop("bw_mn", None)
    control_bam = _args.pop("ct_bam", None)
    control_pl_bw = _args.pop("ct_bw_pl", None)
    control_mn_bw = _args.pop("ct_bw_mn", None)
    logger.info("Command")
    logger.info(" ".join(sys.argv))
    logger.info("Parameters")
    if input_bam is not None:
        logger.info("input_bam(s): {input} ({parser})".format(input=" ".join(input_bam), parser=bam_parser))
    else:
        logger.info("input_pl_bw(s): {input}".format(input=" ".join(input_pl_bw)))
        logger.info("input_mn_bw(s): {input}".format(input=" ".join(input_mn_bw)))
    if control_bam is not None:
        logger.info("ct_bam(s): {input} ({parser})".format(input=control_bam, parser=bam_parser))
    elif control_pl_bw is not None and control_mn_bw is not None:
        logger.info("ct_bw_pl(s): {input}".format(input=" ".join(control_pl_bw)))
        logger.info("ct_bw_mn(s): {input}".format(input=" ".join(control_mn_bw)))

    bam_parse_pars = {"seq_rc", "mapq_threshold", "output_chrom_size"}
    alpha_ft_pars = {"annotation_gtf", "tss_extension", "focused_chrom"}
    if input_bam is None:
        for a in bam_parse_pars:
            _args.pop(a, None)
    if _args.get("annotation_gtf", "None") is None:
        for a in alpha_ft_pars:
            _args.pop(a, None)
    testing_pars = {"window_size_threshold", "ce_trigger", "peak_distance",
                    "peak_width", "div_size_min", "summit_dist_min"}
    for k, v in _args.items():
        if k not in testing_pars:
            logger.info("%s: %s" % (k, v))


def unified_element_definition(coverage_dict, chromosome_of_interest, strand_sign, output_file,
                               join_distance=1, ce_donor=1.0, ce_trigger=3):
    """
    Unified element boundary definition
    If multiple replicates are present, then this function will merge all signal tracks together

    Parameters
    ----------
    coverage_dict : list of dicts
        List of coverage dicts for each rep
    chromosome_of_interest : str
        Name of the chromosome/contig to be working on
    strand_sign : str
        Sign of strand to be added in the bed file, + or -
    output_file : str
        Prefix (including path) for outputs
    join_distance : int
        Force joining sub peaks within this distance, by default 1
    ce_donor : float
        Donor tolerance in best score segments
    ce_trigger : int
        Trigger for receptor tolerance checking

    Returns
    -------
    subpeak_bed : str or None
        file name for the subpeak bed file, or None if there's no peak
    """
    subpeak_bed = output_file.replace(".bed", __SUB_PEAK_TPL__ % chromosome_of_interest)
    bins = []
    merged_coverage = merge_covs(coverage_dict, chromosome_of_interest)

    for window, coord in sliding_window(merged_coverage):
        if window > 0:  # no reads in the bin
            bins.append((chromosome_of_interest, coord[0], coord[1], window))

    tmp_df = pd.DataFrame(bins, columns=("chromosome", "start", "end", "reads"))
    tmp_df["name"] = "."
    tmp_df["strand"] = strand_sign
    tmp_df = tmp_df.loc[:, ("chromosome", "start", "end", "name", "reads", "strand")]
    if tmp_df.shape[0] == 0:  # no hit
        return None
    # merge windows in case peaks are split into different windows
    bed_obj = BedTool(tmp_df.to_csv(sep="\t", index=False, header=False), from_string=True)
    bed_obj = bed_obj.merge(c=(4, 5, 6), o=("distinct", "sum", "distinct"))

    with open(subpeak_bed, "w") as spb_fh:
        index = 1
        for _, row in enumerate(bed_obj):
            sub_peaks = cut_peaks(merged_coverage[row.start:row.end],
                                  donor_tolerance=ce_donor,
                                  ce_trigger=ce_trigger,
                                  join_distance=join_distance)
            for sp in sub_peaks:
                start = sp[0] + row.start
                end = sp[1] + row.start
                peak_region = merged_coverage[start:end]
                summit_coord = start + np.argmax(peak_region)
                spb_fh.write("%s\t%d\t%d\t%s-%d\t%d\n" % (chromosome_of_interest, start, end, chromosome_of_interest,
                                                          index, summit_coord))
                index += 1

    if index > 1:
        index_bed_file(subpeak_bed, logger=logger)
        return subpeak_bed + ".gz"
    return None


def inferring_elements_from_other_reps(prefix, n_samples):
    """
    Infer bidirectional/divergent elements by borrowing signals from reps

    Parameters
    ----------
    prefix : str
        Prefix for outputs (including path)
    n_samples : int
        number of samples/reps
    Returns
    -------

    """
    bids = []
    divs = []
    sigs = []
    for rep in range(1, n_samples + 1):
        sample_prefix = prefix + "_%d" % rep
        bid_file = sample_prefix + "_bidirectional_peaks.bed"
        div_file = sample_prefix + "_divergent_peaks.bed"
        single_file = sample_prefix + "_unidirectional_peaks.bed"
        if os.path.exists(bid_file):
            bids.append(BedTool(bid_file))
        if os.path.exists(div_file):
            divs.append(BedTool(div_file))
        if os.path.exists(single_file):
            sigs.append(BedTool(single_file))
    if len(bids) > 0:
        merged_bids = BedTool.cat(*bids, c=(4, 5, 6,), o=("distinct", "distinct", "distinct",))
    else:
        merged_bids = None
    if len(divs) > 0:
        merged_divs = BedTool.cat(*divs, c=(4, 5, 6,), o=("distinct", "distinct", "distinct",))
    else:
        merged_divs = None

    for separate_calls, pool, label in zip((bids, divs), (merged_bids, merged_divs), ("bidirectional", "divergent")):
        if pool is None:
            continue
        for i, separate_bed in enumerate(separate_calls):
            not_reported_ele = pool.intersect(separate_bed, v=True).intersect(sigs[i], u=True)
            BedTool.cat(*[separate_bed, not_reported_ele], postmerge=False).sort().saveas(
                prefix + "_%d_%s_peaks.bed" % (i + 1, label))


def get_epig_annotation(annotation_source, output_dir):
    """
    Get epigenomic annotation

    Parameters
    ----------
    annotation_source : str
        Biosample name or a path to a local bigBed file
    output_dir : str
        Download annotation to this directory, if necessary

    Returns
    -------
    dst : str
        "" if no annotation is not available or a path to a bigBed file
    """
    # figure out what's the source
    # first assume it's from PINTS webserver
    dst = ""
    from urllib.request import urlretrieve
    from urllib.error import URLError
    import urllib.parse
    import json
    import socket
    expected_url = "https://pints.yulab.org/api/ea/{annotation_source}".format(
        annotation_source=urllib.parse.quote(annotation_source)
    )

    socket.setdefaulttimeout(30)  # if server doesn't respond in 30s, then raise error
    try:
        if annotation_source is None or annotation_source == "":
            raise ValueError("annotation_source cannot be empty")

        with urllib.request.urlopen(expected_url) as response:
            json_raw = response.read()
        parsed_json = json.loads(json_raw)
        logger.info("Downloading epigenomic annotation")
        socket.setdefaulttimeout(None)
        if parsed_json["status"] == 1:
            _, fn = os.path.split(parsed_json["msg"])
            dst = output_dir + "_" + fn
            if not os.path.exists(dst):
                urlretrieve(parsed_json["msg"], dst)
        else:
            logger.error(parsed_json["msg"])

    except URLError as ex:
        logger.error(ex)
    except json.JSONDecodeError as ex:
        logger.error("Cannot parse returned info {s} from PINTS web server.".format(s=ex))
    except Exception as ex:
        logger.error(ex)
    return dst


def pair(sample_prefix, df_dict, fdr_target, stringent_only, specific_suffix="", **kwargs):
    """

    Parameters
    ----------
    sample_prefix : str

    df_dict : dict
        key : "pl" or "mn"
        value: pd.DataFrame
    fdr_target : float

    stringent_only : bool

    specific_suffix : str
        If there's anything you want to add to the outputs

    kwargs :
        window_size_threshold : int
            Peaks larger than this value will be discarded. By default, 2000.
        keep_sticks : bool
            Set this to True to keep significant peaks with signal on a single position.

    Returns
    -------
    (bidirectional_peaks, divergent_peaks, unidirectional_peaks) : (str, str, str)
        Path to bidirectional peaks, path to divergent peaks, path to unidirectional peaks
    intermediate_files : PairedIntermediateFiles
        Files recorded in this obj should be deleted
    """
    data_structure = namedtuple("PairedIntermediateFiles", field_names=(
        "pl_sig_peaks", "div_sig_pl", "bid_sig_pl", "single_sig_pl",
        "mn_sig_peaks", "div_sig_mn", "bid_sig_mn", "single_sig_mn"))
    generated_files = []
    for label, anti_label in zip(("pl", "mn"), ("mn", "pl")):
        peak_df = df_dict[label].copy()
        peak_df = peak_df.loc[peak_df["end"] - peak_df["start"] < kwargs.get("window_size_threshold", 2000), :]

        sig_bins = peak_df.loc[peak_df["padj"] < fdr_target, :]
        generated_files.append(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label))
        with open(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label), "w") as f:
            if kwargs.get("keep_sticks", True):
                sig_bins.to_csv(f, sep="\t", index=False, header=False)
            else:
                sig_bins.loc[sig_bins.non_zeros > 1, :].to_csv(f, sep="\t", index=False, header=False)

        if not os.path.exists(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label)):
            generated_files.append("")
            generated_files.append("")
            generated_files.append("")
            continue
        merge_opposite_peaks(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label),
                             sample_prefix + "_%s.bed.gz" % anti_label,
                             divergent_output_bed=sample_prefix + "%s_sig_%s_divergent_peaks.bed" % (
                                 specific_suffix, label),
                             bidirectional_output_bed=sample_prefix + "%s_sig_%s_bidirectional_peaks.bed" % (
                                 specific_suffix, label),
                             singleton_bed=sample_prefix + "%s_sig_%s_singletons_peaks.bed" % (specific_suffix, label),
                             fdr_target=fdr_target, stringent_only=stringent_only,
                             **kwargs)
        generated_files.append(sample_prefix + "%s_sig_%s_divergent_peaks.bed" % (specific_suffix, label))
        generated_files.append(sample_prefix + "%s_sig_%s_bidirectional_peaks.bed" % (specific_suffix, label))
        generated_files.append(sample_prefix + "%s_sig_%s_singletons_peaks.bed" % (specific_suffix, label))

    for directionality in ("bidirectional", "divergent", "singletons"):
        exp_pl_file = sample_prefix + "%s_sig_pl_%s_peaks.bed" % (specific_suffix, directionality)
        exp_mn_file = sample_prefix + "%s_sig_mn_%s_peaks.bed" % (specific_suffix, directionality)
        if os.path.exists(exp_pl_file) and os.path.exists(exp_mn_file):
            if directionality != "singletons":
                tmp_pl_bed = BedTool(exp_pl_file)
                tmp_mn_bed = BedTool(exp_mn_file)
                if tmp_pl_bed.count() == 0 and tmp_mn_bed.count() == 0:
                    continue
                elif tmp_pl_bed.count() == 0:
                    pri_merged_file = tmp_mn_bed
                elif tmp_mn_bed.count() == 0:
                    pri_merged_file = tmp_pl_bed
                else:
                    pri_merged_file = BedTool.cat(*[BedTool(exp_pl_file),
                                                    BedTool(exp_mn_file)],
                                                  c=(12, 13, 14, 15, 16),
                                                  o=("collapse", "collapse", "collapse", "collapse", "distinct"))
                pri_merged_df = pri_merged_file.to_dataframe(names=("chrom", "start", "end", "tss_fwd",
                                                                    "tss_fwd_vals", "tss_rev", "tss_rev_vals",
                                                                    "confidence"))
                for str_col in ("tss_fwd", "tss_fwd_vals", "tss_rev", "tss_rev_vals"):
                    pri_merged_df[str_col] = pri_merged_df[str_col].astype(str)
                fwd_tss_cols = pri_merged_df["tss_fwd"].str.split(",", expand=True).fillna(value=-1).astype(int)
                fwd_tss_val_cols = pri_merged_df["tss_fwd_vals"].str.split(",", expand=True).fillna(value=-1).astype(
                    int)
                max_index = fwd_tss_val_cols.idxmax(axis=1)
                pri_merged_df["tss_fwd"] = fwd_tss_cols.values[np.arange(fwd_tss_cols.shape[0]), max_index]
                pri_merged_df["tss_fwd_val"] = fwd_tss_val_cols.values[np.arange(fwd_tss_cols.shape[0]), max_index]

                rev_tss_cols = pri_merged_df["tss_rev"].str.split(",", expand=True).fillna(value=-1).astype(int)
                rev_tss_val_cols = pri_merged_df["tss_rev_vals"].str.split(",", expand=True).fillna(value=-1).astype(
                    int)
                max_index = rev_tss_val_cols.idxmax(axis=1)
                pri_merged_df["tss_rev"] = rev_tss_cols.values[np.arange(rev_tss_cols.shape[0]), max_index]
                pri_merged_df["tss_rev_val"] = rev_tss_val_cols.values[np.arange(rev_tss_cols.shape[0]), max_index]
                BedTool.from_dataframe(
                    pri_merged_df.loc[:, ("chrom", "start", "end", "confidence", "tss_fwd", "tss_rev")]).saveas(
                    sample_prefix + "%s_%s_peaks.bed" % (specific_suffix, directionality))
            else:
                BedTool.cat(*[BedTool(exp_pl_file),
                              BedTool(exp_mn_file)],
                            postmerge=False).sort().saveas(
                    sample_prefix + "%s_unidirectional_peaks.bed" % specific_suffix)
    return (sample_prefix + "%s_bidirectional_peaks.bed" % specific_suffix,
            sample_prefix + "%s_divergent_peaks.bed" % specific_suffix,
            sample_prefix + "%s_unidirectional_peaks.bed" % specific_suffix), data_structure._make(generated_files)


def annotate_tre_with_epig_info(peak_file, epig_bigbed_file, only_annotated_records=False, placeholders={}):
    """

    Parameters
    ----------
    peak_file : str
        Path to a bed file
    epig_bigbed_file : str
        Path to a bigbed file
    only_annotated_records : bool
        Set this to True if you only want rows with epigenomic annotations to be kept
    placeholders : dict
        key : column name
        value: values to be put in that column

    Returns
    -------

    """
    peak_df = pd.read_csv(peak_file, sep="\t", header=None)
    annotations = []
    if not os.path.exists(epig_bigbed_file):
        raise FileNotFoundError(epig_bigbed_file)

    with pyBigWig.open(epig_bigbed_file) as ref_bed:
        for _, row in peak_df.iterrows():
            cache = set()
            try:
                hits = ref_bed.entries(row[0], row[1], row[2])
                if hits is not None:
                    for entry in hits:
                        ann = entry[2].split("\t")[-2]
                        if ann.find("ELS") != -1:
                            cache.add("DNase")
                            cache.add("H3K27ac")
                        if ann.find("H3K4me3") != -1 or ann.find("PLS") != -1:
                            cache.add("DNase")
                            cache.add("H3K4me3")
                        if ann.find("CTCF") != -1:
                            cache.add("DNase")
                            cache.add("CTCF")
            except Exception as ex:
                logger.warning(ex)
            finally:
                annotations.append(",".join(cache) if len(cache) > 0 else ".")
    peak_df["epig_annotation"] = annotations

    if only_annotated_records:
        peak_df = peak_df.loc[peak_df.epig_annotation != ".", :]
        for col, value in placeholders.items():
            peak_df[col] = value
    peak_df.to_csv(peak_file, sep="\t", index=False, header=False)


def on_the_fly_qc(output_prefix, min_mu_percent, top_peak_threshold, significant_calls=0, is_strict_mode=False):
    """
    On-the-fly QC

    Parameters
    ----------
    output_prefix : str
        Prefix to all outputs
    min_mu_percent : float
        Current value for --min-mu-percent
    top_peak_threshold : float
        Current value for --top-peak-threshold
    significant_calls : int
        Number of significant calls
    is_strict_mode : bool
        Set this to True will raise RuntimeError if on-the-fly QC find any warnings

    Raises:
        RuntimeError: _description_
    """
    global housekeeping_files
    info_msgs = []
    warning_msgs = []
    mmp_values = []
    for mmp_file in glob("{}*.mmp".format(output_prefix)):
        housekeeping_files.append(mmp_file)
        with open(mmp_file) as fh:
            mmp_values.append(float(fh.read().strip()))
    if len(mmp_values) > 0:
        mmp_suggestion = max(mmp_values)
        if mmp_suggestion > min_mu_percent:
            msg = (
                    "To reduce false positives, PINTS overrided your current "
                    "--min-mu-percent value. "
                    "To dismiss this message, please consider increasing the value "
                    "of --min-mu-percent (current: %.2f) to %.2f." % (
                        min_mu_percent, mmp_suggestion
                    )
            )
            info_msgs.append(msg)

    tpt_values = []
    for tpt_file in glob("{}*.tpt".format(output_prefix)):
        housekeeping_files.append(tpt_file)
        with open(tpt_file) as fh:
            tpt_values.append(float(fh.read().strip()))
    if len(tpt_values) > 0:
        tpt_suggestion = max(tpt_values)
        if tpt_suggestion > top_peak_threshold:
            if tpt_suggestion < 1:
                tpt_suggestion_str = "increasing the value of --top-peak-threshold (current: %.2f) to %.2f or " % (
                    top_peak_threshold, tpt_suggestion)
            else:
                tpt_suggestion_str = ""
            msg = ("The proportion of significant short peaks is relatively high, "
                   "which usually indicates the cap-selection process didn't work well as expected. "
                   "To reduce false positives, please consider %s"
                   "using --disable-small to remove all significant short peaks.") % tpt_suggestion_str
            warning_msgs.append(msg)

    if significant_calls > 150_000:
        msg = ("The number of significant calls is higher than what we usually observe in TSS assays, "
               "please consider using a smaller FDR cutoff (--fdr-target)")
        warning_msgs.append(msg)

    for im in info_msgs:
        logger.info(im)
    if len(warning_msgs) > 0:
        if is_strict_mode:
            raise RuntimeError("\n".join(warning_msgs))
        else:
            for msg in warning_msgs:
                logger.warning(msg)


def parse_input_files(output_dir, output_prefix, filters, pl_cov_target, mn_cov_target, rc_target,
                      bam_files=None, bw_pl_files=None, bw_mn_files=None, merge_replicates=False, **kwargs):
    """
    Parses and processes input files for genomic coverage and read signal data. This function supports
    various input types including BAM files and bigwig files, handles error checking, manages
    chromosomal data filtering, and optionally merges replicate data.

    Parameters
    ----------
    output_dir : str
        Directory where output files will be saved.
    output_prefix : str
        Prefix for output file names.
    filters : dict
        Dictionary containing filtering parameters for reads or coverage.
    pl_cov_target : list
        List to store processed plus strand coverage data.
    mn_cov_target : list
        List to store processed minus strand coverage data.
    rc_target : list
        List to store read count data.
    bam_files : list, optional
        List of input BAM files for processing. Default is None.
    bw_pl_files : list, optional
        List of bigwig files representing plus strand coverage. Default is None.
    bw_mn_files : list, optional
        List of bigwig files representing minus strand coverage. Default is None.
    merge_replicates : bool, optional
        Flag indicating whether replicate data should be combined into a single dataset. Default is False.
    **kwargs : dict, optional
        Additional parameters for read signal parsing, such as 'bam_parser', 'chromosome_startswith',
        or 'seq_rc'.

    Returns
    -------
    None
        Modifies pl_cov_target, mn_cov_target, and rc_target lists in-place.
    """
    global housekeeping_files

    log_assert(
        os.path.isdir(output_dir) and os.access(output_dir, os.W_OK),
        "Output directory %s is not writable." % output_dir, logger
    )

    log_assert(
        isinstance(pl_cov_target, list) and isinstance(mn_cov_target, list) and isinstance(rc_target, list),
        "pl_cov_target, mn_cov_target, and rc_target must be lists.", logger
    )

    if bam_files is not None:
        log_assert(kwargs.get("bam_parser", None) is not None,
                   "Please specify which type of experiment this data "
                   "was generated from with --exp-type", logger)
        for i, bf in enumerate(bam_files):
            log_assert(os.path.exists(bf), "Cannot find input bam file %s" % bf, logger)
            logger.info("Loading {0}...".format(bf))
            plc, mnc, rc = get_read_signal(input_bam=bf,
                                           loc_prime=kwargs["bam_parser"],
                                           chromosome_startswith=kwargs.pop("chromosome_startswith", ""),
                                           reverse_complement=kwargs.get("seq_rc", False),
                                           output_dir=output_dir,
                                           output_prefix=output_prefix + "_%d" % i,
                                           filters=filters,
                                           **kwargs)

            pl_cov_target.append(plc)
            mn_cov_target.append(mnc)
            rc_target.append(rc)
            housekeeping_files.extend(plc.values())
            housekeeping_files.extend(mnc.values())
            logger.info("{0} loaded.".format(bf))
    elif bw_pl_files is not None and bw_mn_files is not None:
        log_assert(len(bw_pl_files) == len(bw_mn_files),
                   "Must provide the same amount of bigwig files for both strands", logger)

        for i, bw_pl in enumerate(bw_pl_files):
            log_assert(os.path.exists(bw_pl), "Cannot find bigwig file %s" % bw_pl, logger)
            log_assert(os.path.exists(bw_mn_files[i]), "Cannot find bigwig file %s" % bw_mn_files[i], logger)
            logger.info("Loading {0} and {1}...".format(bw_pl, bw_mn_files[i]))
            plc, mnc, rc = get_coverage_bw(bw_pl=bw_pl, bw_mn=bw_mn_files[i],
                                           chromosome_startswith=kwargs.get("chromosome_startswith", ""),
                                           output_dir=output_dir,
                                           output_prefix=output_prefix + "_%d" % i)

            pl_cov_target.append(plc)
            mn_cov_target.append(mnc)
            rc_target.append(rc)
            housekeeping_files.extend(plc.values())
            housekeeping_files.extend(mnc.values())
            logger.info("{0} and {1} loaded.".format(bw_pl, bw_mn_files[i]))

    if merge_replicates and len(pl_cov_target) > 1:
        pl_covs, mn_covs, pl_rc, mn_rc = merge_replicates_bw(pl_cov_target, mn_cov_target, output_dir,
                                                             output_prefix + "_merged")
        pl_cov_target[:] = pl_cov_target[:1]
        mn_cov_target[:] = mn_cov_target[:1]
        rc_target[:] = [pl_rc + mn_rc, ]
        pl_cov_target[0] = pl_covs
        mn_cov_target[0] = mn_covs
        housekeeping_files.extend(pl_covs.values())
        housekeeping_files.extend(mn_covs.values())
        logger.info("Replicates merged")


def peak_calling(input_bam, output_dir=".", output_prefix="pints", **kwargs):
    """
    Peak calling wrapper

    Parameters
    ----------
    input_bam : str
        Path to the input bam file
    output_dir : str
        Path to write output
    output_prefix : str
        Prefix for all outputs
    kwargs :

    Returns
    -------

    """
    global housekeeping_files
    global stat_tester
    global iqr_obj
    global __STRICT_QC__
    log_assert(os.path.exists(output_dir) and os.path.isdir(output_dir), "Cannot write to {0}".format(output_dir),
               logger)

    runtime_check(check_updates=kwargs.pop("check_updates", True))
    logger.info("PINTS version: {0}".format(__version__))
    model = kwargs.get("model", "ZIP")
    iqr_strategy = kwargs.get("iqr_strategy", "bgIQR")
    show_parameter_info(input_bam, **kwargs)

    prefix = os.path.join(output_dir, output_prefix)
    pybdt_tmp = os.path.join(output_dir,
                             "pybdt_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + str(os.getpid()))
    if not os.path.exists(pybdt_tmp):
        os.mkdir(pybdt_tmp)
    pybedtools.set_tempdir(pybdt_tmp)

    log_assert(model in ("Poisson", "ZIP"), "Unsupported model", logger)
    if model == "ZIP":
        stat_tester = ZIP()
    elif model == "Poisson":
        stat_tester = Poisson()
    else:
        logger.error("The model you specified {model} is not supported".format(model=model))

    if iqr_strategy == "bgIQR":
        iqr_obj = bgIQR()
    elif iqr_strategy == "pkIQR":
        iqr_obj = pkIQR()
    else:
        logger.error("The IQR strategy you specified {iqr_strategy} is not supported".format(iqr_strategy=iqr_strategy))

    log_assert(input_bam is not None or (kwargs["bw_pl"] is not None and kwargs["bw_mn"] is not None),
               "You must provide PINTS a BAM file (--bam-file) or two bigwig files (--bw-pl and --bw-mn) "
               "for the experiment", logger)

    if kwargs["bw_pl"] is not None and kwargs["bw_mn"] is not None:
        log_assert(len(kwargs["bw_pl"]) == len(kwargs["bw_mn"]),
                   "If you want to use bigwig files as input, make sure "
                   "you provide both bws for forward and reverse strand",
                   logger)

    __STRICT_QC__ = kwargs.get("strict_qc", False)
    disable_qc = kwargs.get("disable_qc", False)
    input_coverage_pl = []
    input_coverage_mn = []
    chromosome_coverage_pl = []
    chromosome_coverage_mn = []
    rcs = []  # read counts for experiment
    ircs = []  # read counts for input/control
    filters = kwargs.pop("filters", ())

    parse_input_files(output_dir=output_dir, output_prefix=output_prefix, filters=filters,
                      pl_cov_target=chromosome_coverage_pl, mn_cov_target=chromosome_coverage_mn,
                      rc_target=rcs, bam_files=input_bam,
                      bw_pl_files=kwargs.get("bw_pl", None), bw_mn_files=kwargs.get("bw_mn", None), **kwargs)
    chromosomes = set().union(*[set(d.keys()) for d in chromosome_coverage_pl])
    chromosomes = chromosomes.union(*[set(d.keys()) for d in chromosome_coverage_mn])

    subpeak_pl_beds = {}
    subpeak_mn_beds = {}
    if kwargs.get("annotation_gtf", None) is not None:
        highlight_chromosome = kwargs.get("highlight_chromosome", "chr1")
        log_assert(os.path.exists(kwargs.get("annotation_gtf", None)), "Cannot find gene annotation file", logger)
        if highlight_chromosome in chromosome_coverage_pl[0]:
            kwargs["donor_tolerance"] = cut_peaks_dry_run(kwargs.get("annotation_gtf", None), chromosome_coverage_pl,
                                                          chromosome_coverage_mn,
                                                          tss_extension=kwargs.get("tss_extension", 200),
                                                          highlight_chromosome=highlight_chromosome,
                                                          output_diagnostics=kwargs.get("output_diagnostics", False),
                                                          save_to=prefix + "_alpha.pdf")
            logger.info("Override the default for --donor-tolerance with {0}".format(kwargs["donor_tolerance"]))

    for chromosome in chromosomes:
        for target_dict, chrom_coverage, sign, strand_short in zip((subpeak_pl_beds, subpeak_mn_beds),
                                                                   (chromosome_coverage_pl, chromosome_coverage_mn),
                                                                   ("+", "-"), ("pl", "mn")):
            target_dict[chromosome] = unified_element_definition(chrom_coverage, chromosome, sign,
                                                                 prefix + "_{0}.bed".format(strand_short),
                                                                 ce_donor=kwargs.get("donor_tolerance", 0.3),
                                                                 join_distance=kwargs.get("peak_distance", 1),
                                                                 ce_trigger=kwargs.get("ce_trigger", 3))

            if target_dict[chromosome] is not None:
                housekeeping_files.append(target_dict[chromosome])
                housekeeping_files.append(target_dict[chromosome] + ".csi")

    parse_input_files(output_dir=output_dir, output_prefix="ct_" + output_prefix, filters=filters,
                      pl_cov_target=input_coverage_pl, mn_cov_target=input_coverage_mn,
                      rc_target=ircs, bam_files=kwargs.get("ct_bam", None),
                      bw_pl_files=kwargs.get("ct_bw_pl", None), bw_mn_files=kwargs.get("ct_bw_mn", None), **kwargs)

    if len(ircs) > 0:
        if len(ircs) == 1 and len(rcs) > 1:
            logger.info("Only one input sample is provided, it will be shared among all treatment libraries")
            for _ in range(len(rcs) - len(ircs)):
                ircs.append(ircs[0])
                input_coverage_pl.append(input_coverage_pl[0])
                input_coverage_mn.append(input_coverage_mn[0])
        for i, (rc, irc), in enumerate(zip(rcs, ircs)):
            scale_factor = rc / irc
            logger.info("Adjusting signals based-on input/control (scale factor: %.4f)" % scale_factor)
            plc, mnc = normalize_using_input(chromosome_coverage_pl[i],
                                             chromosome_coverage_mn[i],
                                             input_coverage_pl[i],
                                             input_coverage_mn[i],
                                             scale_factor=scale_factor,
                                             output_dir=output_dir,
                                             output_prefix=output_prefix + "_inputnorm_%d" % i,
                                             logger=logger)
            chromosome_coverage_pl[i] = plc
            chromosome_coverage_mn[i] = mnc
            logger.info("Signals adjusted.")

            housekeeping_files.extend(plc.values())
            housekeeping_files.extend(mnc.values())

    fdr_target = kwargs.pop("fdr_target", 0.1)
    # peak calling (IQR)
    for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
        logger.info("Working on sample %d" % (rep + 1))
        sample_prefix = prefix + "_%d" % (rep + 1)
        df_dict = {}
        n_significant = 0
        for cov_dict, label, spb, strand_sign in zip(
                (pl_cov_dict, chromosome_coverage_mn[rep]),
                ("pl", "mn"), (subpeak_pl_beds, subpeak_mn_beds), ("+", "-")):
            if spb is None:
                continue
            peaks_bed = peaks_single_strand(per_base_cov=cov_dict,
                                            output_file=sample_prefix + "_{0}.bed".format(label),
                                            shared_peak_definitions=spb,
                                            strand_sign=strand_sign,
                                            fdr_target=fdr_target,
                                            **kwargs)

            peak_df = pd.read_csv(peaks_bed, sep="\t", header=None, names=COMMON_HEADER)
            peak_df = peak_df.loc[peak_df["end"] - peak_df["start"] < kwargs.get("window_size_threshold", 2000), :]
            n_significant += np.sum(peak_df["padj"] < fdr_target)
            df_dict[label] = peak_df

        logger.info("Pairing peaks")
        (bid, div, unid), gc = pair(sample_prefix, df_dict=df_dict, fdr_target=fdr_target,
                                    stringent_only=kwargs.get("stringent_only", False), **kwargs)
        # reformat unidirectional output
        unid_df = pd.read_csv(unid, sep="\t", names=COMMON_HEADER)
        unid_df[["chromosome", "start", "end", "name", "padj", "strand",
                 "reads", "summit", "summit_val"]].to_csv(
            unid, sep="\t", header=False, index=False)
        logger.info("Pairing finished on sample %d" % (rep + 1))
        for g in gc:
            housekeeping_files.append(g)

        epig_annotation = kwargs.get("epig_annotation", "")
        if epig_annotation is not None and epig_annotation != "":
            logger.info(f"Calling get_epig_annotation with {epig_annotation}, {prefix}")
            epig_annotation = get_epig_annotation(epig_annotation, prefix)

            # in situ update canonical peak calls
            annotate_tre_with_epig_info(bid, epig_annotation)
            annotate_tre_with_epig_info(div, epig_annotation)
            annotate_tre_with_epig_info(unid, epig_annotation)

            relaxed_fdr = kwargs.get("relaxed_fdr_target", fdr_target * 2)
            (r_bid, r_div, r_unid), r_gc = pair(sample_prefix, df_dict=df_dict, fdr_target=relaxed_fdr,
                                                stringent_only=kwargs.get("stringent_only", False),
                                                specific_suffix="_EA", **kwargs)
            r_unid_df = pd.read_csv(r_unid, sep="\t", names=COMMON_HEADER)
            r_unid_df[["chromosome", "start", "end", "name", "padj", "strand",
                       "reads", "summit", "summit_val"]].to_csv(
                r_unid, sep="\t", header=False, index=False)
            # in situ update relaxed peak calls
            annotate_tre_with_epig_info(r_bid, epig_annotation, only_annotated_records=True,
                                        placeholders={3: "Marginal"})
            annotate_tre_with_epig_info(r_div, epig_annotation, only_annotated_records=True,
                                        placeholders={3: "Marginal"})
            annotate_tre_with_epig_info(r_unid, epig_annotation, only_annotated_records=True)

            # merge results
            BedTool.cat(*[BedTool(bid), BedTool(r_bid).intersect(bid, v=True).saveas()], postmerge=False).sort().saveas(
                bid)
            BedTool.cat(*[BedTool(div), BedTool(r_div).intersect(div, v=True).saveas()], postmerge=False).sort().saveas(
                div)
            BedTool.cat(*[BedTool(unid), BedTool(r_unid).intersect(unid, v=True).saveas()],
                        postmerge=False).sort().saveas(
                unid)
            for g in r_gc:
                housekeeping_files.append(g)
            housekeeping_files.append(r_bid)
            housekeeping_files.append(r_div)
            housekeeping_files.append(r_unid)

        # on-the-fly QC
        if not disable_qc:
            on_the_fly_qc(
                sample_prefix,
                kwargs.get("min_mu_percent", 0.1),
                kwargs.get("top_peak_threshold", 0.75),
                n_significant,
                is_strict_mode=__STRICT_QC__)

        if kwargs.get("output_diagnostics", False):
            peak_bed_to_gtf(pl_df=df_dict["pl"], mn_df=df_dict["mn"],
                            save_to=sample_prefix + "_peaks.gtf", version=__version__)

    for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
        housekeeping_files.append(sample_prefix + "_pl.bed.gz")
        housekeeping_files.append(sample_prefix + "_pl.bed.gz.csi")
        housekeeping_files.append(sample_prefix + "_mn.bed.gz")
        housekeeping_files.append(sample_prefix + "_mn.bed.gz.csi")

        logger.info("Finished on sample %d" % (rep + 1))
        logger.info("Divergent peaks were saved to %s" % sample_prefix + "_divergent_peaks.bed")
        logger.info("Bidirectional peaks were saved to %s" % sample_prefix + "_bidirectional_peaks.bed")
        logger.info(
            "Significant peaks which failed to pair were saved to %s" % sample_prefix + "_unidirectional_peaks.bed")
    # delete intermediate files
    is_borrow_info_from_reps = kwargs.pop("borrow_info_reps", False)
    if is_borrow_info_from_reps and len(chromosome_coverage_pl) > 1:
        logger.info("Enhanced support for biological replicates is enabled.")
        inferring_elements_from_other_reps(prefix=prefix, n_samples=len(chromosome_coverage_pl))
    housekeeping(pybdt_tmp)
