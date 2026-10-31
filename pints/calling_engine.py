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
    from .io_engine import get_read_signal, get_coverage_bw, log_assert, normalize_using_input, is_zero_track, \
        npy_layout, npy_read, NPY_BLOCK, \
        index_bed_file, peak_bed_to_gtf, merge_replicates_bw, _read_signal, _coverage_bw, merge_replicate_tracks, \
        normalize_tracks_using_input
    from .coverage_store import register as register_track, lookup as lookup_track, release as release_tracks, \
        is_key
    from pints import __version__
except ImportError as e:
    missing_package = str(e).replace("No module named '", "").replace("'", "")
    logger.error("Please install %s first!" % missing_package)
    sys.exit(-1)

housekeeping_files = []
# Coverage lives in the in-memory store (pints/coverage_store.py, reached by registry keys) instead of per-base
# .npy dumps. PINTS_COVERAGE_STORE=0 uses the dumps (same results; kept for differential testing).
COVERAGE_STORE = os.environ.get("PINTS_COVERAGE_STORE", "1") != "0"
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


def window_sums(chromosome_coverage, window_size=100, step_size=100):
    """
    Vectorized equivalent of `sliding_window` for integer coverage arrays.

    Returns the same window read counts (same dtype as ``np.sum`` of an integer slice, i.e. the
    platform integer) and window start coordinates, without a Python-level loop.
    Floating-point input falls back to per-window ``np.sum`` so that summation order is unchanged.

    Returns
    -------
    sums : np.ndarray
        Read counts per window
    starts : np.ndarray
        0-based start coordinates of the windows (ends are ``starts + window_size``)
    """
    if step_size < 1:
        logger.error("step_size must >= 1")
        raise ValueError("step_size must >= 1")
    if len(chromosome_coverage) < 1:
        logger.error("chromosome_coverage must >= 1")
        raise ValueError("chromosome_coverage must >= 1")
    total_bins = int(np.floor(chromosome_coverage.shape[0] / step_size - window_size / step_size + 1))
    total_bins = max(total_bins, 0)
    starts = np.arange(total_bins, dtype=np.int64) * step_size
    cov = np.asarray(chromosome_coverage)
    if total_bins == 0:
        return np.zeros(0, dtype=np.sum(cov[:0]).dtype), starts
    if not np.issubdtype(cov.dtype, np.integer):
        sums = np.array([np.sum(cov[s:s + window_size]) for s in starts])
        return sums, starts
    acc = np.sum(cov[:0]).dtype  # accumulator dtype numpy uses for an integer sum (int64 / uint64)
    if step_size == window_size:
        sums = cov[:total_bins * window_size].reshape(total_bins, window_size).sum(axis=1, dtype=acc)
    else:
        cs = np.zeros(cov.shape[0] + 1, dtype=acc)
        np.cumsum(cov, dtype=acc, out=cs[1:])
        sums = cs[starts + window_size] - cs[starts]
    return sums, starts


def nonzero_window_regions(chromosome_coverage, window_size=100, step_size=100):
    """
    Regions formed by merging consecutive windows with positive read counts.

    Equivalent to building a bed of windows with ``reads > 0`` (from `sliding_window`) and running
    ``bedtools merge`` (overlapping / book-ended features are merged).

    Returns
    -------
    regions : list of tuples
        (start, end) of merged regions, sorted by start
    """
    sums, starts = window_sums(chromosome_coverage, window_size, step_size)
    keep = sums > 0
    if not keep.any():
        return []
    s = starts[keep]
    e = s + window_size
    # a new region starts when a window begins after the end of the previous window
    breaks = np.ones(s.shape[0], dtype=bool)
    if step_size >= window_size:
        breaks[1:] = s[1:] > e[:-1]
        idx = np.flatnonzero(breaks)
        ends = np.maximum.reduceat(e, idx)
        return [(int(a), int(b)) for a, b in zip(s[idx], ends)]
    # overlapping windows: generic interval merge
    return [tuple(x) for x in merge_intervals(list(zip(s.tolist(), e.tolist())))]


def check_window(coord_start, coord_end, mu_peak, var_peak, pi_peak, chromosome_coverage, peak_in_bg_threshold,
                 mu_bkg_minimum, sp_bed_handler, chromosome_name, fdr_target, cache, small_window_threshold=5,
                 flanking=(10000, 5000, 1000), disable_ler=False, enable_eler=True, top_peak_mu=None,
                 compute_lrt=True):
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
    compute_lrt : bool
        Whether to run the likelihood-ratio test. Its p-value is only consumed in sensitive mode,
        so callers can skip it otherwise (the returned lrt_pval is then a 1.0 placeholder).

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
        if compute_lrt:
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

    return pvalue, window_value, mu_0, pi_0, ler_counts, np.mean(lrt_pvals) if compute_lrt else 1., sig_dens


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


def merge_covs(covs, chromosome_of_interest, skip_if_empty=False):
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
    if skip_if_empty and all(is_zero_track(_resolve_coverage(cd[chromosome_of_interest])) for cd in covs):
        return None
    merged_coverage = None
    for cd in covs:
        ref = _resolve_coverage(cd[chromosome_of_interest])
        cov = np.load(ref) if isinstance(ref, str) else ref.dense()
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
              max_distance=30, peak_width=None):
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
    peak_width : int or None
        Required width of peaks in samples (`width` of scipy.signal.find_peaks, measured at peak_rel_height).
        None (default) applies no width requirement.
    Returns
    -------
    merged_intervals : list
        List of tuples of merged intervals [(start_1, end_1), ... , (start_n, end_n)]
    """
    if peak_width is None:
        peaks, _ = find_peaks(window, rel_height=peak_rel_height)
    else:
        peaks, _ = find_peaks(window, rel_height=peak_rel_height, width=peak_width)
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
    ctx = _subpeak_phase1(rc_file, output_file, strand_sign, chromosome_name, subpeak_file,
                          small_peak_threshold=small_peak_threshold, min_mu_percent=min_mu_percent,
                          disable_qc=disable_qc, enable_eler=enable_eler, eler_min=eler_min)
    bins = []
    if ctx["error"] is None:
        bins, err = _check_window_block(rc_file, chromosome_name, ctx["lines"], ctx["lines"], ctx,
                                        fdr_target, disable_ler, sensitive, fc_cutoff, init_dens_cutoff,
                                        init_height_cutoff,
                                        context_bounds=(ctx["index"].start_arr, ctx["index"].end_arr))
        if err is not None:
            logger.error(str(chromosome_name) + "\t" + str(subpeak_file))
            logger.error(err)
    return _bins_to_df(bins, strand_sign)

# ---------------------------------------------------------------------------------------------------
# Window evaluation engine (phase 1: subpeak statistics + thresholds; phase 2: local-environment tests)
#
# Phase 2 is sequential within a chromosome only through the LER cache, whose keys are (clipped)
# coordinates of subpeaks overlapping the flanks (<= max(flanking) // 2 bp) of the tested peak. Two
# runs of subpeaks separated by a gap of more than 2 * max flank can never share a cache key, so they
# can be evaluated independently (each with a fresh cache) and concatenated in file order. This is
# what allows splitting a chromosome into chunks without changing any output byte.
# ---------------------------------------------------------------------------------------------------
_FLANK_MAX_HALF = 10000 // 2
_SEGMENT_GAP = 15000  # > 2 * _FLANK_MAX_HALF, conservative
_MIN_P1_BLOCK = 500  # minimum subpeaks per phase-1 task (tests lower these to force splitting)
_MIN_P2_CHUNK = 200  # minimum subpeaks per phase-2 chunk
_RESULT_COLUMNS = ("chromosome", "start", "end", "name", "pval", "reads",
                   "mu_0", "pi_0", "mu_1", "pi_1", "var_1", "summit", "summit_val",
                   "ler_1", "ler_2", "ler_3", "non_zeros")


def _resolve_coverage(ref):
    """A coverage dict value as readers use it: the track for a registry key, otherwise the path itself."""
    return lookup_track(ref) if is_key(ref) else ref


def _register_coverage(coverage, sample, strand):
    """Register the tracks of a chromosome -> CoverageTrack dict; returns chromosome -> key (same order)."""
    return {c: register_track(sample, strand, c, t) for c, t in coverage.items()}


def _load_coverage(rc_file, lo=0, hi=None):
    """
    Read-only, page-cache-shared view of a coverage dump (same values/dtype as np.load). For a registry key,
    a fresh full-length array of the track's dense dtype holding its bases in [lo, hi) and zeros elsewhere:
    callers only read inside that span, and pages of `np.zeros` that are never written cost no memory.
    """
    if is_key(rc_file):
        track = lookup_track(rc_file)
        buf = track.empty_buffer()
        track.fill(buf, lo, hi)
        return buf
    try:
        return np.asarray(np.load(rc_file, mmap_mode="r"))
    except ValueError:
        return np.load(rc_file, allow_pickle=True)


# Phase 2 reads coverage at most _FLANK_MAX_HALF bases beyond a tested peak (check_window's flanks) and 2500
# bases beyond a subpeak overlapping those flanks (remove_peaks_in_local_env), and every such subpeak is in the
# task's context lines.
_PHASE2_REACH = _FLANK_MAX_HALF + 2500 + 2


class SubpeakIndex:
    """
    In-memory stand-in for ``pysam.TabixFile`` over one chromosome's subpeak bed.

    ``fetch`` reproduces tabix semantics for 0-based half-open BED records: records with
    ``start < query_end`` and ``end > query_start`` are returned in file order; an empty query
    (``query_end <= query_start``) returns nothing. Records must be sorted by start and have positive
    length (`is_supported`); otherwise callers keep using tabix.

    `starts`/`ends` (the lines' start and end columns, e.g. from the phase-1 worker that wrote them) spare
    parsing the lines. The main process only needs the arrays (support check, chunking, contexts), so the
    split fields and list copies that `fetch` uses are built on the first query.
    """

    def __init__(self, chromosome, lines, starts=None, ends=None):
        from bisect import bisect_left, bisect_right
        self._bl, self._br = bisect_left, bisect_right
        self.chromosome = chromosome
        self.lines = lines
        if starts is None:
            starts, ends = _line_bounds(lines)
        self.start_arr = np.asarray(starts, dtype=np.int64)
        self.end_arr = np.asarray(ends, dtype=np.int64)
        self.prefix_max_end_arr = np.maximum.accumulate(self.end_arr)
        self._supported = None
        self._query = None

    def is_supported(self):
        if self._supported is None:
            st, en = self.start_arr, self.end_arr
            # strictly increasing starts: then file order is the unique (chrom, start)-sorted order that
            # `bedtools sort` (unstable for ties) produces for the tabix file
            self._supported = bool((en > st).all() and (st[1:] > st[:-1]).all())
        return self._supported

    def _query_state(self):
        if self._query is None:
            self._query = ([ln.split("\t") for ln in self.lines], self.start_arr.tolist(), self.end_arr.tolist(),
                           self.prefix_max_end_arr.tolist())
        return self._query

    def indices(self, start, end):
        if end <= start:
            return range(0)
        _, starts, ends, prefix_max_end = self._query_state()
        hi = self._bl(starts, end)
        lo = self._br(prefix_max_end, start)
        return [i for i in range(lo, hi) if ends[i] > start]

    def overlapping(self, start, end):
        """`indices` as an int array, without building the query state (sorted starts only)."""
        if end <= start:
            return np.empty(0, dtype=np.intp)
        hi = int(np.searchsorted(self.start_arr, end, side="left"))
        lo = int(np.searchsorted(self.prefix_max_end_arr, start, side="right"))
        return np.flatnonzero(self.end_arr[lo:hi] > start) + lo

    def fetch(self, chromosome, start, end, parser=None):
        if chromosome != self.chromosome:
            raise ValueError("could not create iterator for region '%s:%d-%d'" % (chromosome, start + 1, end))
        if start < 0:
            raise ValueError("start out of range (%d)" % start)
        f = self._query_state()[0]
        return [f[i] for i in self.indices(start, end)]


def _line_bounds(lines):
    """Start and end columns of BED lines, as int64 arrays."""
    fields = [ln.split("\t", 3) for ln in lines]
    return (np.array([int(f[1]) for f in fields], dtype=np.int64),
            np.array([int(f[2]) for f in fields], dtype=np.int64))


class PeakBedIndex:
    """
    In-memory stand-in for ``pysam.TabixFile(path).fetch(chrom, start, end, parser=pysam.asTuple())``
    over a bgzipped BED file written by `stratified_filtering` (sorted by chromosome, then start).

    Lines are read from the very file that tabix indexes, so file order (including ties in start) is
    the order tabix returns. ``fetch`` returns the tab-split fields of records with
    ``start < query_end`` and ``end > query_start`` in file order; argument errors and unknown
    chromosomes raise ValueError and empty regions return nothing, as with pysam. `open_peak_index`
    only uses this class when every record has positive length, starts are non-decreasing within each
    chromosome, each chromosome is one contiguous block, and there are no header/comment lines or
    carriage returns; otherwise it falls back to tabix.
    """

    def __init__(self, blocks):
        from bisect import bisect_left, bisect_right
        self._bl, self._br = bisect_left, bisect_right
        self._blocks = blocks  # chromosome -> (lines, starts, ends, prefix_max_end)

    @classmethod
    def from_file(cls, path):
        """Return a PeakBedIndex, or None when the file is outside the supported subset."""
        blocks = {}
        chrom, lines, starts, ends, pme, m = None, None, None, None, None, -1
        with gzip.open(path, "rt", newline="") as fh:  # bgzip output is multi-member gzip
            for line in fh:
                if not line.endswith("\n") or "\r" in line or line.startswith("#"):
                    return None
                parts = line.split("\t", 3)
                if len(parts) < 4:
                    return None
                if not (parts[1].isascii() and parts[1].isdigit() and parts[2].isascii() and parts[2].isdigit()):
                    return None
                s, e = int(parts[1]), int(parts[2])
                if parts[0] != chrom:
                    if parts[0] in blocks:
                        return None  # chromosome not contiguous
                    chrom, lines, starts, ends, pme, m = parts[0], [], [], [], [], -1
                    blocks[chrom] = (lines, starts, ends, pme)
                elif s < starts[-1]:
                    return None
                if e <= s or s < 0:
                    return None
                lines.append(line)
                starts.append(s)
                ends.append(e)
                m = e if e > m else m
                pme.append(m)
        return cls(blocks)

    def fetch(self, chromosome, start, end, parser=None):
        # argument checks in pysam's order: end, start, start > end, empty region, then the contig
        if end < 0:
            raise ValueError("end out of range (%d)" % end)
        if start < 0:
            raise ValueError("start out of range (%d)" % start)
        if start > end:
            raise ValueError("start (%d) >= end (%d)" % (start, end))
        if start == end:
            return []
        block = self._blocks.get(chromosome)
        if block is None:
            raise ValueError("could not create iterator for region '%s:%d-%d'" % (chromosome, start + 1, end))
        lines, starts, ends, pme = block
        hi = self._bl(starts, end)
        lo = self._br(pme, start)
        return [lines[i][:-1].split("\t") for i in range(lo, hi) if ends[i] > start]


_PEAK_INDEX_BYTES_PER_QUERY = 2000


def open_peak_index(path, n_queries=None):
    """
    `PeakBedIndex` for `path` when supported and worthwhile, else a pysam.TabixFile over the same file.

    Building the index costs ~1.2 us per record and a tabix query ~0.15 ms, so tabix is kept when there
    are few queries relative to the file (compressed size is ~15-22 bytes per record). Either object
    returns the same records, so this choice affects speed only.
    """
    if n_queries is None or n_queries * _PEAK_INDEX_BYTES_PER_QUERY >= os.path.getsize(path):
        idx = PeakBedIndex.from_file(path)
        if idx is not None:
            return idx
    return pysam.TabixFile(path, index=path + ".csi")


def _subpeak_block_stats(rc_file, chromosome_name, strand_sign, raw_lines, small_peak_threshold):
    """Phase-1 worker: per-subpeak statistics, formatted exactly as the subpeak bed lines, and their bounds."""
    if is_key(rc_file):  # each subpeak reads its own bases only
        starts, ends = _line_bounds(raw_lines)
        per_base_cov = _load_coverage(rc_file, int(starts.min()), int(ends.max())) if len(raw_lines) else None
    else:
        per_base_cov = _load_coverage(rc_file)
    out_lines = []
    starts, ends = [], []
    all_peak_mus = []
    try:
        for line in raw_lines:
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

            out_lines.append("%s\t%d\t%d\t%s\t%f\t%s\t%s\t%f\t%d\t%d\t%d\n" % (
                chromosome_name, start, end, items[3], mu_peak, var_peak, strand_sign, pi_peak, non_zero_loci,
                summit_coord, peak_region[x]))
            starts.append(start)
            ends.append(end)
    except TypeError as ex:
        return out_lines, all_peak_mus, ex, _bounds(starts, ends)
    return out_lines, all_peak_mus, None, _bounds(starts, ends)


def _bounds(starts, ends):
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def _read_subpeak_input(subpeak_file):
    opener = gzip.open if subpeak_file.endswith(".gz") else open  # plain from the pipeline's element definition
    with opener(subpeak_file, "rt") as peak_obj:
        return peak_obj.readlines()


def _subpeak_thresholds(output_file, chromosome_name, subpeak_file, out_lines, all_peak_mus, error,
                        min_mu_percent=0.1, disable_qc=False, enable_eler=True, eler_min=1., indexer=None,
                        bounds=(None, None)):
    """Phase-1 reduce: derive the per-chromosome thresholds, and write + index the subpeak bed when needed.

    Phase 2 reads the subpeak bed through `ctx["index"]` (a `SubpeakIndex` over `out_lines`). Only when that
    index does not support the lines (ties, unsorted or empty records) is the bed written, compressed and
    indexed for the tabix fallback, as before. `indexer`, if given, is called instead of `index_bed_file`
    (used to compress/index on the pool while the parent continues). `bounds` are the lines' start and end
    columns, if known.
    """
    subpeak_bed = output_file.replace(".bed", __SUB_PEAK_TPL__ % chromosome_name)
    ctx = {"lines": out_lines, "error": None, "chromosome": chromosome_name, "subpeak_file": subpeak_file,
           "subpeak_bed": subpeak_bed}
    if error is not None:
        ctx["error"] = error
        logger.error(str(chromosome_name) + "\t" + str(subpeak_file))
        logger.error(error)
        return ctx
    index = SubpeakIndex(chromosome_name, out_lines, *bounds)
    ctx["index"] = index
    try:
        if not index.is_supported():
            with open(subpeak_bed, "w") as spb_fh:
                spb_fh.writelines(out_lines)
            if indexer is None:
                index_bed_file(subpeak_bed, logger=logger)
            else:
                indexer(subpeak_bed)
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
    except TypeError as ex:
        ctx["error"] = ex
        logger.error(str(chromosome_name) + "\t" + str(subpeak_file))
        logger.error(ex)
        return ctx
    ctx.update(peak_threshold=peak_threshold, bkg_mu_threshold=bkg_mu_threshold,
               empirical_true_peak_threshold=empirical_true_peak_threshold, enable_eler=enable_eler)
    return ctx


def _subpeak_phase1(rc_file, output_file, strand_sign, chromosome_name, subpeak_file, small_peak_threshold=5,
                    min_mu_percent=0.1, disable_qc=False, enable_eler=True, eler_min=1.):
    raw = _read_subpeak_input(subpeak_file)
    out_lines, mus, err, bounds = _subpeak_block_stats(rc_file, chromosome_name, strand_sign, raw,
                                                       small_peak_threshold)
    return _subpeak_thresholds(output_file, chromosome_name, subpeak_file, out_lines, mus, err,
                               min_mu_percent=min_mu_percent, disable_qc=disable_qc, enable_eler=enable_eler,
                               eler_min=eler_min, bounds=bounds)


def _check_window_block(rc_file, chromosome_name, peak_lines, context_lines, ctx, fdr_target, disable_ler=False,
                        sensitive=False, fc_cutoff=1.5, init_dens_cutoff=0.25, init_height_cutoff=4,
                        chromosome_coverage=None, context_bounds=(None, None)):
    """
    Phase-2 worker: evaluate `peak_lines` (consecutive subpeak bed lines) with a fresh LER cache.

    `context_lines` must contain every subpeak overlapping the flanks of the tested peaks (in file
    order); `context_bounds`, if given, are their start and end columns. Returns (bins, error) where error is a
    caught TypeError (as in the original loop) or None.
    """
    if chromosome_coverage is not None:
        per_base_cov = chromosome_coverage
    elif is_key(rc_file):
        ctx_starts, ctx_ends = context_bounds if context_bounds[0] is not None else _line_bounds(context_lines)
        if len(ctx_starts):
            per_base_cov = _load_coverage(rc_file, int(ctx_starts.min()) - _PHASE2_REACH,
                                          int(ctx_ends.max()) + _PHASE2_REACH)
        else:
            per_base_cov = _load_coverage(rc_file, 0, 0)
    else:
        per_base_cov = _load_coverage(rc_file)
    if fc_cutoff > 0:
        real_fc_cutoff = 1. / fc_cutoff
    else:
        real_fc_cutoff = 1.
    index = SubpeakIndex(chromosome_name, context_lines, *context_bounds)
    if index.is_supported():
        bed_handler = index
    else:  # ties / unsorted / empty records: use the tabix file itself
        bed_handler = pysam.TabixFile(ctx["subpeak_bed"] + ".gz", index=ctx["subpeak_bed"] + ".gz.csi")
    peak_threshold = ctx["peak_threshold"]
    bkg_mu_threshold = ctx["bkg_mu_threshold"]
    enable_eler = ctx["enable_eler"]
    empirical_true_peak_threshold = ctx["empirical_true_peak_threshold"]
    bins = []
    global_cache = {}
    try:
        for peak in peak_lines:
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
                enable_eler=enable_eler, top_peak_mu=empirical_true_peak_threshold,
                compute_lrt=sensitive)
            if wv > 0:
                use_lrt = sensitive and pi_peak > 0. and pi_bg > 0. and (
                        mu_bg / mu_peak < real_fc_cutoff) and sig_dens > init_dens_cutoff and \
                          peak_summit_val > init_height_cutoff
                bins.append(
                    (chromosome_name, peak_start, peak_end, peak_id, lrtp if use_lrt else pval,
                     wv, mu_bg, pi_bg, mu_peak, pi_peak, peak_var, peak_summit, peak_summit_val,
                     lerc[0], lerc[1], lerc[2], peak_non_zeros))
    except TypeError as ex:
        return bins, ex
    return bins, None


def _bins_to_df(bins, strand_sign):
    result_df = pd.DataFrame(bins, columns=_RESULT_COLUMNS)
    result_df["strand"] = strand_sign
    result_df = result_df.loc[:, ("chromosome", "start", "end", "name", "pval", "strand", "reads",
                                  "mu_0", "pi_0", "mu_1", "pi_1", "var_1",
                                  "summit", "summit_val", "ler_1", "ler_2", "ler_3", "non_zeros")]
    return result_df


def _segment_chunks(lines, target, index=None):
    """
    Split subpeak lines (file order) into chunks of whole independent segments (gap >= _SEGMENT_GAP)
    with roughly `target` lines each. Returns list of (lo, hi) line-index ranges. `index`, a SubpeakIndex
    over `lines`, supplies the bounds without parsing the lines again.
    """
    if target <= 0 or len(lines) <= target:
        return [(0, len(lines))]
    if index is None:
        index = SubpeakIndex(None, lines)
    # line i can start a chunk if it starts >= _SEGMENT_GAP after the end of every earlier line
    gaps = np.flatnonzero(index.start_arr[1:] - index.prefix_max_end_arr[:-1] >= _SEGMENT_GAP) + 1
    chunks = []
    lo = 0
    for i in gaps.tolist():
        if i - lo >= target:
            chunks.append((lo, i))
            lo = i
    chunks.append((lo, len(lines)))
    return chunks


def _context_for(index, lines, lo, hi):
    """Lines overlapping the flanks of lines[lo:hi] (superset of anything LER may query), and their bounds."""
    q0 = max(int(index.start_arr[lo:hi].min()) - _FLANK_MAX_HALF - 2, 0)
    q1 = int(index.prefix_max_end_arr[hi - 1]) + _FLANK_MAX_HALF + 2
    sel = index.overlapping(q0, q1)
    bounds = (index.start_arr[sel], index.end_arr[sel])
    if sel.shape[0] and sel[-1] - sel[0] + 1 == sel.shape[0]:
        return lines[sel[0]:sel[-1] + 1], bounds
    return [lines[i] for i in sel.tolist()], bounds


def _worker_init(model_name, iqr_name):
    """Pool initializer: make the statistical model/IQR strategy available in every worker (spawn-safe)."""
    global stat_tester, iqr_obj
    from . import stats_engine as _se
    stat_tester = getattr(_se, model_name)() if model_name else None
    iqr_obj = getattr(_se, iqr_name)() if iqr_name else None


def _make_pool(n):
    import multiprocessing as mp
    from .stats_engine import numba_warmup
    numba_warmup()
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - platforms without fork
        ctx = mp.get_context()
    init_args = (type(stat_tester).__name__ if stat_tester is not None else None,
                 type(iqr_obj).__name__ if iqr_obj is not None else None)
    # The workers' cyclic GC never visits objects that exist at the fork, so it doesn't write to (and copy) the
    # parent's pages in every worker; the parent collects them again after the fork.
    import gc
    gc.collect()
    gc.freeze()
    try:
        return ctx.Pool(n, initializer=_worker_init, initargs=init_args)
    finally:
        gc.unfreeze()


def _run_tasks(pool, func, tasks, costs=None):
    """Run star-tasks (ordered results) on `pool` (or inline when pool is None), longest first."""
    if pool is None or len(tasks) <= 1:
        return [func(*t) for t in tasks]
    order = list(range(len(tasks)))
    if costs is not None:
        order.sort(key=lambda i: -costs[i])
    out = [None] * len(tasks)
    res = pool.starmap(_indexed_call, [(i, func, tasks[i]) for i in order], chunksize=1)
    for i, r in res:
        out[i] = r
    return out


def _indexed_call(i, func, args):
    return i, func(*args)


def _index_bed_task(path):
    index_bed_file(path, logger=logger)


def _pair_task(sample_prefix, df_dict, fdr_target, stringent_only, kwargs):
    """`pair` + the unidirectional-output reformat of `peak_calling`, for one sample (pool task)."""
    (bid, div, unid), gc = pair(sample_prefix, df_dict=df_dict, fdr_target=fdr_target,
                                stringent_only=stringent_only, **kwargs)
    unid_df = pd.read_csv(unid, sep="\t", names=COMMON_HEADER)
    unid_df[["chromosome", "start", "end", "name", "padj", "strand",
             "reads", "summit", "summit_val"]].to_csv(
        unid, sep="\t", header=False, index=False)
    return (bid, div, unid), tuple(gc)  # gc is a locally defined namedtuple (not picklable); only iterated


def _stratified_filtering_task(tmp_df, output_file, fdr_target, kwargs):
    return stratified_filtering(tmp_df=tmp_df, output_file=output_file, fdr_target=fdr_target, **kwargs)


_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
_BGZF_BLOCK = 0xff00  # uncompressed bytes per BGZF block, as in htslib
_PARALLEL_WRITE_MIN_ROWS = 1  # empty tables are written by `stratified_filtering` itself


def bgzf_compress(data, level=6):
    """BGZF blocks (no EOF marker) holding `data`; any concatenation of these, plus _BGZF_EOF, is a BGZF file."""
    import struct
    import zlib
    out = []
    for a in range(0, len(data), _BGZF_BLOCK):
        chunk = data[a:a + _BGZF_BLOCK]
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        cdata = c.compress(chunk) + c.flush()
        if len(cdata) > 65536 - 26:  # incompressible: store
            c = zlib.compressobj(0, zlib.DEFLATED, -15)
            cdata = c.compress(chunk) + c.flush()
        out.append(struct.pack("<4BI2BH2BHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, len(cdata) + 25))
        out.append(cdata)
        out.append(struct.pack("<2I", zlib.crc32(chunk), len(chunk)))
    return b"".join(out)


def _filter_frame_task(tmp_df, output_file, fdr_target, kwargs):
    """
    Pool task: the stratified-filtering table, returned for `_ParallelIndexedBed` to write when the file can be
    produced without bedtools sort, else written and indexed here exactly as `stratified_filtering` does.

    index_bed_file sorts with `bedtools sort` (chromosome, then start; ties in no guaranteed order). The table
    is already sorted by the same keys, so when no (chromosome, start) pair repeats, the sorted file is the
    table as written.
    """
    result_df = _stratified_frame(tmp_df, output_file, fdr_target, **kwargs)
    if result_df.shape[0] >= _PARALLEL_WRITE_MIN_ROWS and not result_df.duplicated(["chromosome", "start"]).any():
        return "frame", result_df
    result_df.to_csv(output_file, sep="\t", index=False, header=False)
    index_bed_file(output_file, logger=logger)
    return "path", output_file + ".gz"


def _bed_chunk_task(chunk):
    """Pool task: rows of the filtering table as `to_csv` writes them, BGZF-compressed."""
    return bgzf_compress(chunk.to_csv(sep="\t", index=False, header=False).encode("utf-8"))


def _bed_assemble_task(path, parts):
    """Pool task: concatenate BGZF parts into `path` and build its CSI index (same index as index_bed_file)."""
    with open(path, "wb") as fh:
        for part in parts:
            fh.write(part)
        fh.write(_BGZF_EOF)
    pysam.tabix_index(path, preset="bed", force=True, csi=True)
    return path


class _ParallelIndexedBed:
    """
    Stand-in for the AsyncResult of `_stratified_filtering_task`: get() returns the same .gz path, holding the
    same lines. Filtering runs on one worker, then the table is formatted and compressed in row chunks across
    the pool and indexed. Each step is submitted from the previous one's callback, so strands overlap.
    """

    def __init__(self, pool, tmp_df, output_file, fdr_target, kwargs, n_chunks):
        import threading
        self._pool, self._gz, self._n_chunks = pool, output_file + ".gz", max(1, n_chunks)
        self._done = threading.Event()
        self._value = self._error = None
        pool.apply_async(_filter_frame_task, (tmp_df, output_file, fdr_target, kwargs),
                         callback=self._filtered, error_callback=self._fail)

    def _filtered(self, res):
        try:
            kind, payload = res
            if kind == "path":
                return self._finish(payload)
            n = payload.shape[0]
            step = -(-n // self._n_chunks)
            chunks = [(payload.iloc[a:a + step],) for a in range(0, n, step)]
            self._pool.starmap_async(_bed_chunk_task, chunks, chunksize=1,
                                     callback=self._compressed, error_callback=self._fail)
        except Exception as ex:
            self._fail(ex)

    def _compressed(self, parts):
        try:
            self._pool.apply_async(_bed_assemble_task, (self._gz, parts),
                                   callback=self._finish, error_callback=self._fail)
        except Exception as ex:
            self._fail(ex)

    def _finish(self, path):
        self._value = path
        self._done.set()

    def _fail(self, ex):
        self._error = ex
        self._done.set()

    def get(self):
        self._done.wait()
        if self._error is not None:
            raise self._error
        return self._value


def _phase(label, t0=[None]):
    """Wall-clock phase markers on stderr when PINTS_PHASE_TIMING is set (diagnostics only)."""
    import time
    now = time.perf_counter()
    if label is not None and os.environ.get("PINTS_PHASE_TIMING"):
        sys.stderr.write("[phase] %-28s %8.2fs\n" % (label, now - (t0[0] if t0[0] is not None else now)))
    t0[0] = now


def _element_definition_task(chrom_coverage, chromosome, sign, output_file, ce_donor, join_distance, ce_trigger,
                             peak_width=None):
    return unified_element_definition(chrom_coverage, chromosome, sign, output_file, ce_donor=ce_donor,
                                      join_distance=join_distance, ce_trigger=ce_trigger, peak_width=peak_width,
                                      compress=False)


def _coverage_cost(coverage_dicts, chromosome):
    """Scheduling weight of a chromosome: total on-disk size of its coverage tracks (dense size for tracks)."""
    total = 0
    for d in coverage_dicts:
        f = d.get(chromosome)
        if is_key(f):
            t = lookup_track(f)
            total += t.length * t.dtype.itemsize
        elif isinstance(f, str) and os.path.exists(f):
            total += os.path.getsize(f)
    return total


def evaluate_strand_jobs(jobs, fdr_target, pool=None, thread_n=1, **kwargs):
    """
    Evaluate candidate peaks for several (sample, strand) jobs at once.

    Parameters
    ----------
    jobs : list of tuples
        (per_base_cov, output_file, shared_peak_definitions, strand_sign) as in `peaks_single_strand`
    fdr_target : float
    pool : multiprocessing pool or None
    thread_n : int
        Used for chunk sizing only

    Returns
    -------
    sub_dfs : list of lists of pd.DataFrame
        For every job, the per-chromosome result frames (in per_base_cov order), identical to the
        frames produced by `check_window_chromosome`.
    """
    small_peak_threshold = kwargs.get("small_peak_threshold", 5)
    min_mu_percent = kwargs.get("min_mu_percent", 0.1)
    disable_qc = kwargs.get("disable_qc", False)
    disable_ler = kwargs.get("disable_ler", False)
    enable_eler = kwargs.get("enable_eler", True)
    eler_min = kwargs.get("eler_lower_bound", 1.)
    sensitive = kwargs.get("sensitive", False)
    fc_cutoff = kwargs.get("fc_cutoff", 1.5)
    init_dens_cutoff = kwargs.get("init_dens_cutoff", 0.25)
    init_height_cutoff = kwargs.get("init_height_cutoff", 4)

    units = []  # (job index, chromosome, rc_file, output_file, strand, subpeak_file)
    for j, (per_base_cov, output_file, spd, strand_sign) in enumerate(jobs):
        for chrom, pbc_npy in per_base_cov.items():
            if spd[chrom] is None:
                continue
            units.append((j, chrom, pbc_npy, output_file, strand_sign, spd[chrom]))

    # ---- phase 1: per-subpeak statistics, in blocks
    raws = [_read_subpeak_input(u[5]) for u in units]
    total = sum(len(r) for r in raws)
    workers = max(1, thread_n if pool is not None else 1)
    block = max(_MIN_P1_BLOCK, total // (workers * 4) + 1) if pool is not None else 0
    p1_tasks, p1_owner = [], []
    for ui, (u, raw) in enumerate(zip(units, raws)):
        if block <= 0 or len(raw) <= block:
            ranges = [(0, len(raw))]
        else:
            ranges = [(a, min(a + block, len(raw))) for a in range(0, len(raw), block)]
        for a, b in ranges:
            p1_tasks.append((u[2], u[1], u[4], raw[a:b], small_peak_threshold))
            p1_owner.append(ui)
    _phase("phase1 setup")
    p1_res = _run_tasks(pool, _subpeak_block_stats, p1_tasks, costs=[len(t[3]) for t in p1_tasks])
    _phase("phase1 stats")

    ctxs = []
    pending_index = []
    indexer = None
    if pool is not None:
        def indexer(path):
            pending_index.append(pool.apply_async(_index_bed_task, (path,)))
    for ui, u in enumerate(units):
        out_lines, mus, err, starts, ends = [], [], None, [], []
        for owner, (ol, m, e, (st, en)) in zip(p1_owner, p1_res):
            if owner != ui or err is not None:
                continue
            out_lines.extend(ol)
            mus.extend(m)
            starts.append(st)
            ends.append(en)
            err = e
        bounds = (np.concatenate(starts), np.concatenate(ends)) if starts else (None, None)
        ctxs.append(_subpeak_thresholds(u[3], u[1], u[5], out_lines, mus, err, min_mu_percent=min_mu_percent,
                                        disable_qc=disable_qc, enable_eler=enable_eler, eler_min=eler_min,
                                        indexer=indexer, bounds=bounds))

    _phase("phase1 thresholds")
    # ---- phase 2: local-environment tests, in independent chunks
    total2 = sum(len(c["lines"]) for c in ctxs if c["error"] is None)
    target = max(_MIN_P2_CHUNK, total2 // (workers * 8) + 1) if pool is not None else 0
    p2_tasks, p2_owner = [], []
    for ui, (u, c) in enumerate(zip(units, ctxs)):
        if c["error"] is not None:
            continue
        lines = c["lines"]
        index = c["index"]
        if not index.is_supported():  # tabix fallback reads the compressed subpeak bed
            for r in pending_index:
                r.get()
            pending_index = []
        slim = {k: v for k, v in c.items() if k not in ("lines", "index")}
        for lo, hi in (_segment_chunks(lines, target, index) if index.is_supported() else [(0, len(lines))]):
            if lo == 0 and hi == len(lines):
                context, bounds = lines, (index.start_arr, index.end_arr)
            else:
                context, bounds = _context_for(index, lines, lo, hi)
            p2_tasks.append((u[2], u[1], lines[lo:hi], context, slim, fdr_target, disable_ler, sensitive,
                             fc_cutoff, init_dens_cutoff, init_height_cutoff, None, bounds))
            p2_owner.append(ui)
    _phase("phase2 setup")
    if pool is not None:
        start_deferred_releases()  # the tests below write no files
    p2_res = _run_tasks(pool, _check_window_block, p2_tasks, costs=[len(t[2]) for t in p2_tasks])
    _phase("phase2 tests")
    for r in pending_index:
        r.get()

    per_unit_bins = [[] for _ in units]
    stopped = [False] * len(units)
    for ui, (bins, err) in zip(p2_owner, p2_res):
        if stopped[ui]:
            continue
        per_unit_bins[ui].extend(bins)
        if err is not None:  # original loop stops at the first TypeError of a chromosome
            stopped[ui] = True
            logger.error(str(units[ui][1]) + "\t" + str(units[ui][5]))
            logger.error(err)

    sub_dfs = [[] for _ in jobs]
    for ui, u in enumerate(units):
        sub_dfs[u[0]].append(_bins_to_df(per_unit_bins[ui], u[4]))
    return sub_dfs


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
    result_df = _stratified_frame(tmp_df, output_file, fdr_target, dry_run=dry_run, **kwargs)
    result_df.to_csv(output_file, sep="\t", index=False, header=False)
    index_bed_file(output_file, logger=logger)
    return output_file + ".gz"


def _stratified_frame(tmp_df, output_file, fdr_target, dry_run=False, **kwargs):
    """The table `stratified_filtering` writes (same operations), sorted by chromosome and start."""
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
            if tmp_df.shape[0] > 0 and tmp_df["reads"].dtype.kind in "iuf":
                # vectorised form of `cached_pois`: identical poisson.sf element-wise evaluations
                lengths = tmp_df["end"] - tmp_df["start"]
                small = (lengths <= small_peak_threshold).to_numpy()
                pv = tmp_df["pval"].to_numpy(dtype=np.float64, copy=True)
                if small.any():
                    pv[small] = poisson.sf(tmp_df["reads"].to_numpy()[small],
                                           (lamb_global * lengths).to_numpy()[small])
                tmp_df["pval"] = pv
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
    result_df = result_df.sort_values(by=['chromosome', 'start'])
    return result_df


def peaks_single_strand(per_base_cov, output_file, shared_peak_definitions, strand_sign, fdr_target,
                        precomputed=None, defer_filtering=False, **kwargs):
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

    if precomputed is not None:
        sub_dfs = precomputed
    elif kwargs.get("thread_n", 1) == 1:
        sub_dfs = evaluate_strand_jobs([(per_base_cov, output_file, shared_peak_definitions, strand_sign)],
                                       fdr_target, pool=None, **kwargs)[0]
    else:
        with _make_pool(kwargs.get("thread_n", 1)) as pool:
            sub_dfs = evaluate_strand_jobs([(per_base_cov, output_file, shared_peak_definitions, strand_sign)],
                                           fdr_target, pool=pool, **kwargs)[0]

    sub_dfs = [sdf for sdf in sub_dfs if sdf is not None]
    if defer_filtering and len(sub_dfs) == 0:
        return None  # let the caller re-run inline so the failure surfaces at the original point
    log_assert(len(sub_dfs) > 0, "No signal found across all chromosomes!", logger)
    tmp_df = _concat_result_frames(sub_dfs)

    if kwargs.get("output_diagnostics", False):
        tmp_df.to_csv(output_file.replace(".bed", "_debug.csv"), index=False)

    if defer_filtering:
        return tmp_df
    return stratified_filtering(tmp_df=tmp_df, output_file=output_file, fdr_target=fdr_target, **kwargs)


def _concat_result_frames(sub_dfs):
    """
    Concatenate per-chromosome result frames, leaving out empty ones.

    A chromosome without any testable peak yields an empty frame whose columns are all object dtype.
    Concatenating it with the other frames upcasts every integer column (start, end, reads, ...) to
    object, which makes the quantiles in `independent_filtering` run on object arrays (tens of times
    slower). Empty frames contribute no rows, so leaving them out changes only the dtypes, not the
    values or their order.
    """
    non_empty = [sdf for sdf in sub_dfs if sdf.shape[0] > 0]
    return pd.concat(non_empty if non_empty else sub_dfs)


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
    with open(sig_peak_bed, "r") as sig_fh:
        n_queries = sum(1 for _ in sig_fh)
    tbx = open_peak_index(peak_candidate_bed, n_queries=n_queries)
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


_background_deletions = []
_deferred_deletions = []


def release_coverage_files(old_dicts, live_dicts=(), defer=False):
    """
    Delete, on a background thread, the coverage dumps in `old_dicts` that no dict in `live_dicts` still
    uses, and drop the in-memory tracks of `old_dicts` (registry keys) that no dict in `live_dicts` uses. They stay in `housekeeping_files`; `housekeeping` waits for the thread before its own pass.

    Parameters
    ----------
    old_dicts : iterable of dict
        chromosome -> .npy path maps that are no longer needed
    live_dicts : iterable of dict
        maps that are still in use (their paths are kept)
    defer : bool
        Only queue the files; `start_deferred_releases` deletes them. Unlinking tens of GB of cached dumps
        slows down processes that write files at the same time, so the pool path starts the deletion when
        the workers only compute.
    """
    old_dicts, live_dicts = list(old_dicts), list(live_dicts)
    live_keys = {k for d in live_dicts for k in d.values() if is_key(k)}
    release_tracks({k for d in old_dicts for k in d.values() if is_key(k)} - live_keys)  # in this process
    live = {os.path.abspath(f) for d in live_dicts for f in d.values() if isinstance(f, str)}
    paths = sorted({f for d in old_dicts for f in d.values()
                    if isinstance(f, str) and f.endswith(".npy") and os.path.abspath(f) not in live})
    if defer:
        _deferred_deletions.extend(paths)
    else:
        _remove_in_background(paths)


def start_deferred_releases():
    """Start deleting the files queued by `release_coverage_files(..., defer=True)`."""
    paths = list(_deferred_deletions)
    del _deferred_deletions[:]
    _remove_in_background(paths)


def _remove_in_background(paths):
    import threading

    def _remove():
        for f in paths:
            try:
                os.remove(f)
            except OSError:
                pass

    if paths:
        t = threading.Thread(target=_remove, name="pints-release", daemon=True)
        t.start()
        _background_deletions.append(t)


def housekeeping(pybedtools_prefix=""):
    """
    Delete intermediate files

    Returns
    -------

    """
    global housekeeping_files
    del _deferred_deletions[:]  # still in housekeeping_files
    while _background_deletions:
        _background_deletions.pop().join()
    try:
        for hf in housekeeping_files:
            if isinstance(hf, str) and os.path.exists(hf):
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


def _release_free_heap():
    """Return freed heap memory to the OS (glibc only; no-op elsewhere).

    The block buffers are a few tens of MB; once glibc's dynamic mmap threshold has grown, such buffers come
    from the heap and stay mapped in the (long-lived) worker after they are freed.
    """
    global _MALLOC_TRIM
    if _MALLOC_TRIM is None:
        try:
            import ctypes
            _MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _MALLOC_TRIM = False
    if _MALLOC_TRIM:
        _MALLOC_TRIM(0)


_MALLOC_TRIM = None


def _blockwise_element_definition(coverage_dict, chromosome_of_interest, subpeak_bed, join_distance, ce_donor,
                                  ce_trigger, block=None, compress=True, peak_width=None):
    """Block-wise form of the body of `unified_element_definition` (same file, same return value).

    The merged int32 track is built one block at a time (``zeros(int32)`` then ``+=`` per replicate, the same
    element-wise operations as `merge_covs`) instead of for the whole chromosome. Windows never straddle a
    block (blocks are multiples of the window and start at multiples of it), and a region that touches the
    end of a block is carried into the next buffer, so regions, sub-peaks and summits are exactly those of
    the whole-chromosome computation. Returns False when the inputs are not plain 1-D integer .npy files of
    one common length (the caller then runs the original code).
    """
    window = 100
    block = NPY_BLOCK if block is None else int(block)
    block = max(window, block // window * window)
    tracks = []
    n = None
    for cd in coverage_dict:
        path = _resolve_coverage(cd[chromosome_of_interest])
        lay = npy_layout(path) if isinstance(path, str) else (None, path.dtype, path.length)
        if lay is None or lay[1].kind not in "biu":
            return False
        if n is None:
            n = lay[2]
        elif lay[2] != n:
            return False
        if not is_zero_track(path):  # adding an all-zero track is a no-op
            tracks.append((path, lay))
    if n is None or n < 1:
        return False

    fhs = [open(p, "rb") if isinstance(p, str) else p for p, _ in tracks]  # tracks: windows instead of reads
    spb_fh = None
    index = 1
    try:
        def merged_slice(a, b):
            m = np.zeros(b - a, dtype=np.int32)
            for fh, (_, lay) in zip(fhs, tracks):
                m += npy_read(fh, lay, a, b) if lay[0] is not None else fh.window(a, b)
            return m

        buf_start = 0
        buf = None
        pos = 0
        while pos < n:
            nxt = min(n, pos + block)
            if n - nxt < window:  # final buffer extends to the end of the chromosome (partial window included)
                nxt = n
            fresh = merged_slice(pos, nxt)
            buf = fresh if buf is None else np.concatenate((buf, fresh))
            pos = nxt
            final = pos >= n
            regions = nonzero_window_regions(buf) if buf.shape[0] >= window else []
            carry_from = None
            if regions and not final and regions[-1][1] == buf.shape[0]:
                carry_from = regions.pop()[0]  # may continue in the next block
            for row_start, row_end in regions:
                if spb_fh is None:
                    spb_fh = open(subpeak_bed, "w")
                sub_peaks = cut_peaks(buf[row_start:row_end],
                                      donor_tolerance=ce_donor,
                                      ce_trigger=ce_trigger,
                                      join_distance=join_distance,
                                      peak_width=peak_width)
                for sp in sub_peaks:
                    start = sp[0] + row_start
                    end = sp[1] + row_start
                    if end <= buf.shape[0] or final:
                        peak_region = buf[start:end]
                    else:
                        peak_region = merged_slice(buf_start + start, min(n, buf_start + end))
                    summit_coord = start + np.argmax(peak_region)
                    spb_fh.write("%s\t%d\t%d\t%s-%d\t%d\n" % (
                        chromosome_of_interest, start + buf_start, end + buf_start, chromosome_of_interest,
                        index, summit_coord + buf_start))
                    index += 1
            if carry_from is None:
                buf = None
                buf_start = pos
            else:
                buf = buf[carry_from:].copy()
                buf_start += carry_from
    finally:
        for (p, _), fh in zip(tracks, fhs):
            if isinstance(p, str):
                fh.close()
        if spb_fh is not None:
            spb_fh.close()
        buf = fresh = None
        _release_free_heap()

    if index > 1:
        if not compress:
            return subpeak_bed
        index_bed_file(subpeak_bed, logger=logger)
        return subpeak_bed + ".gz"
    return None


def unified_element_definition(coverage_dict, chromosome_of_interest, strand_sign, output_file,
                               join_distance=1, ce_donor=1.0, ce_trigger=3, peak_width=None,
                               compress=True):
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
    compress : bool
        Compress and index the subpeak bed (returns the .gz path). The pipeline passes False: phase 1 only
        reads the file from start to end, so it is left plain (same lines).

    Returns
    -------
    subpeak_bed : str or None
        file name for the subpeak bed file, or None if there's no peak
    """
    subpeak_bed = output_file.replace(".bed", __SUB_PEAK_TPL__ % chromosome_of_interest)
    if all(is_zero_track(_resolve_coverage(cd[chromosome_of_interest])) for cd in coverage_dict):
        return None
    blockwise = _blockwise_element_definition(coverage_dict, chromosome_of_interest, subpeak_bed,
                                              join_distance, ce_donor, ce_trigger, compress=compress,
                                              peak_width=peak_width)
    if blockwise is not False:
        return blockwise
    merged_coverage = merge_covs(coverage_dict, chromosome_of_interest, skip_if_empty=True)
    if merged_coverage is None:  # all tracks are known to be empty for this chromosome
        return None

    # windows with reads, merged when book-ended (same as bedtools merge on the window bed)
    regions = nonzero_window_regions(merged_coverage)
    if len(regions) == 0:  # no hit
        return None

    with open(subpeak_bed, "w") as spb_fh:
        index = 1
        for row_start, row_end in regions:
            sub_peaks = cut_peaks(merged_coverage[row_start:row_end],
                                  donor_tolerance=ce_donor,
                                  ce_trigger=ce_trigger,
                                  join_distance=join_distance,
                                  peak_width=peak_width)
            for sp in sub_peaks:
                start = sp[0] + row_start
                end = sp[1] + row_start
                peak_region = merged_coverage[start:end]
                summit_coord = start + np.argmax(peak_region)
                spb_fh.write("%s\t%d\t%d\t%s-%d\t%d\n" % (chromosome_of_interest, start, end, chromosome_of_interest,
                                                          index, summit_coord))
                index += 1

    if index > 1:
        if not compress:
            return subpeak_bed
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


def _highlight_chromosome(kwargs):
    """`--focused-chrom` is stored as `focused_chrom` by the CLI; `highlight_chromosome` by the API."""
    return kwargs.get("highlight_chromosome", kwargs.get("focused_chrom", "chr1"))


def _stringent_only(kwargs):
    """`--stringent-pairs-only` is stored as `stringent_pairs_only` by the CLI; `stringent_only` by the API."""
    return bool(kwargs.get("stringent_only", kwargs.get("stringent_pairs_only", False)))


def _significant_bins(peak_df, fdr_target, kwargs):
    """The rows of one strand's peak table that `pair` writes as its significant peaks."""
    peak_df = peak_df.copy()
    peak_df = peak_df.loc[peak_df["end"] - peak_df["start"] < kwargs.get("window_size_threshold", 2000), :]

    sig_bins = peak_df.loc[peak_df["padj"] < fdr_target, :]
    if kwargs.get("keep_sticks", True):
        return sig_bins
    return sig_bins.loc[sig_bins.non_zeros > 1, :]


def _pair_strand(sample_prefix, label, anti_label, sig_bins, fdr_target, stringent_only, specific_suffix, kwargs):
    """One strand of `pair`: write its significant peaks and pair them with the opposite strand's candidates.
    Returns the intermediate files `pair` records for this strand."""
    generated_files = []
    generated_files.append(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label))
    with open(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label), "w") as f:
        sig_bins.to_csv(f, sep="\t", index=False, header=False)

    if not os.path.exists(sample_prefix + __SIG_PEAK_TPL__ % (specific_suffix, label)):
        return generated_files + ["", "", ""]
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
    return generated_files


def _bed_is_empty(path):
    """`BedTool(path).count() == 0`, stopping at the first feature (same rules for header and blank lines)."""
    try:
        next(iter(BedTool(path)))
    except StopIteration:
        return True
    return False


def pair(sample_prefix, df_dict, fdr_target, stringent_only, specific_suffix="", pool=None, **kwargs):
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
    pool : multiprocessing pool or None
        If given, the two strands are paired with the opposite strand's peaks on it, in parallel

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
    strands = [(sample_prefix, label, anti_label, _significant_bins(df_dict[label], fdr_target, kwargs), fdr_target,
                stringent_only, specific_suffix, kwargs) for label, anti_label in zip(("pl", "mn"), ("mn", "pl"))]
    if pool is None:
        generated_files = sum((_pair_strand(*a) for a in strands), [])
    else:  # the strands only read each other's candidate files, written before pairing
        generated_files = sum((r.get() for r in [pool.apply_async(_pair_strand, a) for a in strands]), [])

    for directionality in ("bidirectional", "divergent", "singletons"):
        exp_pl_file = sample_prefix + "%s_sig_pl_%s_peaks.bed" % (specific_suffix, directionality)
        exp_mn_file = sample_prefix + "%s_sig_mn_%s_peaks.bed" % (specific_suffix, directionality)
        if os.path.exists(exp_pl_file) and os.path.exists(exp_mn_file):
            if directionality != "singletons":
                tmp_pl_bed = BedTool(exp_pl_file)
                tmp_mn_bed = BedTool(exp_mn_file)
                pl_empty, mn_empty = _bed_is_empty(exp_pl_file), _bed_is_empty(exp_mn_file)
                if pl_empty and mn_empty:
                    continue
                elif pl_empty or mn_empty:
                    # only one strand produced pairs: merge it on its own (previously the raw 16-column
                    # records were parsed with 8 names, giving a malformed BED)
                    pri_merged_file = (tmp_mn_bed if pl_empty else tmp_pl_bed).sort().merge(
                        c=(12, 13, 14, 15, 16), o=("collapse", "collapse", "collapse", "collapse", "distinct"))
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
                fwd_tss_cols = pri_merged_df["tss_fwd"].str.split(",", expand=True).fillna(value="-1").astype(int)
                fwd_tss_val_cols = pri_merged_df["tss_fwd_vals"].str.split(",", expand=True).fillna(value="-1").astype(
                    int)
                max_index = fwd_tss_val_cols.idxmax(axis=1)
                pri_merged_df["tss_fwd"] = fwd_tss_cols.values[np.arange(fwd_tss_cols.shape[0]), max_index]
                pri_merged_df["tss_fwd_val"] = fwd_tss_val_cols.values[np.arange(fwd_tss_cols.shape[0]), max_index]

                rev_tss_cols = pri_merged_df["tss_rev"].str.split(",", expand=True).fillna(value="-1").astype(int)
                rev_tss_val_cols = pri_merged_df["tss_rev_vals"].str.split(",", expand=True).fillna(value="-1").astype(
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


def _add_coverage(pl_cov_target, mn_cov_target, rc_target, plc, mnc, rc, cached, sample):
    """Append one sample's coverage: registry keys for tracks (its reused .npy cache is deleted at the end, as
    dumps are), else the dump paths, which are deleted at the end."""
    if COVERAGE_STORE:
        plc, mnc = _register_coverage(plc, sample, "pl"), _register_coverage(mnc, sample, "mn")
        housekeeping_files.extend(cached)
    else:
        housekeeping_files.extend(plc.values())
        housekeeping_files.extend(mnc.values())
    pl_cov_target.append(plc)
    mn_cov_target.append(mnc)
    rc_target.append(rc)


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
            plc, mnc, rc, cached = _read_signal(input_bam=bf,
                                                loc_prime=kwargs["bam_parser"],
                                                chromosome_startswith=kwargs.pop("chromosome_startswith", ""),
                                                reverse_complement=kwargs.get("seq_rc", False),
                                                output_dir=output_dir,
                                                output_prefix=output_prefix + "_%d" % i,
                                                filters=filters, as_tracks=COVERAGE_STORE,
                                                **kwargs)
            _add_coverage(pl_cov_target, mn_cov_target, rc_target, plc, mnc, rc, cached, output_prefix + "_%d" % i)
            logger.info("{0} loaded.".format(bf))
    elif bw_pl_files is not None and bw_mn_files is not None:
        log_assert(len(bw_pl_files) == len(bw_mn_files),
                   "Must provide the same amount of bigwig files for both strands", logger)

        for i, bw_pl in enumerate(bw_pl_files):
            log_assert(os.path.exists(bw_pl), "Cannot find bigwig file %s" % bw_pl, logger)
            log_assert(os.path.exists(bw_mn_files[i]), "Cannot find bigwig file %s" % bw_mn_files[i], logger)
            logger.info("Loading {0} and {1}...".format(bw_pl, bw_mn_files[i]))
            plc, mnc, rc, cached = _coverage_bw(bw_pl=bw_pl, bw_mn=bw_mn_files[i],
                                                chromosome_startswith=kwargs.get("chromosome_startswith", ""),
                                                output_dir=output_dir,
                                                output_prefix=output_prefix + "_%d" % i,
                                                threads=kwargs.get("thread_n", 1), as_tracks=COVERAGE_STORE)
            _add_coverage(pl_cov_target, mn_cov_target, rc_target, plc, mnc, rc, cached, output_prefix + "_%d" % i)
            logger.info("{0} and {1} loaded.".format(bw_pl, bw_mn_files[i]))

    if merge_replicates and len(pl_cov_target) > 1 and COVERAGE_STORE:
        pl_covs, mn_covs, pl_rc, mn_rc = merge_replicate_tracks(
            [{c: lookup_track(k) for c, k in d.items()} for d in pl_cov_target],
            [{c: lookup_track(k) for c, k in d.items()} for d in mn_cov_target])
        pl_covs = _register_coverage(pl_covs, output_prefix + "_merged", "pl")
        mn_covs = _register_coverage(mn_covs, output_prefix + "_merged", "mn")
        release_coverage_files(pl_cov_target + mn_cov_target, (pl_covs, mn_covs))
        pl_cov_target[:] = [pl_covs]
        mn_cov_target[:] = [mn_covs]
        rc_target[:] = [pl_rc + mn_rc, ]
        logger.info("Replicates merged")
    elif merge_replicates and len(pl_cov_target) > 1:
        pl_covs, mn_covs, pl_rc, mn_rc = merge_replicates_bw(pl_cov_target, mn_cov_target, output_dir,
                                                             output_prefix + "_merged",
                                                             threads=kwargs.get("thread_n", 1))
        # the per-replicate dumps are not read again once merged
        release_coverage_files(pl_cov_target + mn_cov_target, (pl_covs, mn_covs),
                               defer=kwargs.get("thread_n", 1) > 1)
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
    _phase(None)  # start the phase clock
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
        highlight_chromosome = _highlight_chromosome(kwargs)
        log_assert(os.path.exists(kwargs.get("annotation_gtf", None)), "Cannot find gene annotation file", logger)
        if highlight_chromosome in chromosome_coverage_pl[0]:
            kwargs["donor_tolerance"] = cut_peaks_dry_run(kwargs.get("annotation_gtf", None), chromosome_coverage_pl,
                                                          chromosome_coverage_mn,
                                                          tss_extension=kwargs.get("tss_extension", 200),
                                                          highlight_chromosome=highlight_chromosome,
                                                          output_diagnostics=kwargs.get("output_diagnostics", False),
                                                          save_to=prefix + "_alpha.pdf")
            logger.info("Override the default for --donor-tolerance with {0}".format(kwargs["donor_tolerance"]))

    thread_n = kwargs.get("thread_n", 1)
    pool = None

    def control_and_normalise():
        """Parse the control and normalise the samples against it; returns the un-normalised sample dicts."""
        pre_norm = chromosome_coverage_pl + chromosome_coverage_mn
        parse_input_files(output_dir=output_dir, output_prefix="ct_" + output_prefix, filters=filters,
                          pl_cov_target=input_coverage_pl, mn_cov_target=input_coverage_mn,
                          rc_target=ircs, bam_files=kwargs.get("ct_bam", None),
                          bw_pl_files=kwargs.get("ct_bw_pl", None), bw_mn_files=kwargs.get("ct_bw_mn", None),
                          **kwargs)

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
                if COVERAGE_STORE:
                    plc, mnc = normalize_tracks_using_input(
                        *[{c: lookup_track(k) for c, k in d.items()} for d in (
                            chromosome_coverage_pl[i], chromosome_coverage_mn[i], input_coverage_pl[i],
                            input_coverage_mn[i])], scale_factor=scale_factor, logger=logger)
                    plc = _register_coverage(plc, output_prefix + "_inputnorm_%d" % i, "pl")
                    mnc = _register_coverage(mnc, output_prefix + "_inputnorm_%d" % i, "mn")
                else:
                    plc, mnc = normalize_using_input(chromosome_coverage_pl[i],
                                                     chromosome_coverage_mn[i],
                                                     input_coverage_pl[i],
                                                     input_coverage_mn[i],
                                                     scale_factor=scale_factor,
                                                     output_dir=output_dir,
                                                     output_prefix=output_prefix + "_inputnorm_%d" % i,
                                                     logger=logger, threads=thread_n)
                    housekeeping_files.extend(plc.values())
                    housekeeping_files.extend(mnc.values())
                chromosome_coverage_pl[i] = plc
                chromosome_coverage_mn[i] = mnc
                logger.info("Signals adjusted.")
        return pre_norm

    # element definition reads the un-normalised sample coverage
    ued_coverage_pl, ued_coverage_mn = list(chromosome_coverage_pl), list(chromosome_coverage_mn)
    if COVERAGE_STORE:
        # pool workers reach tracks only through the registry they inherit when the pool forks, so the control
        # and the normalised tracks are built first
        _phase("input parsing")
        pre_norm = control_and_normalise()
        release_coverage_files(input_coverage_pl + input_coverage_mn,
                               chromosome_coverage_pl + chromosome_coverage_mn + pre_norm)
        _phase("control parsing+normalisation")

    # one persistent worker pool for all parallel stages (None -> everything runs inline, as before)
    pool = _make_pool(thread_n) if thread_n > 1 else None

    ued_keys, ued_tasks, ued_costs = [], [], []
    for chromosome in chromosomes:
        for target_dict, chrom_coverage, sign, strand_short in zip((subpeak_pl_beds, subpeak_mn_beds),
                                                                   (ued_coverage_pl, ued_coverage_mn),
                                                                   ("+", "-"), ("pl", "mn")):
            ued_keys.append((target_dict, chromosome))
            ued_tasks.append((chrom_coverage, chromosome, sign, prefix + "_{0}.bed".format(strand_short),
                              kwargs.get("donor_tolerance", 0.3), kwargs.get("peak_distance", 1),
                              kwargs.get("ce_trigger", 3), kwargs.get("peak_width", None)))
            ued_costs.append(_coverage_cost(chrom_coverage, chromosome))
    if not COVERAGE_STORE:
        _phase("input parsing")
    ued_res = _run_tasks(pool, _element_definition_task, ued_tasks, costs=ued_costs)
    _phase("element definition")
    # results are consumed in the original (chromosome, strand) order
    for (target_dict, chromosome), res in zip(ued_keys, ued_res):
        target_dict[chromosome] = res
        if res is not None:
            housekeeping_files.append(res)
            housekeeping_files.append(res + ".csi")

    _phase("subpeak bookkeeping")
    if COVERAGE_STORE:
        release_coverage_files(ued_coverage_pl + ued_coverage_mn, chromosome_coverage_pl + chromosome_coverage_mn)
    else:
        pre_norm = control_and_normalise()
        if len(ircs) > 0:
            # un-normalised experiment and control dumps are not read again
            release_coverage_files(pre_norm + input_coverage_pl + input_coverage_mn,
                                   chromosome_coverage_pl + chromosome_coverage_mn, defer=pool is not None)
        _phase("control parsing+normalisation")
    fdr_target = kwargs.pop("fdr_target", 0.1)
    # window evaluation for every sample x strand at once (shared pool, longest tasks first)
    precomputed = {}
    if pool is not None:
        eval_jobs, eval_keys = [], []
        for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
            sample_prefix = prefix + "_%d" % (rep + 1)
            for cov_dict, label, spb, strand_sign in zip(
                    (pl_cov_dict, chromosome_coverage_mn[rep]),
                    ("pl", "mn"), (subpeak_pl_beds, subpeak_mn_beds), ("+", "-")):
                if spb is None:
                    continue
                eval_jobs.append((cov_dict, sample_prefix + "_{0}.bed".format(label), spb, strand_sign))
                eval_keys.append((rep, label))
        for key, res in zip(eval_keys, evaluate_strand_jobs(eval_jobs, fdr_target, pool=pool, **kwargs)):
            precomputed[key] = res
    _phase("input normalisation/eval")
    # stratified filtering (BH/independent filtering, sort, bgzip/tabix) of every sample x strand runs on the
    # pool; the per-sample pairing below consumes the results in the original order
    deferred_filtering = {}
    if pool is not None:
        for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
            sample_prefix = prefix + "_%d" % (rep + 1)
            for cov_dict, label, spb, strand_sign in zip(
                    (pl_cov_dict, chromosome_coverage_mn[rep]),
                    ("pl", "mn"), (subpeak_pl_beds, subpeak_mn_beds), ("+", "-")):
                if spb is None:
                    continue
                out_file = sample_prefix + "_{0}.bed".format(label)
                tmp_df = peaks_single_strand(per_base_cov=cov_dict, output_file=out_file,
                                             shared_peak_definitions=spb, strand_sign=strand_sign,
                                             fdr_target=fdr_target, precomputed=precomputed.get((rep, label)),
                                             defer_filtering=True, **kwargs)
                if tmp_df is not None:
                    deferred_filtering[(rep, label)] = _ParallelIndexedBed(pool, tmp_df, out_file, fdr_target,
                                                                           kwargs, n_chunks=thread_n)
                    precomputed.pop((rep, label), None)
    # pairing of every sample on the pool (only for the plain path: no epigenomic annotation / diagnostics,
    # which re-use df_dict afterwards); results are consumed below in sample order
    deferred_pairing = {}
    epig_requested = kwargs.get("epig_annotation", "") not in (None, "")
    all_deferred = all((rep, label) in deferred_filtering
                       for rep in range(len(chromosome_coverage_pl)) for label in ("pl", "mn"))
    if pool is not None and all_deferred:
        # every strand is evaluated and its filtering dispatched: nothing reads the coverage dumps again
        start_deferred_releases()
        release_coverage_files(chromosome_coverage_pl + chromosome_coverage_mn)
    if pool is not None and len(chromosome_coverage_pl) > 1 and all_deferred and not epig_requested \
            and not kwargs.get("output_diagnostics", False):
        for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
            sample_prefix = prefix + "_%d" % (rep + 1)
            df_dict = {}
            n_significant = 0
            for label in ("pl", "mn"):
                peaks_bed = deferred_filtering.pop((rep, label)).get()
                peak_df = pd.read_csv(peaks_bed, sep="\t", header=None, names=COMMON_HEADER)
                peak_df = peak_df.loc[peak_df["end"] - peak_df["start"] < kwargs.get("window_size_threshold", 2000),
                                      :]
                n_significant += np.sum(peak_df["padj"] < fdr_target)
                df_dict[label] = peak_df
            deferred_pairing[rep] = (n_significant, pool.apply_async(
                _pair_task, (sample_prefix, df_dict, fdr_target, _stringent_only(kwargs), kwargs)))
    # peak calling (IQR)
    for rep, pl_cov_dict in enumerate(chromosome_coverage_pl):
        logger.info("Working on sample %d" % (rep + 1))
        sample_prefix = prefix + "_%d" % (rep + 1)
        df_dict = {}
        n_significant = 0
        if rep in deferred_pairing:
            n_significant, pair_res = deferred_pairing.pop(rep)
            logger.info("Pairing peaks")
            (bid, div, unid), gc = pair_res.get()
            logger.info("Pairing finished on sample %d" % (rep + 1))
            for g in gc:
                housekeeping_files.append(g)
            if not disable_qc:
                on_the_fly_qc(
                    sample_prefix,
                    kwargs.get("min_mu_percent", 0.1),
                    kwargs.get("top_peak_threshold", 0.75),
                    n_significant,
                    is_strict_mode=__STRICT_QC__)
            continue
        for cov_dict, label, spb, strand_sign in zip(
                (pl_cov_dict, chromosome_coverage_mn[rep]),
                ("pl", "mn"), (subpeak_pl_beds, subpeak_mn_beds), ("+", "-")):
            if spb is None:
                continue
            if (rep, label) in deferred_filtering:
                peaks_bed = deferred_filtering.pop((rep, label)).get()
            else:
                peaks_bed = peaks_single_strand(per_base_cov=cov_dict,
                                                output_file=sample_prefix + "_{0}.bed".format(label),
                                                shared_peak_definitions=spb,
                                                strand_sign=strand_sign,
                                                fdr_target=fdr_target,
                                                precomputed=precomputed.get((rep, label)),
                                                **kwargs)

            peak_df = pd.read_csv(peaks_bed, sep="\t", header=None, names=COMMON_HEADER)
            peak_df = peak_df.loc[peak_df["end"] - peak_df["start"] < kwargs.get("window_size_threshold", 2000), :]
            n_significant += np.sum(peak_df["padj"] < fdr_target)
            df_dict[label] = peak_df

        logger.info("Pairing peaks")
        (bid, div, unid), gc = pair(sample_prefix, df_dict=df_dict, fdr_target=fdr_target,
                                    stringent_only=_stringent_only(kwargs), pool=pool, **kwargs)
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
                                                stringent_only=_stringent_only(kwargs),
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
    _phase("filtering+pairing+qc")
    if pool is not None:
        pool.close()
        pool.join()
    housekeeping(pybdt_tmp)
