#!/usr/bin/env python
# coding=utf-8

#  PINTS: Peak Identifier for Nascent Transcript Starts
#  Copyright (C) 2019-2025 Yu Lab.
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
import pysam
import sys
import os
import logging
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import reduce

NP_DT_RANGE = (127, 32767, 2147483647, 9223372036854775807)
NP_DT_NAME = (np.int8, np.int16, np.int32, np.int64)
GTF_HEADER = ["seqname", "source", "feature", "start", "end", "score",
              "strand", "frame"]
__logger_name__ = "PINTS.IO-engine"

# Paths of per-base coverage dumps (.npy) that are known to contain only zeros. They are written as
# sparse files (no data blocks) and downstream steps can skip loading them. The registry is a pure
# optimisation: consumers that do not consult it simply load the (all-zero) arrays as before.
_ZERO_TRACKS = set()


def save_zero_track(path, length, dtype=np.int8):
    """Write an all-zero 1-D array as a sparse .npy file (same header as np.save) and register it."""
    if not path.endswith(".npy"):
        path = path + ".npy"
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(int(length),))
    del mm
    _ZERO_TRACKS.add(os.path.abspath(path))
    return path


def is_zero_track(path):
    """True if `path` was written by this process (or its parent) as an all-zero coverage track."""
    return isinstance(path, str) and os.path.abspath(path) in _ZERO_TRACKS


def _smallest_int_dtype(max_value):
    """Same dtype choice as the NP_DT_RANGE loops used when dumping coverage arrays."""
    for k, v in enumerate(NP_DT_RANGE):
        if v > max_value:
            return NP_DT_NAME[k]
    return None


def _bam_contig_worker(task):
    """
    Count 5'/3' ends for one contig (exactly the per-read logic of BamParsers) and dump the npys.

    Returns (contig, pl_path, mn_path, total_reads, n_records, pl_is_zero, mn_is_zero)
    """
    (input_bam, contig, length, mapq, layout, end, rc, pl_fn, mn_fn) = task
    bam = pysam.AlignmentFile(input_bam, "rb")
    pl_pos = []
    mn_pos = []
    total = 0
    n_rec = 0
    skip_r1 = layout == "R2"
    skip_r2 = layout == "R1"
    five = end == "5"
    for read in bam.fetch(contig):
        n_rec += 1
        if skip_r2 and read.is_read2:
            continue
        if skip_r1 and read.is_read1:
            continue
        if read.mapq < mapq:
            continue
        total += 1
        rev = read.is_reverse
        if five:
            pos = read.reference_end - 1 if rev else read.reference_start
        else:
            pos = read.reference_start if rev else read.reference_end - 1
        if rev != rc:  # minus strand (after optional reverse complement)
            mn_pos.append(pos)
        else:
            pl_pos.append(pos)
    bam.close()

    out = []
    for positions, fn in ((pl_pos, pl_fn), (mn_pos, mn_fn)):
        if len(positions) == 0:
            out.append((save_zero_track(fn, length), True))
            continue
        uniq, cnt = np.unique(np.asarray(positions, dtype=np.int64), return_counts=True)
        if uniq[0] < 0 or uniq[-1] >= length:
            raise IndexError("index %d is out of bounds for axis 0 with size %d" % (
                uniq[0] if uniq[0] < 0 else uniq[-1], length))
        # original accumulates in uint32 before choosing the dump dtype
        cnt = cnt.astype(np.uint32)
        dt = _smallest_int_dtype(cnt.max())
        arr = np.zeros(length, dtype=dt)
        arr[uniq] = cnt.astype(dt)
        np.save(fn, arr)
        del arr
        out.append((fn + ".npy", False))
    return contig, out[0][0], out[1][0], total, n_rec, out[0][1], out[1][1]


def parse_bam_by_contig(input_bam, contigs, mapq, layout, end, rc, output_dir, output_prefix, threads=1):
    """
    Index-based, contig-parallel equivalent of BamParsers + dtype conversion in get_read_signal.

    Parameters
    ----------
    contigs : list of (name, length) in header order (already filtered)

    Returns
    -------
    (result_pl, result_mn, total_reads) or None if the fast path cannot guarantee identical results
    (no index, or index statistics do not match the records fetched), in which case callers must
    fall back to the serial parser.
    """
    logger = logging.getLogger(__logger_name__)
    try:
        bam = pysam.AlignmentFile(input_bam, "rb")
        if not bam.has_index():
            bam.close()
            return None
        stats = {s.contig: s.total for s in bam.get_index_statistics()}
        bam.close()
    except Exception as ex:  # pragma: no cover - defensive
        logger.debug("Contig-parallel BAM parsing unavailable (%s)" % ex)
        return None

    result_pl = {}
    result_mn = {}
    tasks = []
    for name, length in contigs:
        pl_fn = os.path.join(output_dir, output_prefix + "_pl_%s" % name)
        mn_fn = os.path.join(output_dir, output_prefix + "_mn_%s" % name)
        if stats.get(name, None) == 0:
            result_pl[name] = save_zero_track(pl_fn, length)
            result_mn[name] = save_zero_track(mn_fn, length)
        else:
            tasks.append((input_bam, name, length, mapq, layout, end, rc, pl_fn, mn_fn))
    # longest-first scheduling
    tasks.sort(key=lambda t: -stats.get(t[1], 0))
    if threads > 1 and len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(min(threads, len(tasks))) as pool:
            results = pool.map(_bam_contig_worker, tasks, chunksize=1)
    else:
        results = [_bam_contig_worker(t) for t in tasks]

    total_reads = 0
    ok = True
    for contig, pl_path, mn_path, total, n_rec, pl_zero, mn_zero in results:
        if stats.get(contig, None) != n_rec:
            ok = False
        total_reads += total
        result_pl[contig] = pl_path
        result_mn[contig] = mn_path
        if pl_zero:
            _ZERO_TRACKS.add(os.path.abspath(pl_path))
        if mn_zero:
            _ZERO_TRACKS.add(os.path.abspath(mn_path))
    if not ok:
        logger.warning("BAM index statistics do not match fetched records; using the serial parser")
        for d in (result_pl, result_mn):
            for v in d.values():
                _ZERO_TRACKS.discard(os.path.abspath(v))
                if os.path.exists(v):
                    os.remove(v)
        return None
    # keep header order in the returned dicts
    result_pl = {name: result_pl[name] for name, _ in contigs}
    result_mn = {name: result_mn[name] for name, _ in contigs}
    return result_pl, result_mn, total_reads


def log_assert(bool_, message, logger):
    """
    Assert function, result will be wrote to logger

    Parameters
    ----------
    bool_ : bool expression
        Condition which should be true
    message : str
        This message will show up when bool_ is False
    logger : logger
        Specific logger

    Returns
    -------

    """
    try:
        assert bool_, message
    except AssertionError as err:
        logger.error("%s" % err)
        sys.exit(1)


def determine_data_type(max_value):
    data_type = np.int32
    for k, v in enumerate(NP_DT_RANGE):
        if v > max_value:
            return NP_DT_NAME[k]
    return data_type


def _warn_fractional(values, bw_file, chromosome):
    """PINTS models integer read counts; non-integer bigWig values are truncated by the integer cast.

    Logs a warning (results are unchanged) so that normalised (CPM/RPM) tracks are not used silently.
    Returns True if fractional values were found.
    """
    v = np.asarray(values)
    if v.dtype.kind not in "fc":
        return False
    v = v[np.isfinite(v)]
    if np.any(v != np.trunc(v)):
        logging.getLogger(__logger_name__).warning(
            "BigWig %s (%s) contains non-integer (float) values; PINTS expects raw read counts and truncates them "
            "to integers. Please provide unnormalised bigWig files." % (bw_file, chromosome))
        return True
    return False


def _get_coverage_bw(bw_file, chromosome_startswith, output_dir, output_prefix):
    """
    Get coverage information from bigwig files

    Parameters
    ----------
    bw_file : str
        Path to bigwig file
    chromosome_startswith : str
         Filter out reads whose reference don't start with chromosome_startswith
    output_dir : str
        Path to write outputs
    output_prefix : str
        Prefix of outputs
    kwargs :

    Returns
    -------

    """
    logger = logging.getLogger(__logger_name__)
    try:
        import pyBigWig
    except ImportError as e:
        missing_package = str(e).replace("No module named '", "").replace("'", "")
        logger.error("Please install %s first!" % missing_package)
        sys.exit(-1)
    bw = pyBigWig.open(bw_file)
    log_assert(bw.isBigWig(), "BigWig file %s is not valid." % bw_file, logger)
    chromosome_coverage = {}
    chromosome_pre_accessible_dict = {}
    chromosome_reads_dict = {}

    bw_max = max(abs(bw.header()["maxVal"]), abs(bw.header()["minVal"]))
    data_type = determine_data_type(bw_max)

    chrom_size_list = []
    read_counts = 0

    for chromosome, csize in bw.chroms().items():
        if chromosome.startswith(chromosome_startswith):
            fn = os.path.join(output_dir, "%s_%s" % (output_prefix, chromosome))
            chromosome_coverage[chromosome] = fn + ".npy"
            if not os.path.exists(chromosome_coverage[chromosome]):
                if pyBigWig.numpy:
                    _chromosome_cov = np.nan_to_num(bw.values(chromosome, 0, csize, numpy=True))
                else:
                    _chromosome_cov = np.nan_to_num(bw.values(chromosome, 0, csize))
                _warn_fractional(_chromosome_cov, bw_file, chromosome)
                _chromosome_cov[_chromosome_cov < 0] *= -1
                _chromosome_cov = _chromosome_cov.astype(data_type, copy=False)
                np.save(fn, _chromosome_cov)
            else:
                _chromosome_cov = np.load(chromosome_coverage[chromosome])
            read_counts += np.sum(_chromosome_cov)
            chrom_size_list.append((chromosome, csize))
            chromosome_pre_accessible_dict[chromosome] = []
            chromosome_reads_dict[chromosome] = 0
    
    bw.close()
    return chromosome_coverage, read_counts


def get_coverage_bw(bw_pl, bw_mn, chromosome_startswith, output_dir, output_prefix):
    """
    Get 5' coverage

    Parameters
    ----------
    bw_pl : str
        Full path to input bigwig file, plus strand
    bw_mn : str
        Full path to input bigwig file, minus strand
    chromosome_startswith : str
        Filter out reads whose reference don't start with chromosome_startswith
    output_dir : str
        Path to write outputs
    output_prefix : str
        Prefix of outputs
    **kwargs


    Returns
    -------
    pl : dict
        Dictionary of per base coverage per chromosome (positive strand)
    mn : dict
        Dictionary of per base coverage per chromosome (negative strand)
    rc : int
        Number of total read counts
    """
    logger = logging.getLogger(__logger_name__)
    pl, pc = _get_coverage_bw(bw_pl, chromosome_startswith, output_dir, output_prefix + "_pl")
    mn, mc = _get_coverage_bw(bw_mn, chromosome_startswith, output_dir, output_prefix + "_mn")

    pl_chrs = set(pl.keys())
    mn_chrs = set(mn.keys())

    if pl_chrs != mn_chrs:
        logger.warning("BW files do not have the same chromosomes. "
                       "Only chromosomes appear in both files will be analysed.")
        allowed_bws = pl_chrs.intersection(mn_chrs)
        logger.info("Consistent chromosomes: %s" % sorted(allowed_bws))

        def _clean(in_dict, whitelist):
            adjust_counts = 0  # need to adjust the total read counts
            static_keys = list(in_dict.keys())
            for c in static_keys:
                if c not in whitelist:
                    adjust_counts += np.abs(np.nansum(np.load(in_dict[c])))
                    os.remove(in_dict[c])
                    del in_dict[c]
            return adjust_counts

        c0 = _clean(pl, allowed_bws)
        pc -= c0
        c1 = _clean(mn, allowed_bws)
        mc -= c1
    return pl, mn, pc + mc


class BamParsers:
    def __init__(self, bam_obj, pl_cov, mn_cov, mapq, **kwargs):
        c = list(pl_cov.keys())
        c.extend(list(mn_cov.keys()))
        self.total_reads = 0
        self.chroms = set()
        if len(c) > 0:
            self.chroms = set(c)
        self.bam_obj = bam_obj
        self.pl_cov = pl_cov
        self.mn_cov = mn_cov
        self.mapq = mapq

    def se_5p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def se_5p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.reference_name not in self.chroms:
                continue
            if read.mapq < self.mapq:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1
            read_strand = "-" if read_strand == "+" else "+"

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def se_3p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1

    def se_3p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start
            read_strand = "-" if read_strand == "+" else "+"

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1

    def pe_r1_5p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read2 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def pe_r1_5p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read2 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1
            read_strand = "-" if read_strand == "+" else "+"

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def pe_r2_5p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read1 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def pe_r2_5p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read1 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            read_strand = "-" if read.is_reverse else "+"
            pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1
            read_strand = "-" if read_strand == "+" else "+"

            self.total_reads += 1
            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_5prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_5prime] += 1

    def pe_r1_3p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read2 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start

            self.total_reads += 1
            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1

    def pe_r1_3p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read2 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue
            self.total_reads += 1

            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start
            read_strand = "-" if read_strand == "+" else "+"

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1

    def pe_r2_3p_ss(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read1 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1

    def pe_r2_3p_rc(self):
        self.total_reads = 0
        for read in self.bam_obj:
            if read.is_read1 or read.mapq < self.mapq:
                continue
            if read.reference_name not in self.chroms:
                continue

            self.total_reads += 1
            read_strand = "-" if read.is_reverse else "+"
            pos_3prime = read.reference_end - 1 if read_strand == "+" else read.reference_start
            read_strand = "-" if read_strand == "+" else "+"

            if read_strand == "+":
                self.pl_cov[read.reference_name][pos_3prime] += 1
            elif read_strand == "-":
                self.mn_cov[read.reference_name][pos_3prime] += 1


def get_read_signal(input_bam, loc_prime, chromosome_startswith, output_dir, output_prefix, filters=(),
                    reverse_complement=False, **kwargs):
    """

    Parameters
    ----------
    input_bam : input_bam : str
        Full path to input bam file
    loc_prime : str, format: Read_End

    chromosome_startswith : str or None
        Filter out reads whose reference don't start with chromosome_startswith
    output_dir : str
        Path to write outputs
    output_prefix : str
        Prefix of outputs
    filters : list or tuple
        List of keywords to filter chromosomes
    reverse_complement :
    **kwargs :
        Keyword arguments, mapq_threshold (int, default 30) and output_chrom_size (bool, default True) is effective
    Returns
    -------
    chromosome_coverage_pl : dict
        Dictionary of per base coverage per chromosome (positive strand)
    chromosome_coverage_mn : dict
        Dictionary of per base coverage per chromosome (negative strand)
    read_counts : int
        Total read counts passing filters
    """
    logger = logging.getLogger(__logger_name__)
    supported_protocols = {
        "PROcap": "R_5_f",
        "GROcap": "R_5_f",
        "PROseq": "R_5_r",
        "GROseq": "R_5_f",
        "CoPRO": "R2_5_f",
        "CAGE": "R_5_f",
        "NETCAGE": "R_5_f",
        "RAMPAGE": "R1_5_f",
        "STRIPEseq": "R1_5_f",
        "csRNAseq": "R_5_f",
        "mNETseq": "R2_5_r"
    }
    if loc_prime in supported_protocols:
        read_num, read_end, fw_rv = supported_protocols[loc_prime].split("_")
        loc_prime = "%s_%s" % (read_num, read_end)
        reverse_complement = bool(fw_rv != "f")
    log_assert(loc_prime in ("R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3"),
               "library_type must be R1_5, R1_3, R2_5 or R2_3. Current value: {0}".format(loc_prime), logger)
    library_layout, interested_end = loc_prime.split("_")
    mapq_threshold = kwargs["mapq_threshold"] if "mapq_threshold" in kwargs.keys() else 30
    chromosome_coverage_pl = {}
    chromosome_coverage_mn = {}

    result_pl = {}
    result_mn = {}

    try:
        bam = pysam.AlignmentFile(input_bam, "rb")
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

    for chromosome in bam.header["SQ"]:
        if "SN" in chromosome.keys() and "LN" in chromosome.keys() and \
                chromosome["SN"].startswith(chromosome_startswith):
            ignore_flag = 0
            for keyword in filters:
                if chromosome["SN"].find(keyword) != -1:
                    ignore_flag = 1
            if not ignore_flag:
                if chromosome["LN"] > 0:
                    chromosome_coverage_pl[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
                    chromosome_coverage_mn[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
                else:
                    logger.warning(
                        "The sequence length of chromosome {} is 0."
                        "Ignoring this chromosome.".format(chromosome["SN"]))

    def load_signal_from_cache(pre_chromosome_coverage, results, direction="pl"):
        read_counts = 0
        flag = 1
        for chrom in pre_chromosome_coverage:
            fn = os.path.join(output_dir, output_prefix + "_%s_%s" % (direction, chrom)) + ".npy"
            if not os.path.exists(fn):
                flag = 0
                break
            results[chrom] = fn
            tmp = np.load(fn)
            read_counts += np.sum(tmp)
            del tmp
        return flag, read_counts

    lcf, rc = load_signal_from_cache(chromosome_coverage_pl, result_pl, direction="pl")
    lcf1, rc1 = load_signal_from_cache(chromosome_coverage_mn, result_mn, direction="mn")
    load_cache_flag = 1 if lcf and lcf1 else 0
    read_counts = rc + rc1

    fast = None
    if not load_cache_flag and kwargs.get("fast_bam", True):
        fast = parse_bam_by_contig(input_bam, [(c, chromosome_coverage_pl[c].shape[0]) for c in chromosome_coverage_pl],
                                   mapq=mapq_threshold, layout=library_layout, end=interested_end,
                                   rc=bool(reverse_complement), output_dir=output_dir, output_prefix=output_prefix,
                                   threads=kwargs.get("thread_n", 1))

    if load_cache_flag:
        logger.info("Loading cache from previous result, if you want to re-parse the bam file, "
                    "please delete all files end with npy first.")
    elif fast is not None:
        result_pl, result_mn, read_counts = fast
    else:
        bam_parser = None
        parser_obj = BamParsers(bam_obj=bam, pl_cov=chromosome_coverage_pl,
                                mn_cov=chromosome_coverage_mn, mapq=mapq_threshold)
        if library_layout == "R":
            if interested_end == "5":
                if reverse_complement:
                    bam_parser = parser_obj.se_5p_rc
                else:
                    bam_parser = parser_obj.se_5p_ss
            else:
                if reverse_complement:
                    bam_parser = parser_obj.se_3p_rc
                else:
                    bam_parser = parser_obj.se_3p_ss
        elif library_layout == "R1":
            if interested_end == "5":
                if reverse_complement:
                    bam_parser = parser_obj.pe_r1_5p_rc
                else:
                    bam_parser = parser_obj.pe_r1_5p_ss
            else:
                if reverse_complement:
                    bam_parser = parser_obj.pe_r1_3p_rc
                else:
                    bam_parser = parser_obj.pe_r1_3p_ss
        elif library_layout == "R2":
            if interested_end == "5":
                if reverse_complement:
                    bam_parser = parser_obj.pe_r2_5p_rc
                else:
                    bam_parser = parser_obj.pe_r2_5p_ss
            else:
                if reverse_complement:
                    bam_parser = parser_obj.pe_r2_3p_rc
                else:
                    bam_parser = parser_obj.pe_r2_3p_ss

        log_assert(bam_parser is not None, "Cannot initiate a parser for this experiment", logger)

        bam_parser()
        read_counts = parser_obj.total_reads
        # convert data type if necessary
        for chrom in chromosome_coverage_pl.keys():
            pl_chrom_max = np.max(chromosome_coverage_pl[chrom])
            mn_chrom_max = np.max(chromosome_coverage_mn[chrom])
            for k, v in enumerate(NP_DT_RANGE):
                if v > pl_chrom_max:
                    chromosome_coverage_pl[chrom] = chromosome_coverage_pl[chrom].astype(NP_DT_NAME[k], copy=False)
                    fn = os.path.join(output_dir, output_prefix + "_pl_%s" % chrom)
                    np.save(fn, chromosome_coverage_pl[chrom])
                    result_pl[chrom] = fn + ".npy"
                    break
            for k, v in enumerate(NP_DT_RANGE):
                if v > mn_chrom_max:
                    chromosome_coverage_mn[chrom] = chromosome_coverage_mn[chrom].astype(NP_DT_NAME[k], copy=False)
                    fn = os.path.join(output_dir, output_prefix + "_mn_%s" % chrom)
                    np.save(fn, chromosome_coverage_mn[chrom])
                    result_mn[chrom] = fn + ".npy"
                    break
    return result_pl, result_mn, read_counts


def normalize_using_input(assay_pl, assay_mn, input_pl, input_mn, scale_factor, output_dir, output_prefix, logger):
    """
    Normalize signal using input/control

    Parameters
    ----------
    assay_pl : dict
        signal (npys) from treatment group for plus strand
    assay_mn : dict
        signal (npys) from treatment group for minus strand
    input_pl : dict
        signal (npys) from input/control group for plus strand
    input_mn : dict
        signal (npys) from input/control group for minus strand
    scale_factor : float
        scale factor to normalize sequencing depth
    output_dir : str
        where to put the outputs
    output_prefix : str
        prefix for all outputs
    logger : Logger object

    Returns
    -------
    coverage_pl : dict
        Normalized signal (npys) from treatment group for plus strand
    coverage_mn : dict
        Normalized signal (npys) from treatment group for minus strand

    """

    def strand_atom(assay_coverage, input_coverage, sf=1., direction="pl"):
        results = {}
        for chrom_name, npy_dump in assay_coverage.items():
            if chrom_name in input_coverage and is_zero_track(npy_dump) and np.isfinite(sf) and sf >= 0:
                # exp == 0 everywhere, so max(exp - input * sf, 0) == 0 everywhere -> int8 zeros
                length = np.load(npy_dump, mmap_mode="r").shape[0]
                fn = os.path.join(output_dir, output_prefix + "_%s_%s" % (direction, chrom_name)) + ".npy"
                results[chrom_name] = save_zero_track(fn, length, NP_DT_NAME[0])
            elif chrom_name in input_coverage:
                matched_input = np.load(input_coverage[chrom_name]) * sf
                exp = np.load(npy_dump)
                adjusted_signal = exp - matched_input
                adjusted_signal[adjusted_signal < 0] = 0
                chrom_max = np.max(adjusted_signal)
                for k, v in enumerate(NP_DT_RANGE):
                    if v > chrom_max:
                        adjusted_signal = adjusted_signal.astype(NP_DT_NAME[k], copy=False)
                        fn = os.path.join(output_dir, output_prefix + "_%s_%s" % (direction, chrom_name)) + ".npy"
                        np.save(fn, adjusted_signal)
                        results[chrom_name] = fn
                        break
            else:
                logger.warning("%s doesn't have paired input/control" % chrom_name)
        return results

    coverage_pl = strand_atom(assay_pl, input_pl, sf=scale_factor, direction="pl")
    coverage_mn = strand_atom(assay_mn, input_mn, sf=scale_factor, direction="mn")
    return coverage_pl, coverage_mn


def get_coverage(input_bam, library_type, chromosome_startswith, output_dir, output_prefix, **kwargs):
    """
    Get 5' coverage

    Parameters
    ----------
    input_bam : str
        Full path to input bam file
    library_type : str
        Library type, available options: se, pe_fr, pe_rf
    chromosome_startswith : str
        Filter out reads whose reference don't start with chromosome_startswith
    output_dir : str
        Path to write outputs
    output_prefix : str
        Prefix of outputs
    **kwargs
        Keyword arguments, mapq_threshold (int, default 30) and output_chrom_size (bool, default True) is effective

    Returns
    -------
    chromosome_coverage_pl : dict
        Dictionary of per base coverage per chromosome (positive strand)
    chromosome_coverage_mn : dict
        Dictionary of per base coverage per chromosome (negative strand)
    """
    logger = logging.getLogger(__logger_name__)
    log_assert(library_type in ("se", "pe_fr", "pe_rf"), "library_type must be se, pe_fr or pe_rf", logger)
    mapq_threshold = kwargs["mapq_threshold"] if "mapq_threshold" in kwargs.keys() else 30
    chromosome_coverage_pl = {}
    chromosome_coverage_mn = {}
    chromosome_pre_accessible_dict = {}
    chromosome_reads_dict = {}
    bam = pysam.AlignmentFile(input_bam, "rb")

    chrom_size_list = []

    for chromosome in bam.header["SQ"]:
        if "SN" in chromosome.keys() and "LN" in chromosome.keys() and \
                chromosome["SN"].startswith(chromosome_startswith):
            chromosome_coverage_pl[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
            chromosome_coverage_mn[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
            chrom_size_list.append((chromosome["SN"], chromosome["LN"]))
            chromosome_pre_accessible_dict[chromosome["SN"]] = []
            chromosome_reads_dict[chromosome["SN"]] = 0

    if "output_chrom_size" in kwargs.keys() and kwargs["output_chrom_size"]:
        csize_fn = os.path.join(output_dir, output_prefix + ".csize")
        with open(csize_fn, "w") as f:
            pd.DataFrame(chrom_size_list, columns=["Chromosome", "Size"]).to_csv(f,
                                                                                 sep="\t",
                                                                                 index=False,
                                                                                 header=False)

    for read in bam:
        if read.mapq < mapq_threshold or \
                (library_type == "pe_fr" and read.is_reverse) or \
                (library_type == "pe_rf" and not read.is_reverse):
            continue
        if not read.reference_name.startswith(chromosome_startswith):
            continue

        read_strand = "-" if read.is_reverse else "+"
        pos_5prime = read.reference_start if read_strand == "+" else read.reference_end - 1
        chromosome_pre_accessible_dict[read.reference_name].append((read.reference_start, read.reference_end))
        chromosome_reads_dict[read.reference_name] += 1
        if read_strand == "+" and read.reference_name.startswith(chromosome_startswith):
            chromosome_coverage_pl[read.reference_name][pos_5prime] += 1
        elif read_strand == "-" and read.reference_name.startswith(chromosome_startswith):
            chromosome_coverage_mn[read.reference_name][pos_5prime] += 1

    # convert data type if necessary
    result_pl = {}
    result_mn = {}
    for chrom in chromosome_coverage_pl.keys():
        pl_chrom_max = np.max(chromosome_coverage_pl[chrom])
        mn_chrom_max = np.max(chromosome_coverage_mn[chrom])
        for k, v in enumerate(NP_DT_RANGE):
            if v > pl_chrom_max:
                chromosome_coverage_pl[chrom] = chromosome_coverage_pl[chrom].astype(NP_DT_NAME[k], copy=False)
                fn = os.path.join(output_dir, output_prefix + "_pl_%s" % chrom)
                np.save(fn, chromosome_coverage_pl[chrom])
                result_pl[chrom] = fn + ".npy"
                break
        for k, v in enumerate(NP_DT_RANGE):
            if v > mn_chrom_max:
                chromosome_coverage_mn[chrom] = chromosome_coverage_mn[chrom].astype(NP_DT_NAME[k], copy=False)
                fn = os.path.join(output_dir, output_prefix + "_mn_%s" % chrom)
                np.save(fn, chromosome_coverage_mn[chrom])
                result_mn[chrom] = fn + ".npy"
                break
    return result_pl, result_mn


def index_bed_file(file_path, logger=None):
    """
    Compress and index a bed file

    Parameters
    ----------
    file_path : str
        Path to the plain bed file to be compressed and indexed
    logger : None or a logger
        Logger to write error info
    Returns
    -------

    """
    from pybedtools import BedTool
    if logger is None:
        assert os.path.exists(file_path), "Bed file {} cannot be located".format(file_path)
    else:
        log_assert(os.path.exists(file_path), "Bed file {} cannot be located".format(file_path), logger)
    # compress and index
    fn = BedTool(file_path).bgzip(in_place=True, force=True)
    pysam.tabix_index(fn, preset="bed" if fn.find("bed") != -1 else "gff", force=True, csi=True)
    # remove original file
    os.remove(file_path)


def parse_gtf(filename, cache=1, cache_suffix="_pd_cache.csv"):
    """
    Parse GTF file

    Parameters
    ----------
    filename : str
        path to plain GTF or gzipped GTF
    cache : int or bool
        1/True for generating cache
    cache_suffix : str
        suffix for cache file
    Returns
    -------
    df: pd.DataFrame
        Parsed file in pd.DataFrame
    Author
    ------
    Kamil Slowikowski (https://gist.github.com/slowkow/8101481)
    """
    assert os.path.exists(filename), "Cannot access the file you provided."

    cache_file = filename + cache_suffix
    if cache and os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(filename):
        return pd.read_csv(cache_file, index_col=None)

    result = defaultdict(list)

    for i, line in enumerate(read_lines(filename)):
        for key in line.keys():
            if key not in result:
                result[key] = [None] * i

        for key in result.keys():
            result[key].append(line.get(key, None))

    df = pd.DataFrame(result)
    if cache:
        try:
            df.to_csv(cache_file, index=False)
        except OSError:  # e.g. the annotation sits in a read-only folder; the cache is only a speed-up
            pass
    if not pd.api.types.is_numeric_dtype(df["start"]):
        df["start"] = df["start"].astype(int)
    if not pd.api.types.is_numeric_dtype(df["end"]):
        df["end"] = df["end"].astype(int)
    return df


def read_lines(filename):
    """
    Open an GTF file and generate a dict for each line.

    Parameters
    ----------
    filename : str
        path to the gtf file

    Returns
    -------
    yield

    Author
    ------
    Kamil Slowikowski (https://gist.github.com/slowkow/8101481)
    """
    fn_open = gzip.open if filename.endswith('.gz') else open
    mode = "rt" if filename.endswith('.gz') else "r"

    with fn_open(filename, mode) as fh:
        for line in fh:
            if not line.startswith('#'):
                yield parse(line)


def parse(line):
    """Parse a single line from a GTF file and return a dict

    Parameters
    ----------
    line : str

    Returns
    -------
    line_elements : dict
        elements in this line

    Author
    ------
    Kamil Slowikowski (https://gist.github.com/slowkow/8101481)
    """
    result = {}

    fields = line.rstrip().split('\t')

    for i, col in enumerate(GTF_HEADER):
        result[col] = _get_value(fields[i])

    # INFO field consists of "key1=value;key2=value;...".
    # infos = [x for x in re.split(R_SEMICOLON, fields[8]) if x.strip()]
    infos = [x for x in fields[8].split(";") if x.strip()]

    for i, info in enumerate(infos, 1):
        # It should be key="value".
        try:
            key, value = info.split()
        # But sometimes it is just "value".
        except ValueError:
            key = 'INFO{}'.format(i)
            value = info
        # Ignore the field if there is no value.
        if value:
            result[key] = _get_value(value)

    return result


def _get_value(value):
    """Get value of the key

    Parameters
    ----------
    value : str
        value

    Returns
    -------
    value : str or list
        Handled value

    Author
    ------
    Kamil Slowikowski (https://gist.github.com/slowkow/8101481)
    """
    if not value:
        return None

    # Strip double and single quotes.
    value = value.strip('"\'')

    # Return a list if the value has a comma.
    if ',' in value:
        value = value.split(",")
    # These values are equivalent to None.
    elif value in ['', '.', 'NA']:
        return None

    return value


def peak_bed_to_gtf(pl_df, mn_df, save_to, version=""):
    """
    Transform intermediate peak files to gtf format

    Parameters
    ----------
    pl_df : pd.DataFrame
        Information about peaks on positive strand
    mn_df : pd.DataFrame
        Information about peaks on negative strand
    save_to : str
        Full path for the output
    version : str
        Version info

    Returns
    -------

    """
    result = pd.concat([pl_df, mn_df])
    result["attribute"] = "peak_name " + result["name"] + "; reads " + result["reads"].map(str) + "; pval " + result[
        "pval"].map(str) + "; qval " + result["padj"].map(str) + "; mu_bg " + result["mu_0"].map(str) + "; pi_bg " + \
                          result["pi_0"].map(str) + "; mu_peak " + result["mu_1"].map(str) + "; pi_peak " + result[
                              "pi_1"].map(str) + "; ler1 " + result["ler_1"].map(str) + "; ler2 " + result["ler_2"].map(
        str) + "; ler3 " + result["ler_3"].map(str) + "; summit " + result["summit"].map(str)

    result["source"] = "PINTS_ver{ver}".format(ver=version)
    result["feature"] = "peak"
    result["frame"] = 0
    result["start"] += 1
    result = result.sort_values(by=["chromosome", "start"])
    result.loc[:, ("chromosome", "source", "feature",
                   "start", "end", "padj", "strand",
                   "frame", "attribute")].to_csv(save_to,
                                                 sep="\t",
                                                 index=False,
                                                 header=False)
    index_bed_file(save_to)


def merge_replicates_bw(per_sample_coverage_pls, per_sample_coverage_mns, output_dir, output_prefix):
    """
    Merge replicate files for 'plus' and 'minus' strand coverage into a consolidated format and save the
    merged data to disk. This function processes chromosome-specific coverage files for each replicate,
    merging them into a single file per chromosome. It returns the paths of the merged files and the
    combined read counts for both 'plus' and 'minus' strands.

    Parameters
    ----------
    per_sample_coverage_pls : list of dict
        List of dictionaries representing the file paths for 'plus' strand coverage per sample. Each
        dictionary contains chromosome-specific file paths for a given replicate.
    per_sample_coverage_mns : list of dict
        List of dictionaries representing the file paths for 'minus' strand coverage per sample. Each
        dictionary contains chromosome-specific file paths for a given replicate.
    output_dir : str
        Directory where the merged chromosome files will be saved.
    output_prefix : str
        Prefix to append to the filenames of the merged chromosome-specific files.

    Returns
    -------
    tuple
        A tuple containing the following:
        - merged_pl : dict
            Dictionary where keys are chromosome names and values are paths to the merged 'plus' strand
            coverage files.
        - merged_mn : dict
            Dictionary where keys are chromosome names and values are paths to the merged 'minus' strand
            coverage files.
        - pl_rc : int
            Total read count for the 'plus' strand after merging all replicates.
        - mn_rc : int
            Total read count for the 'minus' strand after merging all replicates.

    Raises
    ------
    ValueError
        If the number of replicates for 'plus' and 'minus' strands are not equal.
        If the input lists of replicates are empty (no data to merge).

    Notes
    -----
    1. Ensure that all input dictionaries in `per_sample_coverage_pls` and `per_sample_coverage_mns`
       have the same chromosomes.
    2. The merged output files are stored in the specified `output_dir` using NumPy arrays with
       appropriate data types determined from the maximum values in the merged data.
    """
    merged_pl = {}
    merged_mn = {}
    pl_rc = 0
    mn_rc = 0

    shared_chromosomes = reduce(set.intersection, (set(d.keys()) for d in per_sample_coverage_pls+per_sample_coverage_mns))

    if len(per_sample_coverage_pls) != len(per_sample_coverage_mns):
        raise ValueError("Number of replicates for pl and mn must be the same")

    def merge_chromosome_data(sample_data_list, chromosome):
        merged = np.load(sample_data_list[0][chromosome]).astype(np.int32)
        for replicate in sample_data_list[1:]:
            merged += np.load(replicate[chromosome])
        return merged

    # no need to merge if there is only one replicate
    if len(per_sample_coverage_pls) == 1:
        merged_pl = per_sample_coverage_pls[0]
        merged_mn = per_sample_coverage_mns[0]
    elif len(per_sample_coverage_pls) == 0:
        raise ValueError("Number of replicates for pl and mn must be greater than 1.")
    else:
        for chrom in shared_chromosomes:
            pl_fn = os.path.join(output_dir, "%s_pl_%s.npy" % (output_prefix, chrom))
            mn_fn = os.path.join(output_dir, "%s_mn_%s.npy" % (output_prefix, chrom))
            if all(is_zero_track(d[chrom]) for d in per_sample_coverage_pls + per_sample_coverage_mns):
                length = np.load(per_sample_coverage_pls[0][chrom], mmap_mode="r").shape[0]
                merged_pl[chrom] = save_zero_track(pl_fn, length, determine_data_type(0))
                merged_mn[chrom] = save_zero_track(mn_fn, length, determine_data_type(0))
                continue
            # merge data
            merged_pl_data = merge_chromosome_data(per_sample_coverage_pls, chrom)
            pl_rc += np.sum(merged_pl_data)
            merged_mn_data = merge_chromosome_data(per_sample_coverage_mns, chrom)
            mn_rc += np.sum(merged_mn_data)

            # merged file paths
            pl_fn = os.path.join(output_dir, "%s_pl_%s.npy" % (output_prefix, chrom))
            mn_fn = os.path.join(output_dir, "%s_mn_%s.npy" % (output_prefix, chrom))

            merged_pl[chrom] = pl_fn
            merged_mn[chrom] = mn_fn

            # dump data with appropriate data types
            data_type = determine_data_type(np.max(merged_pl_data))
            np.save(pl_fn, merged_pl_data.astype(data_type, copy=False))

            data_type = determine_data_type(np.max(merged_mn_data))
            np.save(mn_fn, merged_mn_data.astype(data_type, copy=False))
    return merged_pl, merged_mn, pl_rc, mn_rc
