#!/usr/bin/env python
# coding=utf-8
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
import pysam
import sys
import os
import logging
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict

NP_DT_RANGE = (127, 32767, 2147483647, 9223372036854775807)
NP_DT_NAME = (np.int8, np.int16, np.int32, np.int64)
GTF_HEADER = ["seqname", "source", "feature", "start", "end", "score",
              "strand", "frame"]


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
    logger = logging.getLogger("PINTS - IO engine")
    try:
        import pyBigWig
    except ImportError as e:
        missing_package = str(e).replace("No module named '", "").replace("'", "")
        logger.error("Please install %s first!" % missing_package)
        sys.exit(-1)
    bw = pyBigWig.open(bw_file)
    log_assert(bw.isBigWig(), "BigWig file %s is not valid." % bw_file, logger)
    chromosome_coverage = dict()
    chromosome_pre_accessible_dict = dict()
    chromosome_reads_dict = dict()

    data_type = np.int32
    bw_max = max(abs(bw.header()["maxVal"]), abs(bw.header()["minVal"]))
    for k, v in enumerate(NP_DT_RANGE):
        if v > bw_max:
            data_type = NP_DT_NAME[k]
            break

    chrom_size_list = []
    # max_reference_length = 0
    genome_size = 0
    read_counts = 0

    # result = {}
    for chromosome, csize in bw.chroms().items():
        if chromosome.startswith(chromosome_startswith):
            fn = os.path.join(output_dir, "%s_%s" % (output_prefix, chromosome))
            chromosome_coverage[chromosome] = fn + ".npy"
            if not os.path.exists(chromosome_coverage[chromosome]):
                _chromosome_cov = np.nan_to_num(bw.values(chromosome, 0, csize))
                _chromosome_cov[_chromosome_cov < 0] *= -1
                _chromosome_cov = _chromosome_cov.astype(data_type, copy=False)
                np.save(fn, _chromosome_cov)
            else:
                _chromosome_cov = np.load(chromosome_coverage[chromosome])
            read_counts += np.sum(_chromosome_cov)
            chrom_size_list.append((chromosome, csize))
            chromosome_pre_accessible_dict[chromosome] = []
            chromosome_reads_dict[chromosome] = 0
            genome_size += csize
    """
    if "output_chrom_size" in kwargs.keys() and kwargs["output_chrom_size"]:
        csize_fn = os.path.join(output_dir, output_prefix + ".csize")
        with open(csize_fn, "w") as f:
            pd.DataFrame(chrom_size_list, columns=["Chromosome", "Size"]).to_csv(f,
                                                                                 sep="\t",
                                                                                 index=False,
                                                                                 header=False)
    """
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
    """
    logger = logging.getLogger("IO engine")
    pl, pc = _get_coverage_bw(bw_pl, chromosome_startswith, output_dir, output_prefix + "_pl")
    mn, mc = _get_coverage_bw(bw_mn, chromosome_startswith, output_dir, output_prefix + "_mn")
    log_assert(pl.keys() == mn.keys(), "bw_pl and bw_mn should have the same chromosomes", logger)
    return pl, mn, pc + mc


class BamParsers(object):
    def __init__(self, bam_obj, pl_cov, mn_cov, mapq, **kwargs):
        c = list(pl_cov.keys())
        c.extend(list(mn_cov.keys()))
        self.total_reads = 0
        self.chroms = set()
        if c is not None:
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


def get_read_signal(input_bam, loc_prime, chromosome_startswith, output_dir, output_prefix, filters=[],
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
    logger = logging.getLogger("IO engine")
    supported_protocols = {
        "PROcap": "R_5_f",
        "GROcap": "R_5_f",
        "PROseq": "R_5_r",
        "GROseq": "R_5_f",
        "CoPRO": "R2_5_f",
        "CAGE": "R_5_f",
        "NETCAGE": "R_5_f",
        "csRNAseq": "R_5_f",
        "mNETseq": "R2_5_r"
    }
    if loc_prime in supported_protocols:
        read_num, read_end, fw_rv = supported_protocols[loc_prime].split("_")
        loc_prime = "%s_%s" % (read_num, read_end)
        reverse_complement = False if fw_rv == "f" else True
    log_assert(loc_prime in ("R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3"),
               "library_type must be R1_5, R1_3, R2_5 or R2_3", logger)
    library_layout, interested_end = loc_prime.split("_")
    mapq_threshold = kwargs["mapq_threshold"] if "mapq_threshold" in kwargs.keys() else 30
    chromosome_coverage_pl = dict()
    chromosome_coverage_mn = dict()

    result_pl = {}
    result_mn = {}

    try:
        bam = pysam.AlignmentFile(input_bam, "rb")
    except Exception as e:
        logger.error(e)
        sys.exit(-1)
    genome_size = 0

    for chromosome in bam.header["SQ"]:
        if "SN" in chromosome.keys() and "LN" in chromosome.keys() and \
                chromosome["SN"].startswith(chromosome_startswith):
            ignore_flag = 0
            for keyword in filters:
                if chromosome["SN"].find(keyword) != -1:
                    ignore_flag = 1
            if not ignore_flag:
                chromosome_coverage_pl[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
                chromosome_coverage_mn[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
                genome_size += chromosome["LN"]

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

    if load_cache_flag:
        logger.info("Loading cache from previous result, if you want to re-parse the bam file, "
                    "please delete all files end with npy first.")
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
        results = dict()
        for chrom_name, npy_dump in assay_coverage.items():
            if chrom_name in input_coverage:
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
    logger = logging.getLogger("IO engine")
    log_assert(library_type in ("se", "pe_fr", "pe_rf"), "library_type must be se, pe_fr or pe_rf", logger)
    mapq_threshold = kwargs["mapq_threshold"] if "mapq_threshold" in kwargs.keys() else 30
    chromosome_coverage_pl = dict()
    chromosome_coverage_mn = dict()
    chromosome_pre_accessible_dict = dict()
    chromosome_reads_dict = dict()
    # chromosome_lambda_dict = dict()
    bam = pysam.AlignmentFile(input_bam, "rb")

    chrom_size_list = []
    # max_reference_length = 0
    genome_size = 0

    for chromosome in bam.header["SQ"]:
        if "SN" in chromosome.keys() and "LN" in chromosome.keys() and \
                chromosome["SN"].startswith(chromosome_startswith):
            chromosome_coverage_pl[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
            chromosome_coverage_mn[chromosome["SN"]] = np.zeros(chromosome["LN"], dtype=np.uint32)
            chrom_size_list.append((chromosome["SN"], chromosome["LN"]))
            chromosome_pre_accessible_dict[chromosome["SN"]] = []
            chromosome_reads_dict[chromosome["SN"]] = 0
            # chromosome_lambda_dict[chromosome["SN"]] = 0.0
            genome_size += chromosome["LN"]

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
    BedTool(file_path).tabix(force=True)
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

    if cache and os.path.exists(filename + cache_suffix):
        return pd.read_csv(filename + cache_suffix, index_col=None)

    result = defaultdict(list)

    for i, line in enumerate(read_lines(filename)):
        for key in line.keys():
            if key not in result:
                result[key] = [None] * i

        for key in result.keys():
            result[key].append(line.get(key, None))

    df = pd.DataFrame(result)
    if cache:
        df.to_csv(filename + cache_suffix, index=False)
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
            if line.startswith('#'):
                continue
            else:
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
            # key, _, value = re.split(R_KEYVALUE, info, 1)
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
    result.sort_values(by=["chromosome", "start"], inplace=True)
    result.loc[:, ("chromosome", "source", "feature",
                   "start", "end", "padj", "strand",
                   "frame", "attribute")].to_csv(save_to,
                                                 sep="\t",
                                                 index=False,
                                                 header=False)
