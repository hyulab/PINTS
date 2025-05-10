# coding=utf-8
import os.path

#  PINTS: Peak Identifier for Nascent Transcript Starts
#  Copyright (c) 2019-2025 Yu Lab.
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
import pyBigWig
from pints.io_engine import parse_gtf


def _row_atom(X, pl_handler, mn_handler):
    """

    Parameters
    ----------
    X
    pl_handler
    mn_handler

    Returns
    -------

    """
    if X.strand == "+":
        tss_start = X.start - 500
        tss_end = X.start + 500
        gbody_start = tss_end + 1
        gbody_end = X.end - 500
        handler = pl_handler
    else:
        tss_start = X.end - 500
        tss_end = X.end + 500
        gbody_start = X.start + 500
        gbody_end = tss_start - 1
        handler = mn_handler

    try:
        tss_counts = handler.stats(X.seqname, tss_start, tss_end, "sum", exact=True)[0]
    except:
        tss_counts = 0
    tss_counts = tss_counts if tss_counts is not None else 0
    tss_counts = tss_counts if tss_counts >= 0 else -1*tss_counts
    try:
        gbody_counts = handler.stats(X.seqname, gbody_start, gbody_end, "sum", exact=True)[0]
    except:
        gbody_counts = 0
    gbody_counts = gbody_counts if gbody_counts is not None else 0
    gbody_counts = gbody_counts if gbody_counts >= 0 else -1*gbody_counts
    return X.gene_name, X.transcript_id, tss_counts, gbody_counts


def calculate_gbody_tss_ratio(pl_bw_file, mn_bw_file, reference_gtf):
    """
    Calculate read count ratio of gene body to tss regions

    Parameters
    ----------
    pl_bw_file : str
        Path to the pl bw file
    mn_bw_file : str
        Path to the mn bw file
    reference_gtf : str
        Path to the gene annotation gtf file

    Returns
    -------
    gb_tss_ratio : float

    """
    if not all([os.path.exists(x) for x in (pl_bw_file, mn_bw_file, reference_gtf)]):
        raise IOError("Please make sure pl_bw_file, mn_bw_file and reference_gtf are accessible!")

    ref = parse_gtf(reference_gtf)
    expected_cols = {"feature", "transcript_type", "start", "end", "seqname"}
    if not all([x in ref.columns for x in expected_cols]):
        raise ValueError("The gtf file doesn't contain all required columns.")
    ref = ref.loc[(ref.feature == "transcript") & (ref.transcript_type == "protein_coding"), :]
    ref = ref.loc[ref.end-ref.start > 2000, :]

    with pyBigWig.open(pl_bw_file) as pl_bw, pyBigWig.open(mn_bw_file) as mn_bw:
        results = ref.apply(_row_atom, axis=1, args=(pl_bw, mn_bw), result_type="expand")
    results = results.sort_values(by=[0, 2], ascending=False).drop_duplicates(subset=0, keep="first")

    total_counts = (results[2]+results[3])
    gb_tss_ratio = (results[3][
        results[2] > results[2].quantile(0.9)
        ]).sum()/total_counts[
            results[2] > results[2].quantile(0.9)
            ].sum()

    return gb_tss_ratio

