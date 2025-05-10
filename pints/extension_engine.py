#!/usr/bin/env python
# coding=utf-8

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

import os
import sys
import logging

try:
    import numpy as np
    import pandas as pd
    from pybedtools import BedTool
    from pints.io_engine import get_read_signal, get_coverage_bw, log_assert
    from collections import namedtuple
except ImportError as e:
    missing_package = str(e).replace("No module named '", "").replace("'", "")
    sys.exit("Please install %s first!" % missing_package)

logger = logging.getLogger("PINTS - BoundaryExtender")
__EXTENDED_FILE_TPL__ = "_element_{de_tag}bp.bed"
__EXTENDED_DISTAL_FILE_TPL__ = "_element_{de_tag}bp_e.bed"


def main(input_file, layout, div_file, bid_file, single_file, divergent_extension=(60, 60),
         unidirectional_extension=(60, 60), promoter_bed=None):
    """
    Extend boundaries

    Parameters
    ----------
    input_file : str
        Path to a bam file
    layout : str
        Layout out the bam file
    div_file : str
        Path to divergent peaks PINTS called
    bid_file : str
        Path to bidirectional peaks PINTS called
    single_file : str
        Path to unidirectional peaks PINTS called
    divergent_extension : tuple
        BPs to be extended for both divergent and bidirectional peaks
    unidirectional_extension : tuple
        BPs to be extended for unidirectional peaks
    promoter_bed : str or None
        Path to a bed file which defines promoter regions

    Returns
    -------
    None
    """

    de_tag = "_".join(set(map(str, divergent_extension)))
    parent_path = os.path.dirname(div_file)
    if isinstance(input_file, str):
        log_assert(layout is not None,
                   "Please specify which type of experiment this data was generated from with --exp-type", logger)
        pl, mn, _ = get_read_signal(input_file, loc_prime=layout, chromosome_startswith="chr",
                                    output_dir=parent_path, output_prefix=str(os.getpid()))
    else:
        log_assert(len(input_file[0]) == len(input_file[1]),
                   "Must provide the same amount of bigwig files for both strands", logger)

        pl, mn, _ = get_coverage_bw(bw_pl=input_file[0], bw_mn=input_file[1],
                                    chromosome_startswith="chr",
                                    output_dir=parent_path,
                                    output_prefix=str(os.getpid()))

    div = pd.read_csv(div_file, sep="\t", header=None)
    div = div.loc[np.logical_or(div[0].isin(pl), div[0].isin(mn)), :]
    bid = pd.read_csv(bid_file, sep="\t", header=None)
    bid = bid.loc[np.logical_or(bid[0].isin(pl), bid[0].isin(mn)), :]
    single = pd.read_csv(single_file, sep="\t", header=None)
    single = single.loc[np.logical_or(single[0].isin(pl), single[0].isin(mn)), :]
    div["pl_summit"] = "0"
    div["mn_summit"] = "0"
    div["element_start"] = 0
    div["element_end"] = 0
    bid["pl_summit"] = "0"
    bid["mn_summit"] = "0"
    single["summit"] = 0
    for chromosome in pl:
        pl_cov = np.load(pl[chromosome], allow_pickle=True)
        mn_cov = np.load(mn[chromosome], allow_pickle=True)
        div_sub = div.loc[div[0] == chromosome, :]
        bid_sub = bid.loc[bid[0] == chromosome, :]
        for sub_df, all_df in zip((div_sub, bid_sub), (div, bid)):
            for nr, row in sub_df.iterrows():
                pcov = pl_cov[row[1]:row[2]]
                mcov = mn_cov[row[1]:row[2]]
                cpls = np.where(pcov == pcov.max())[0] + row[1]
                cmns = np.where(mcov == mcov.max())[0] + row[1]
                all_df.loc[nr, "pl_summit"] = ",".join([str(x) for x in cpls])
                all_df.loc[nr, "mn_summit"] = ",".join([str(x) for x in cmns])

                # extend boundaries with the following conditions:
                # Find the prominent peaks at basepair resolution (any peaks with ⅓ of the highest peak and >5 reads)
                # and extend x (60, 200, or others) bps beyond the furthest prominent peak
                plb = np.nanmax(pcov)
                mlb = np.nanmax(mcov)
                pl_threshold = min(plb, max(plb * 0.3, 5))
                mn_threshold = min(mlb, max(mlb * 0.3, 5))

                pl_probe = np.where(pcov > pl_threshold)[0]
                if pl_probe.shape[0] > 1:
                    cpl = min(pl_probe[-1] + row[1], row[2])
                else:
                    cpl = cpls[-1]
                mn_probe = np.where(mcov > mn_threshold)[0]
                if mn_probe.shape[0] > 1:
                    cmn = max(mn_probe[0] + row[1], row[1])
                else:
                    cmn = cmns[0]

                f = min(cpl, cmn) - divergent_extension[0]
                r = max(cpl, cmn) + divergent_extension[1]
                # only update the boundaries if the new ones are larger than the old ones
                all_df.loc[nr, "element_start"] = f if f < row[1] else row[1]
                all_df.loc[nr, "element_end"] = r if r > row[2] else row[2]

        # unidirectional elements are defined as:
        # peak region boundaries defined by PINTS
        # go upstream 300bp (we assume the opposite peak should be within 300 bp),
        # then further +60 or +200bp to define the whole element
        single_sub = single.loc[single[0] == chromosome, :]
        for nr, row in single_sub.iterrows():
            if row[5] == "+":
                f = row[1] - unidirectional_extension[0] - 300
                r = row[2] + unidirectional_extension[1]
            else:
                f = row[1] - unidirectional_extension[0]
                r = row[2] + unidirectional_extension[1] + 300

            single.loc[nr, "element_start"] = f
            single.loc[nr, "element_end"] = r
    div = div.loc[:, (0, "element_start", "element_end", 3, 4, 5)]
    div.element_start = div.element_start.astype(int)
    div.element_end = div.element_end.astype(int)
    div.loc[div.element_start < 0, "element_start"] = 0
    div["ID"] = ["Divergent" + str(i) for i in list(div.index)]
    div["strand"] = "."
    div = div[[0, "element_start", "element_end", "ID", 3, "strand", 5, 4]]

    bid = bid.loc[:, (0, "element_start", "element_end", 3, 4, 5)]
    bid.element_start = bid.element_start.astype(int)
    bid.element_end = bid.element_end.astype(int)
    bid.loc[bid.element_start < 0, "element_start"] = 0
    bid["ID"] = ["Bidirectional" + str(i) for i in list(bid.index)]
    bid["strand"] = "."
    bid = bid[[0, "element_start", "element_end", "ID", 3, "strand", 5, 4]]

    single = single.loc[:, (0, "element_start", "element_end", 3, 4, 5, 7)]
    single.element_start = single.element_start.astype(int)
    single.element_end = single.element_end.astype(int)
    single.loc[single.element_start < 0, "element_start"] = 0
    single["ID"] = ["Unidirectional" + str(i) for i in list(single.index)]
    div.to_csv(div_file.replace(".bed", __EXTENDED_FILE_TPL__.format(de_tag=de_tag)), sep="\t", index=False,
               header=False)
    bid.to_csv(bid_file.replace(".bed", __EXTENDED_FILE_TPL__.format(de_tag=de_tag)), sep="\t", index=False,
               header=False)
    single_obj = BedTool(single.to_csv(sep="\t", index=False, header=False), from_string=True)

    if promoter_bed is not None:
        promoter_bed_obj = BedTool(promoter_bed)
        BedTool.from_dataframe(div).intersect(promoter_bed_obj, v=True).saveas(
            div_file.replace(".bed", __EXTENDED_DISTAL_FILE_TPL__.format(de_tag=de_tag)))
        BedTool.from_dataframe(bid).intersect(promoter_bed_obj, v=True).saveas(
            bid_file.replace(".bed", __EXTENDED_DISTAL_FILE_TPL__.format(de_tag=de_tag)))
        single_obj.intersect(promoter_bed_obj, v=True).saveas(
            single_file.replace(".bed", __EXTENDED_DISTAL_FILE_TPL__.format(de_tag=de_tag)))

    single_obj.moveto(single_file.replace(".bed", __EXTENDED_FILE_TPL__.format(de_tag=de_tag)))
    housekeeping_files = []
    housekeeping_files.extend(pl.values())
    housekeeping_files.extend(mn.values())
    for hf in housekeeping_files:
        if os.path.exists(hf):
            try:
                os.remove(hf)
            except IOError:
                pass


def extend(args):
    if sum((args.bw_pl is None, args.bw_mn is None)) == 1:
        raise ValueError("both of the two arguments --bw-pl --bw-mn are required")

    if args.bam_files is not None and not len(args.bam_files) == len(args.divergent_files) == len(
            args.bidirectional_files) == len(args.unidirectional_files):
        raise ValueError("Number of peak calls from different categories should match")

    if args.bw_pl is not None and not len(args.bw_pl) == len(args.bw_mn) == len(args.divergent_files) == len(
            args.bidirectional_files) == len(args.unidirectional_files):
        raise ValueError("Number of peak calls from different categories should match")

    assert len(args.div_ext_left) == len(args.div_ext_right)
    assert len(args.unidirectional_ext_left) == len(args.unidirectional_ext_right)

    for i in range(1, (len(args.bam_files) if args.bam_files is not None else len(args.bw_pl)) + 1):
        groups = {
            "divergent_calls": None,
            "bidirectional_calls": None,
            "unidirectional_calls": None
        }

        element_types = ("divergent_peaks", "bidirectional_peaks", "unidirectional_peaks")

        for et in element_types:
            k = "_{index}_{et}.bed".format(index=i, et=et)
            for df in args.divergent_files:
                if df.find(k) != -1:
                    groups["divergent_calls"] = df

            for bf in args.bidirectional_files:
                if bf.find(k) != -1:
                    groups["bidirectional_calls"] = bf

            for sf in args.unidirectional_files:
                if sf.find(k) != -1:
                    groups["unidirectional_calls"] = sf
        for j in range(len(args.div_ext_left)):
            if args.bam_files is not None:
                main(args.bam_files[i - 1], args.bam_parser[i - 1] if len(args.bam_parser) > 1 else args.bam_parser[0],
                     groups["divergent_calls"], groups["bidirectional_calls"], groups["unidirectional_calls"],
                     divergent_extension=(args.div_ext_left[j], args.div_ext_right[j]),
                     unidirectional_extension=(args.unidirectional_ext_left[j], args.unidirectional_ext_right[j]),
                     promoter_bed=args.promoter_bed)
            else:
                main((args.bw_pl[i - 1], args.bw_mn[i - 1]), None,
                     groups["divergent_calls"], groups["bidirectional_calls"], groups["unidirectional_calls"],
                     divergent_extension=(args.div_ext_left[j], args.div_ext_right[j]),
                     unidirectional_extension=(args.unidirectional_ext_left[j], args.unidirectional_ext_right[j]),
                     promoter_bed=args.promoter_bed)
