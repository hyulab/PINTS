#  PINTS: Peak Identifier for Nascent Transcripts Starts
#  Copyright (c) 2019-2023 Yu Lab.
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
from argparse import Namespace
import unittest
import os
import pandas as pd
import pybedtools
from pandas.api.types import is_object_dtype, is_integer_dtype, is_numeric_dtype
from pints.calling_engine import peak_calling
from pints.extension_engine import extend


__CHR_TYPE_ERR__ = "dtype for the Chromosome column should be object"
__START_TYPE_ERR__ = "dtype for the Start column should be integers"
__END_TYPE_ERR__ = "dtype for the End column should be integers"
__TSS_RE__ = "^[0-9,]*$"
__STRAND_RE__ = "^[+-]*$"


class FunctionalTestCase(unittest.TestCase):
    def setUp(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        pl_bw_file = os.path.join(
            self.current_dir,
            "datasets/pl.chr22.bw")
        mn_bw_file = os.path.join(
            self.current_dir,
            "datasets/mn.chr22.bw"
        )
        self.fdr_target = 0.1
        self.relaxed_fdr_target = 0.2

        expected_epi_annotation = os.path.join(self.current_dir, "ft_ENCFF948HPU.bigBed")
        if not os.path.exists(expected_epi_annotation):
            os.symlink(os.path.join(self.current_dir, "datasets/ENCFF948HPU.bigBed"),
                       expected_epi_annotation)
        # peak calling with bw as input
        peak_calling(input_bam=None, bw_pl=(pl_bw_file,), bw_mn=(mn_bw_file,),
                     output_dir=self.current_dir, output_prefix="ft", fdr_target=self.fdr_target,
                     donor_tolerance=0.3, output_diagnostics=True,
                     annotation_gtf=os.path.join(self.current_dir, "datasets/annotations.chr22.gtf.gz"),
                     highlight_chromosome="chr22",
                     epig_annotation="k562", relaxed_fdr_target=self.relaxed_fdr_target,
                     )

        expected_epi_annotation = os.path.join(self.current_dir, "bft_ENCFF948HPU.bigBed")
        if not os.path.exists(expected_epi_annotation):
            os.symlink(os.path.join(self.current_dir, "datasets/ENCFF948HPU.bigBed"),
                       expected_epi_annotation)
        # peak calling with bam as input
        peak_calling(input_bam=(os.path.join(self.current_dir, "datasets/chr22.bam"),),
                     ct_bam=None,
                     output_dir=self.current_dir, output_prefix="bft", fdr_target=self.fdr_target,
                     bam_parser="GROcap", chromosome_startswith="chr", bw_pl=None, bw_mn=None,
                     donor_tolerance=0.3, output_diagnostics=True,
                     annotation_gtf=os.path.join(self.current_dir, "datasets/annotations.chr22.gtf.gz"),
                     highlight_chromosome="chr22",
                     epig_annotation="k562", relaxed_fdr_target=self.relaxed_fdr_target,)

        # # peak calling with control bam
        # peak_calling(input_bam=(os.path.join(self.current_dir, "datasets/chr22.bam"),),
        #              ct_bam=(os.path.join(self.current_dir, "datasets/input.chr22.bam"),),
        #              output_dir=self.current_dir, output_prefix="cft", fdr_target=self.fdr_target,
        #              bam_parser="GROcap", chromosome_startswith="chr", bw_pl=None, bw_mn=None)

        self.div_peaks = (
            os.path.join(self.current_dir, "ft_1_divergent_peaks.bed"),
            os.path.join(self.current_dir, "bft_1_divergent_peaks.bed"),
        )
        self.bid_peaks = (
            os.path.join(self.current_dir, "ft_1_bidirectional_peaks.bed"),
            os.path.join(self.current_dir, "bft_1_bidirectional_peaks.bed"),
        )
        self.uni_peaks = (
            os.path.join(self.current_dir, "ft_1_unidirectional_peaks.bed"),
            os.path.join(self.current_dir, "bft_1_unidirectional_peaks.bed"),
        )

        ext_args = {
            "bam_files": None,
            "bw_pl": (pl_bw_file, ),
            "bw_mn": (mn_bw_file,),
            "divergent_files": (self.div_peaks[0], ),
            "bidirectional_files": (self.bid_peaks[0], ),
            "unidirectional_files": (self.uni_peaks[0], ),
            "save_to": self.current_dir,
            "div_ext_left": (60, ),
            "div_ext_right": (60, ),
            "unidirectional_ext_left": (60, ),
            "unidirectional_ext_right": (60, ),
            "promoter_bed": None,
        }
        pybedtools.set_tempdir(self.current_dir)
        extend(Namespace(**ext_args))

        self.div_eles = os.path.join(self.current_dir, "ft_1_divergent_peaks_element_60bp.bed")
        self.bid_eles = os.path.join(self.current_dir, "ft_1_bidirectional_peaks_element_60bp.bed")
        self.uni_eles = os.path.join(self.current_dir, "ft_1_unidirectional_peaks_element_60bp.bed")

        expected_regions = (
            # The following regions are promoters for selected housekeeping genes
            # defined in E. Eisenberg and E.Y. Levanon, Trends in Genetics, 29 (2013)
            ("chr22", 24951475, 24952474, "p1"),  # SNRPD3
            ("chr22", 29999045, 30000044, "p2"),  # NF2
            ("chr22", 29168162, 29169161, "p3"),  # CCDC117
            ("chr22", 50781252, 50782251, "p4"),  # PPP6R2
            ("chr22", 21212601, 21213600, "p5"),  # PI4KA
            ("chr22", 32145618, 32146617, "p6"),  # PRR14L
            # The following regions are previously validated enhancer regions
            ("chr22", 27028364, 27029827, "e1"),
            ("chr22", 27530320, 27530813, "e2"),
            ("chr22", 29591389, 29592436, "e3"),
            ("chr22", 30612540, 30613150, "e4"),
            ("chr22", 37378085, 37378721, "e5"),
            ("chr22", 38003001, 38003426, "e6"),
        )
        pybedtools.set_tempdir(self.current_dir)
        self.ref_regions = pybedtools.BedTool(expected_regions)

    def check_bid_div_peak_df(self, df):
        self.assertTrue(df.shape[1] == 6 or df.shape[1] == 7, "Expecting 6 or 7 columns in the output file")
        self.assertTrue(is_object_dtype(df[0]), __CHR_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[1]), __START_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[2]), __END_TYPE_ERR__)
        self.assertTrue(sum(df[3].str.contains("Stringent|Relaxed|Marginal")) == df.shape[0],
                        "Confidence level must be the combinations of Stringent, Relaxed, and Marginal")
        self.assertTrue(is_object_dtype(df[3]), "dtype for the Confidence column should be object")
        self.assertTrue(sum(df[4].map(str).str.match(__TSS_RE__)),
                        "Major TSSs on the forward strand must be integers or integers separated by comma")
        self.assertTrue(df.apply(
            lambda x: all([x[1] <= int(tss) <= x[2] for tss in str(x[4]).split(",")]), axis=1
        ).sum() == df.shape[0])
        self.assertTrue(sum(df[5].map(str).str.match(__TSS_RE__)),
                        "Major TSSs on the reverse strand must be integers or integers separated by comma")
        self.assertTrue(df.apply(
            lambda x: all([x[1] <= int(tss) <= x[2] for tss in str(x[5]).split(",")]), axis=1
        ).sum() == df.shape[0])

    def check_unid_peak_df(self, df):
        self.assertTrue(df.shape[1] == 9 or df.shape[1] == 10, "Expecting 9 or 10 columns in the output file")
        self.assertTrue(is_object_dtype(df[0]), __CHR_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[1]), __START_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[2]), __END_TYPE_ERR__)
        self.assertTrue(is_object_dtype(df[3]), "dtype for the Peak ID column should be object")
        self.assertTrue(is_numeric_dtype(df[4]), "dtype for the Q-value column should be numeric")
        self.assertTrue(((0 <= df[4]) & (df[4] <= self.relaxed_fdr_target)).sum() == df.shape[0],
                        "Q-values should be in the range of [0, fdr_target]")
        self.assertTrue(is_object_dtype(df[5]), "dtype for the Strand column should be object")
        self.assertTrue(sum(df[5].map(str).str.match(__STRAND_RE__)), "Strand must either be + or -")
        self.assertTrue(is_integer_dtype(df[6]), "dtype for the Read counts column should be integers")
        self.assertTrue(sum(df[7].map(str).str.match(__TSS_RE__)),
                        "Position of the summit TSS must be integers or integers separated by comma")
        self.assertTrue(sum(df[8].map(str).str.match(__TSS_RE__)),
                        "Height of the summit must be integers or integers separated by comma")

    def test_peak_calling(self):
        peak_groups = (self.div_peaks, self.bid_peaks, self.uni_peaks)
        # expected outputs
        for peak_group in peak_groups:
            for peak_file in peak_group:
                self.assertTrue(os.path.exists(peak_file))

        # check shape and format
        for shape_checker, peak_group in zip(
                (self.check_bid_div_peak_df, self.check_bid_div_peak_df, self.check_unid_peak_df),
                peak_groups):
            for peak_file in peak_group:
                shape_checker(pd.read_csv(peak_file, sep="\t", header=None))

        # check whether statistical engines are working properly
        for i in range(len(self.div_peaks)):
            merged_peaks = pybedtools.BedTool.cat(*[pybedtools.BedTool(self.bid_peaks[i]),
                                                    pybedtools.BedTool(self.uni_peaks[i])])
            identified_regions = self.ref_regions.intersect(merged_peaks, u=True).count()
            total_regions = self.ref_regions.count()
            self.assertEqual(identified_regions, total_regions,
                             "Not all ref regions were identified ({0}, {1})".format(identified_regions, total_regions))

    def check_bid_div_ele_df(self, df):
        self.assertTrue(df.shape[1] == 8 or df.shape[1] == 9, "Expecting 8 or 9 columns in the output file")
        self.assertTrue(is_object_dtype(df[0]), __CHR_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[1]), __START_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[2]), __END_TYPE_ERR__)
        self.assertTrue(sum(df[4].str.contains("Stringent|Relaxed|Marginal")) == df.shape[0],
                        "Confidence level must be the combinations of Stringent, Relaxed, and Marginal")
        self.assertTrue(is_object_dtype(df[4]), "dtype for the Confidence column should be object")
        self.assertTrue(sum(df[7].map(str).str.match(__TSS_RE__)),
                        "Major TSSs on the forward strand must be integers or integers separated by comma")
        self.assertTrue(sum(df[6].map(str).str.match(__TSS_RE__)),
                        "Major TSSs on the reverse strand must be integers or integers separated by comma")
        self.assertTrue(is_object_dtype(df[3]), "dtype for the ID column should be object")
        self.assertTrue(sum(df[3].str.contains("Bidirectional|Divergent")) == df.shape[0],
                        "Element ID must start with Bidirectional or Divergent")
        # check boundary values
        rts_probe = df.apply(lambda x: all(lambda t: x[1] < int(t) < x[2] for t in str(x[6]).split(",")), axis=1)
        fts_probe = df.apply(lambda x: all(lambda t: x[1] < int(t) < x[2] for t in str(x[7]).split(",")), axis=1)
        self.assertEqual(sum(rts_probe), df.shape[0])
        self.assertEqual(sum(fts_probe), df.shape[0])

    def check_unid_ele_df(self, df):
        self.assertTrue(df.shape[1] == 8 or df.shape[1] == 9, "Expecting 8 or 9 columns in the output file")
        self.assertTrue(is_object_dtype(df[0]), __CHR_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[1]), __START_TYPE_ERR__)
        self.assertTrue(is_integer_dtype(df[2]), __END_TYPE_ERR__)
        self.assertTrue(is_object_dtype(df[3]), "dtype for the Peak ID column should be object")
        self.assertTrue(is_numeric_dtype(df[4]), "dtype for the Q-value column should be numeric")
        self.assertTrue(((0 <= df[4]) & (df[4] <= self.relaxed_fdr_target)).sum() == df.shape[0],
                        "Q-values should be in the range of [0, fdr_target]")
        self.assertTrue(is_object_dtype(df[5]), "dtype for the Strand column should be object")
        self.assertTrue(sum(df[5].map(str).str.match(__STRAND_RE__)), "Strand must either be + or -")
        self.assertTrue(sum(df[6].map(str).str.match(__TSS_RE__)),
                        "Position of the summit TSS must be integers or integers separated by comma")
        if df[6].shape[0] > 1:
            self.assertTrue(df[6].nunique() > 1,
                            "Positions of the summit TSS shouldn't be identical")

    def test_bw_bam_consistency(self):
        # peaks identified from bw files should be very similar to that from bam files
        self.assertAlmostEqual(
            pybedtools.BedTool(self.bid_peaks[0]).jaccard(
                pybedtools.BedTool(self.bid_peaks[1]))["jaccard"],
            1, delta=0.05)
        self.assertAlmostEqual(
            pybedtools.BedTool(self.div_peaks[0]).jaccard(
                pybedtools.BedTool(self.div_peaks[1]))["jaccard"],
            1, delta=0.05)
        self.assertAlmostEqual(
            pybedtools.BedTool(self.uni_peaks[0]).jaccard(
                pybedtools.BedTool(self.uni_peaks[1]))["jaccard"],
            1, delta=0.05)

    def test_boundary_extension(self):
        # expected outputs
        self.assertTrue(os.path.exists(self.div_eles))
        self.assertTrue(os.path.exists(self.bid_eles))
        self.assertTrue(os.path.exists(self.uni_eles))

        # check shape and format
        self.check_bid_div_ele_df(pd.read_csv(self.div_eles, sep="\t", header=None))
        self.check_bid_div_ele_df(pd.read_csv(self.bid_eles, sep="\t", header=None))
        self.check_unid_ele_df(pd.read_csv(self.uni_eles, sep="\t", header=None))

        # check whether statistical engines are working properly
        merged_peaks = pybedtools.BedTool.cat(*[pybedtools.BedTool(self.bid_eles), pybedtools.BedTool(self.uni_eles)])
        identified_regions = self.ref_regions.intersect(merged_peaks, u=True).count()
        total_regions = self.ref_regions.count()
        self.assertEqual(identified_regions, total_regions,
                         "Not all ref regions were identified ({0}, {1})".format(identified_regions, total_regions))

        # check whether elements got extended
        for peak_file, element_file in zip((self.bid_peaks[0], self.div_peaks[0], self.uni_peaks[0]),
                                           (self.bid_eles, self.div_eles, self.uni_eles)):
            ele_df = pd.read_csv(element_file, sep="\t", header=None)
            peak_df = pd.read_csv(peak_file, sep="\t", header=None)
            ele_lengths = ele_df[2] - ele_df[1]
            peak_lengths = peak_df[2] - peak_df[1]
            self.assertGreater(ele_lengths.mean(), peak_lengths.mean(),
                               "The average length of elements should be longer than that of peaks'")


if __name__ == '__main__':
    unittest.main()
