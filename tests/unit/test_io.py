"""Characterisation tests: BAM parsing matrix, bigWig loading, input normalisation, replicate merging."""
import itertools
import logging
import os

import numpy as np
import pytest

from pints import io_engine as io
from helpers import make_bam, expected_positions, make_bigwig

CONTIGS = [("chr1", 5000), ("chr2", 3000), ("scaffold_7", 800), ("chrM", 300)]


def _reads():
    r = []
    # single-end style reads on both strands, soft clips, deletions, MAPQ boundaries
    for i, (pos, cig, rev, mq) in enumerate([
        (100, "50M", False, 60), (100, "50M", True, 60), (200, "5S45M", False, 60), (200, "45M5S", True, 60),
        (300, "20M10D20M", True, 60), (400, "30M", False, 29), (410, "30M", False, 30), (420, "30M", True, 31),
        (0, "10M", False, 60), (4990, "10M", True, 60), (2500, "30M2I20M", False, 40),
    ]):
        r.append(dict(contig="chr1", pos=pos, cigar=cig, reverse=rev, mapq=mq))
    # paired-end reads (R1/R2), incl. secondary + supplementary records (not filtered by PINTS)
    for i in range(12):
        rev = bool(i % 2)
        r.append(dict(contig="chr2", pos=100 + 37 * i, cigar="40M", reverse=rev, mapq=60, paired=True,
                      read1=(i % 3 != 0), read2=(i % 3 == 0), name="pp%d" % (i // 2)))
    r.append(dict(contig="chr2", pos=900, cigar="40M", reverse=False, mapq=60, secondary=True, paired=True,
                  read1=True))
    r.append(dict(contig="chr2", pos=950, cigar="40M", reverse=True, mapq=60, supplementary=True, paired=True,
                  read2=True))
    r.append(dict(contig="scaffold_7", pos=10, cigar="30M", reverse=True, mapq=60))
    r.append(dict(contig="chrM", pos=10, cigar="30M", reverse=False, mapq=60))
    r.append(dict(contig=None, pos=0, cigar="30M", mapq=0, unmapped=True))   # unplaced unmapped read
    return r


PARSERS = list(itertools.product(("R", "R1", "R2"), ("5", "3"), (False, True)))


@pytest.fixture(scope="module")
def bam(tmp_path_factory):
    d = tmp_path_factory.mktemp("bam")
    return make_bam(d / "t.bam", CONTIGS, _reads())


@pytest.mark.parametrize("layout,end,rc", PARSERS)
def test_bam_parser_matrix(bam, tmp_path, layout, end, rc):
    pl, mn, total = io.get_read_signal(bam, "%s_%s" % (layout, end), "", str(tmp_path), "x",
                                       reverse_complement=rc, mapq_threshold=30)
    exp, exp_total = expected_positions(_reads(), end, rc, None if layout == "R" else layout)
    assert total == exp_total
    for contig, length in CONTIGS:
        for strand, d in (("+", pl), ("-", mn)):
            arr = np.load(d[contig])
            want = np.bincount(exp.get((contig, strand), []), minlength=length)[:length]
            assert np.array_equal(arr.astype(np.int64), want), (contig, strand)
            assert arr.dtype == np.int8   # downcast to smallest signed dtype


@pytest.mark.parametrize("preset,code", [("PROcap", "R_5_f"), ("GROcap", "R_5_f"), ("PROseq", "R_5_r"),
                                         ("GROseq", "R_5_f"), ("CoPRO", "R2_5_f"), ("CAGE", "R_5_f"),
                                         ("NETCAGE", "R_5_f"), ("RAMPAGE", "R1_5_f"), ("STRIPEseq", "R1_5_f"),
                                         ("csRNAseq", "R_5_f"), ("mNETseq", "R2_5_r")])
def test_exp_type_presets(bam, tmp_path, preset, code):
    layout, end, f = code.split("_")
    os.makedirs(tmp_path / "a")
    os.makedirs(tmp_path / "b")
    pl, mn, total = io.get_read_signal(bam, preset, "", str(tmp_path / "a"), "x")
    pl2, mn2, total2 = io.get_read_signal(bam, "%s_%s" % (layout, end), "", str(tmp_path / "b"), "x",
                                          reverse_complement=(f == "r"))
    assert total == total2
    for c in pl:
        assert np.array_equal(np.load(pl[c]), np.load(pl2[c])) and np.array_equal(np.load(mn[c]), np.load(mn2[c]))


def test_bam_chromosome_prefix_and_filters(bam, tmp_path):
    pl, mn, total = io.get_read_signal(bam, "R_5", "chr", str(tmp_path), "x", filters=("M",))
    assert set(pl) == {"chr1", "chr2"}


def test_bam_cache_reuse(bam, tmp_path):
    pl, mn, total = io.get_read_signal(bam, "R_5", "", str(tmp_path), "x")
    pl2, mn2, total2 = io.get_read_signal(bam, "R_5", "", str(tmp_path), "x")
    assert pl == pl2 and total == total2


def test_bam_dtype_upcast(tmp_path):
    reads = [dict(contig="chr1", pos=10, cigar="10M", reverse=False, mapq=60) for _ in range(200)]
    b = make_bam(tmp_path / "deep.bam", [("chr1", 100)], reads)
    pl, mn, total = io.get_read_signal(b, "R_5", "", str(tmp_path), "d")
    arr = np.load(pl["chr1"])
    assert arr.dtype == np.int16 and arr[10] == 200 and total == 200


# ------------------------------------------------------------ bigWig
def test_bigwig_loader(tmp_path):
    chroms = [("chr1", 1000), ("chrX", 500)]
    pv = {"chr1": np.zeros(1000), "chrX": np.zeros(500)}
    mv = {"chr1": np.zeros(1000), "chrX": np.zeros(500)}
    pv["chr1"][[5, 6, 700]] = [3, 1, 120]
    mv["chr1"][[10, 999]] = [-4, -2]     # minus strand stored as negative values
    pv["chrX"][[0]] = [2]
    pl_bw = make_bigwig(tmp_path / "p.bw", chroms, pv)
    mn_bw = make_bigwig(tmp_path / "m.bw", chroms, mv)
    pl, mn, rc = io.get_coverage_bw(pl_bw, mn_bw, "", str(tmp_path), "bw")
    assert rc == 3 + 1 + 120 + 4 + 2 + 2
    a = np.load(pl["chr1"])
    assert a.dtype == np.int8 and a[700] == 120 and a[5] == 3 and a.shape == (1000,)
    assert np.load(mn["chr1"])[10] == 4


def test_bigwig_mismatched_chromosomes(tmp_path):
    pl_bw = make_bigwig(tmp_path / "p.bw", [("chr1", 100), ("chr2", 100)],
                        {"chr1": np.ones(100), "chr2": np.ones(100)})
    mn_bw = make_bigwig(tmp_path / "m.bw", [("chr1", 100)], {"chr1": -np.ones(100)})
    pl, mn, rc = io.get_coverage_bw(pl_bw, mn_bw, "", str(tmp_path), "bw")
    assert set(pl) == set(mn) == {"chr1"} and rc == 200


# ------------------------------------------------------------ normalisation / merging
def test_normalize_using_input(tmp_path):
    a = str(tmp_path / "a.npy"); i = str(tmp_path / "i.npy")
    np.save(a, np.array([5, 3, 0, 10], dtype=np.int8))
    np.save(i, np.array([1, 4, 2, 3], dtype=np.int8))
    pl, mn = io.normalize_using_input({"c": a}, {"c": a}, {"c": i}, {}, 1.5, str(tmp_path), "n",
                                      logging.getLogger("t"))
    got = np.load(pl["c"])
    assert np.array_equal(got, np.array([3, 0, 0, 5], dtype=np.int8))    # trunc(5-1.5)=3, trunc(10-4.5)=5
    assert mn == {}


def test_merge_replicates(tmp_path):
    f = {}
    for k, v in {"p1": [1, 2, 0], "p2": [0, 120, 1], "m1": [0, 0, 1], "m2": [3, 0, 0]}.items():
        f[k] = str(tmp_path / (k + ".npy"))
        np.save(f[k], np.array(v, dtype=np.int8))
    pl, mn, prc, mrc = io.merge_replicates_bw([{"c": f["p1"]}, {"c": f["p2"]}], [{"c": f["m1"]}, {"c": f["m2"]}],
                                              str(tmp_path), "mg")
    assert np.array_equal(np.load(pl["c"]), [1, 122, 1]) and np.load(pl["c"]).dtype == np.int8
    assert prc == 124 and mrc == 4


def test_merge_single_replicate_passthrough(tmp_path):
    pl, mn, prc, mrc = io.merge_replicates_bw([{"c": "x"}], [{"c": "y"}], str(tmp_path), "mg")
    assert pl == {"c": "x"} and (prc, mrc) == (0, 0)


@pytest.mark.parametrize("v,dt", [(0, np.int8), (126, np.int8), (127, np.int16), (40000, np.int32),
                                  (2 ** 31, np.int64)])
def test_determine_data_type(v, dt):
    assert io.determine_data_type(v) == dt
