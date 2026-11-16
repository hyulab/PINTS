"""bed_ops against bedtools 2.31 and pybedtools: sort (ties included), merge with collapse/distinct, cat."""
import os
import random
import shutil
import subprocess

import pytest

from pints import bed_ops

BEDTOOLS = shutil.which("bedtools")
CHROMS = ["chr1", "chr10", "chr2", "chrX", "chrUn_KI270742v1", "1", "MT", "chr1_KI270706v1_random", "Chr3"]
needs_bedtools = pytest.mark.skipif(BEDTOOLS is None, reason="bedtools is not installed")


def _records(rng, n, span, chroms, pattern="random", n_extra=3):
    """PINTS-shaped BED records (name, score, strand, then extra columns), many with tied starts."""
    starts = [rng.randrange(span) for _ in range(n)]
    if pattern == "sorted":
        starts.sort()
    elif pattern == "reversed":
        starts.sort(reverse=True)
    elif pattern == "organ":
        starts.sort()
        starts = starts[:n // 2] + starts[n // 2:][::-1]
    lines = []
    for i, s in enumerate(starts):
        e = s + (0 if rng.random() < 0.05 else rng.randrange(1, 400))
        extra = [rng.choice(["a", "b", "B", "10", "2", "9", ".", "1.50"]) for _ in range(n_extra)]
        lines.append("\t".join([rng.choice(chroms), str(s), str(e), "id%d" % i, str(rng.randrange(9)),
                                rng.choice("+-")] + extra) + "\n")
    return lines


def _bedtools(args, text):
    return subprocess.run([BEDTOOLS] + args, input=text, capture_output=True, text=True, check=True).stdout


@needs_bedtools
@pytest.mark.parametrize("seed", range(60))
def test_sort_and_merge_match_bedtools(seed):
    rng = random.Random(seed)
    n = rng.choice([0, 1, 2, 3, 15, 16, 17, 18, 33, 100, 1000, 5000])
    lines = _records(rng, n, rng.choice([1, 2, 5, 50, 1000, 10 ** 6]), rng.sample(CHROMS, rng.choice([1, 2, 5])),
                     rng.choice(["random", "sorted", "reversed", "organ"]))
    expected = _bedtools(["sort", "-i", "stdin"], "".join(lines))
    got = bed_ops.sort_lines(lines)
    assert "".join(got) == expected
    if lines:
        cols, ops = (7, 8, 9, 4, 5), ("collapse", "distinct", "collapse", "distinct", "collapse")
        expected = _bedtools(["merge", "-i", "stdin", "-c", ",".join(map(str, cols)), "-o", ",".join(ops)],
                             expected)
        assert "".join(bed_ops.merge_lines(got, cols, ops)) == expected
        d = rng.choice([1, 10, 500])
        assert "".join(bed_ops.merge_lines(got, distance=d)) == _bedtools(["merge", "-i", "stdin", "-d", str(d)],
                                                                           "".join(got))


def test_std_sort_heapsort_fallback_and_thresholds():
    """Depth-limit and insertion-sort boundaries give a correctly sorted permutation of the input."""
    for n in (0, 1, 16, 17, 64, 1000):
        for key in ([0] * n, list(range(n)), list(range(n))[::-1], [i % 3 for i in range(n)]):
            idx = bed_ops._std_sort(list(range(n)), key)
            assert sorted(idx) == list(range(n))
            assert [key[i] for i in idx] == sorted(key)


@needs_bedtools
@pytest.mark.parametrize("seed", range(20))
def test_cat_matches_pybedtools(seed, tmp_path):
    pybedtools = pytest.importorskip("pybedtools")
    pybedtools.set_tempdir(str(tmp_path))
    rng = random.Random(seed)
    paths = []
    for k, n_extra in enumerate((10, rng.choice([10, 10, 8]))):
        p = str(tmp_path / ("%d.bed" % k))
        bed_ops.write_lines(_records(rng, rng.choice([0, 1, 17, 500]), rng.choice([20, 1000, 10 ** 6]),
                                     ["chr1", "chr2", "chr10"], n_extra=n_extra), p)
        paths.append(p)
    if all(bed_ops.is_empty(p) for p in paths):
        return
    bts = [pybedtools.BedTool(p) for p in paths]
    expected = open(pybedtools.BedTool.cat(*bts, postmerge=False).sort().fn).read()
    assert "".join(bed_ops.sort_lines(bed_ops.cat_lines(paths))) == expected
    if not any(bed_ops.is_empty(p) for p in paths) and len({len(open(p).readline().split("\t")) for p in paths}) == 1:
        cols, ops = (12, 13, 14, 15, 16), ("collapse",) * 4 + ("distinct",)
        expected = open(pybedtools.BedTool.cat(*bts, c=cols, o=ops).fn).read()
        assert "".join(bed_ops.merge_lines(bed_ops.sort_lines(bed_ops.cat_lines(paths)), cols, ops)) == expected


def test_is_empty(tmp_path):
    p = str(tmp_path / "x.bed")
    for text, empty in (("", True), ("\n\n", True), ("# c\ntrack x\n", True), ("chr1\t1\t2\n", False)):
        with open(p, "w") as fh:
            fh.write(text)
        assert bed_ops.is_empty(p) is empty


def test_zero_length_merge_rules():
    lines = ["chr1\t5\t5\ta\n", "chr1\t20\t20\tb\n", "chr1\t21\t30\tc\n"]
    assert bed_ops.merge_lines(lines, (4,), ("collapse",)) == ["chr1\t5\t5\ta\n", "chr1\t19\t30\tb,c\n"]


@needs_bedtools
def test_gtf_is_sorted_on_its_start_column(tmp_path):
    """The format is told from the records, as bedtools does, not from the file name."""
    rng = random.Random(3)
    lines = ["%s\tPINTS_ver1\tpeak\t%d\t%d\t0.01\t%s\t0\tgene_id \"p%d\";\n"
             % (rng.choice(["chr1", "chr2"]), s, s + 50, rng.choice("+-"), i)
             for i, s in enumerate(rng.randrange(1, 300) for _ in range(200))]
    p = str(tmp_path / "bed_named.gtf")
    bed_ops.write_lines(lines, p)
    assert "".join(bed_ops.sort_lines(bed_ops.read_lines(p))) == _bedtools(["sort", "-i", p], "")


@needs_bedtools
@pytest.mark.parametrize("seed", range(60))
def test_strand_merge_and_intersect_match_bedtools(seed, tmp_path):
    rng = random.Random(1000 + seed)
    span = rng.choice([1, 5, 50, 1000, 10 ** 5])
    chroms = rng.sample(CHROMS, rng.choice([1, 2, 4]))
    a = _bedtools(["sort", "-i", "stdin"], "".join(_records(rng, rng.choice([1, 2, 17, 300, 3000]), span, chroms)))
    b = _records(rng, rng.choice([0, 1, 5, 200, 3000]), span, chroms)
    for k, ln in enumerate(b):  # some strands bedtools -s ignores
        if rng.random() < 0.1:
            f = ln.split("\t")
            f[5] = "."
            b[k] = "\t".join(f)
    a_lines = a.splitlines(keepends=True)
    if rng.random() < 0.5:
        rng.shuffle(a_lines)  # intersect keeps the order of -a, sorted or not
    pa, pb = tmp_path / "a.bed", tmp_path / "b.bed"
    pa.write_text("".join(a_lines))
    pb.write_text("".join(b))
    # strand-aware merge: the same merged records as bedtools (whose order differs), zero-length ones refused
    sorted_b = bed_ops.sort_lines([ln for ln in b if ln.split("\t")[1] != ln.split("\t")[2]])
    expected = _bedtools(["merge", "-s", "-i", "stdin", "-c", "4,5,6", "-o", "distinct,distinct,distinct"],
                         "".join(sorted_b)) if sorted_b else ""
    got = bed_ops.merge_lines(sorted_b, (4, 5, 6), ("distinct",) * 3, strand=True)
    assert sorted(got) == sorted(expected.splitlines(keepends=True))
    if len(sorted_b) != len(b):
        with pytest.raises(ValueError):
            bed_ops.merge_lines(bed_ops.sort_lines(b), (4, 5, 6), ("distinct",) * 3, strand=True)
    for mode in ("v", "u", "c"):
        for strand in (False, True):
            args = ["intersect", "-a", str(pa), "-b", str(pb), "-" + mode] + (["-s"] if strand else [])
            run = subprocess.run([BEDTOOLS] + args, capture_output=True, text=True)
            if run.returncode:  # a zero-length record at 0 reaches -1 when widened
                with pytest.raises(ValueError):
                    bed_ops.intersect_lines(a_lines, b, mode, strand=strand)
                continue
            assert "".join(bed_ops.intersect_lines(a_lines, b, mode, strand=strand)) == run.stdout, (mode, strand)


def test_read_records_skips_headers_and_reads_gzip(tmp_path):
    import gzip
    text = "track name=x\n#c\nbrowser p\n\nchr1\t1\t2\tn\t0\t+\nchr2\t3\t4\tm\t0\t-"
    plain, gz = tmp_path / "p.bed", tmp_path / "p.bed.gz"
    plain.write_text(text)
    with gzip.open(gz, "wt") as fh:
        fh.write(text)
    expected = ["chr1\t1\t2\tn\t0\t+\n", "chr2\t3\t4\tm\t0\t-\n"]
    assert bed_ops.read_records(str(plain)) == expected == bed_ops.read_records(str(gz))
