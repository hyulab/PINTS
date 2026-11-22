"""Differential tests for pairing with the in-memory PeakBedIndex: fetch results equal pysam.TabixFile
on the same bgzipped file, and merge_opposite_peaks output bytes equal pints_ref."""
import os
import tempfile

import numpy as np
import pysam
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints_ref.calling_engine as rce
from test_differential import N

FAST = settings(max_examples=N * 4, deadline=None, suppress_health_check=list(HealthCheck))
CHROMS = ["chr1", "chr10", "chr2", "chrUn_KI270755v1", "chrX"]


def _peak_lines(rng, strand, n_per_chrom, ties=True):
    lines = []
    for c in CHROMS[:int(rng.integers(1, len(CHROMS) + 1))]:
        n = int(rng.integers(0, n_per_chrom + 1))
        pool = 30_000 if ties else max(30_000, n * 2)
        starts = np.sort(rng.choice(pool, n, replace=ties))
        lens = rng.integers(1, rng.choice([3, 50, 400, 3_000]), n)
        for s, ln in zip(starts, lens):
            s, e = int(s), int(s + ln)
            padj, pval = float(rng.choice([rng.random(), rng.random() * 0.2])), float(rng.random() * 0.3)
            reads = int(rng.integers(1, 200))
            summit = int(rng.integers(s, e))
            row = (c, s, e, ".", padj, strand, reads, pval, rng.random(), 0.1, 2.5, 0.2, 1.5, 1, 0, 1,
                   int(rng.integers(1, 9)), summit, int(rng.integers(1, 30)))
            lines.append("\t".join(map(str, row)) + "\n")
    return lines


def _bgzip(d, name, lines):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.writelines(lines)
    ce.index_bed_file(p)
    return p + ".gz"


@FAST
@given(st.integers(0, 2 ** 31 - 1))
def test_peak_index_matches_tabix(seed):
    rng = np.random.default_rng(seed)
    lines = _peak_lines(rng, "-", 60)
    if not lines:
        return  # tabix cannot index an empty file
    with tempfile.TemporaryDirectory() as d:
        gz = _bgzip(d, "mn.bed", lines)
        idx = ce.PeakBedIndex.from_file(gz)
        assert idx is not None
        tbx = pysam.TabixFile(gz, index=gz + ".csi")
        queries = [(c, int(a), int(a) + int(w)) for c, a, w in
                   zip(rng.choice(CHROMS + ["chrMissing"], 60), rng.integers(0, 35_000, 60),
                       rng.integers(0, 4_000, 60))]
        queries += [("chr1", 0, 0), ("chr1", 0, 1), ("chr1", 5, 3), ("chrMissing", 4, 4), ("chr1", -1, 5),
                    ("chr1", -3, -1), ("chrMissing", 0, 1)]
        for c, qs, qe in queries:
            try:
                ref = [tuple(r) for r in tbx.fetch(c, qs, qe, parser=pysam.asTuple())]
            except ValueError:
                ref = ValueError
            try:
                got = [tuple(r) for r in idx.fetch(c, qs, qe, parser=pysam.asTuple())]
            except ValueError:
                got = ValueError
            assert ref == got, (c, qs, qe)


def test_peak_index_falls_back_outside_supported_subset(tmp_path):
    import gzip
    mk = lambda rows: "".join("%s\t%d\t%d\t.\t0.1\t+\t3\t0.1\n" % r for r in rows)
    cases = {
        "zero_len": mk([("chr1", 5, 5)]),
        "unsorted": mk([("chr1", 9, 12), ("chr1", 5, 7)]),
        "non_contiguous": mk([("chr1", 1, 2), ("chr2", 1, 2), ("chr1", 5, 7)]),
        "header": "#chrom\tstart\tend\n" + mk([("chr1", 1, 2)]),
        "crlf": mk([("chr1", 1, 2)]).replace("\n", "\r\n"),
        "no_final_newline": mk([("chr1", 1, 2)])[:-1],
        "non_digit": "chr1\t+1\t5\t.\n",
    }
    for name, text in cases.items():
        p = str(tmp_path / (name + ".bed.gz"))
        with gzip.open(p, "wt", newline="") as fh:
            fh.write(text)
        assert ce.PeakBedIndex.from_file(p) is None, name
    p = str(tmp_path / "ok.bed.gz")
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(mk([("chr1", 1, 2), ("chr1", 1, 9), ("chr2", 0, 3)]))  # ties are fine: file order is kept
    assert ce.PeakBedIndex.from_file(p) is not None


def _run_merge(mod, d, tag, sig, cand, stringent, kw):
    outs = [os.path.join(d, "%s_%s.bed" % (tag, x)) for x in ("div", "bid", "single")]
    mod.merge_opposite_peaks(sig, cand, outs[0], outs[1], outs[2], 0.1, stringent_only=stringent, **kw)
    return [open(p, "rb").read() for p in outs]


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.booleans())
def test_merge_opposite_peaks_matches_ref(seed, stringent):
    rng = np.random.default_rng(seed)
    cand = _peak_lines(rng, "-", 80)
    sig = _peak_lines(rng, "+", 40)
    if not cand or not sig:
        return
    kw = dict(close_threshold=int(rng.choice([0, 50, 300, 1000])),
              min_len_opposite_peaks=int(rng.choice([0, 0, 5, 50])),
              div_size_min=int(rng.choice([0, 0, 20])), summit_dist_min=int(rng.choice([0, 0, 10, -10])))
    with tempfile.TemporaryDirectory() as d:
        gz = _bgzip(d, "mn.bed", cand)
        sig_p = os.path.join(d, "sig_pl.bed")
        with open(sig_p, "w") as fh:
            fh.writelines(sig)
        assert ce.PeakBedIndex.from_file(gz) is not None
        assert _run_merge(ce, d, "opt", sig_p, gz, stringent, kw) == _run_merge(rce, d, "ref", sig_p, gz,
                                                                                stringent, kw)


def test_open_peak_index_choice(tmp_path):
    rng = np.random.default_rng(7)
    gz = _bgzip(str(tmp_path), "mn.bed", _peak_lines(rng, "-", 200, ties=False) or ["chr1\t1\t2\t.\n"])
    assert isinstance(ce.open_peak_index(gz), ce.PeakBedIndex)
    assert isinstance(ce.open_peak_index(gz, n_queries=10 ** 6), ce.PeakBedIndex)
    assert isinstance(ce.open_peak_index(gz, n_queries=0), pysam.TabixFile)
