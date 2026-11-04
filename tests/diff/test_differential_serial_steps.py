"""The steps that ran on one or two processes after evaluation: independent filtering with one quantile call
equals pints_ref, the per-chromosome peak index equals tabix, pairing split by chromosome writes the files of
the per-strand pairing, and strand tables read on the pool equal a read in the parent."""
import logging
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import pysam
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints.stats_engine as se
import pints_ref.stats_engine as rse
from test_differential import N
from test_differential_pairing import _peak_lines, _bgzip

FAST = settings(max_examples=N * 2, deadline=None, suppress_health_check=list(HealthCheck))

@pytest.fixture(scope="module")
def pool():
    with mp.get_context("fork").Pool(3) as p:
        yield p


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([0, 1, 7, 300, 3000]))
def test_independent_filtering_matches_ref(seed, n):
    rng = np.random.default_rng(seed)
    reads = rng.negative_binomial(1, float(rng.choice([0.02, 0.2, 0.7])), n)
    df = pd.DataFrame({"chromosome": "chr1", "start": np.arange(n), "end": np.arange(n) + 5, "name": ".",
                       "pval": rng.beta(0.3, 1, n), "strand": "+", "reads": reads})
    log = logging.getLogger("test")
    got = se.independent_filtering(df.copy(), fdr_target=0.1, logger=log)
    ref = rse.independent_filtering(df.copy(), fdr_target=0.1, logger=log)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert got["padj"].to_numpy().tobytes() == ref["padj"].to_numpy().tobytes()


def test_scalar_and_list_quantiles_agree():
    rng = np.random.default_rng(3)
    qs = np.arange(0, 1, 0.005)
    for s in (pd.Series(rng.negative_binomial(1, 0.05, 999)), pd.Series(rng.random(500) * 100),
              pd.Series(rng.integers(0, 5, 300)).astype(object), pd.Series([], dtype="int64"), pd.Series([4])):
        vec = s.quantile(qs).to_numpy()
        for q, v in zip(qs, vec):
            a = s.quantile(q)
            assert np.asarray(a).tobytes() == np.asarray(v).tobytes() and type(a) is type(v), (s.dtype, q)


@FAST
@given(st.integers(0, 2 ** 31 - 1))
def test_peak_index_from_tabix_matches_tabix(seed):
    rng = np.random.default_rng(seed)
    lines = _peak_lines(rng, "-", 60)
    if not lines:
        return
    with tempfile.TemporaryDirectory() as d:
        gz = _bgzip(d, "mn.bed", lines)
        tbx = pysam.TabixFile(gz, index=gz + ".csi")
        present = sorted(set(ln.split("\t", 1)[0] for ln in lines))
        chroms = sorted(set(rng.choice(present, int(rng.integers(1, len(present) + 1)))))
        idx = ce.PeakBedIndex.from_tabix(gz, chroms)
        assert idx is not None
        for c, qs, qe in [(c, int(a), int(a) + int(w)) for c, a, w in
                          zip(rng.choice(chroms, 80), rng.integers(0, 35_000, 80), rng.integers(0, 4_000, 80))]:
            ref = [tuple(r) for r in tbx.fetch(c, qs, qe, parser=pysam.asTuple())]
            assert [tuple(r) for r in idx.fetch(c, qs, qe, parser=pysam.asTuple())] == ref, (c, qs, qe)
        missing = ce.PeakBedIndex.from_tabix(gz, chroms + ["chrMissing"])
        for index in (tbx, missing):
            with pytest.raises(ValueError):
                index.fetch("chrMissing", 0, 10)


def test_peak_index_from_tabix_declines_region_like_names(tmp_path):
    gz = _bgzip(str(tmp_path), "mn.bed", ["HLA-A*01:01\t1\t5\t.\t0.1\t-\t3\t0.1\n"])
    assert ce.PeakBedIndex.from_tabix(gz, ["HLA-A*01:01"]) is None


def _strands(rng, d):
    """Candidate files for both strands and the significant peaks pair() would hand to _pair_strand."""
    names = list(ce.COMMON_HEADER)
    strands = []
    for label, anti, strand in (("pl", "mn", "+"), ("mn", "pl", "-")):
        cand = _peak_lines(rng, strand, 120)
        if not cand:
            return None
        _bgzip(d, "s_%s.bed" % label, cand)
    for label, anti, strand in (("pl", "mn", "+"), ("mn", "pl", "-")):
        sig = [ln.rstrip("\n").split("\t") for ln in _peak_lines(rng, strand, 50)]
        sig_bins = pd.DataFrame([r[:len(names)] for r in sig], columns=names) if sig else pd.DataFrame(columns=names)
        strands.append((os.path.join(d, "s"), label, anti, sig_bins, 0.1, bool(rng.integers(0, 2)), "", {}))
    return strands


@FAST
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 3, 40, 10 ** 6]))
def test_pairing_by_chromosome_matches_per_strand(pool, seed, chunk_lines):
    outs = {}
    for how in ("strand", "chrom"):
        with tempfile.TemporaryDirectory() as d:
            strands = _strands(np.random.default_rng(seed), d)
            if strands is None:
                return
            if how == "strand":
                files = sum((ce._pair_strand(*a) for a in strands), [])
            else:
                old, ce._PAIR_CHUNK_LINES = ce._PAIR_CHUNK_LINES, chunk_lines
                try:
                    files = ce._pair_strands_by_chromosome(pool, strands)
                finally:
                    ce._PAIR_CHUNK_LINES = old
            outs[how] = ([os.path.relpath(f, d) for f in files],
                         [open(f, "rb").read() for f in files], sorted(os.listdir(d)))
    assert outs["chrom"] == outs["strand"]


def test_read_peak_table_on_pool_equals_parent(pool, tmp_path):
    rng = np.random.default_rng(11)
    gz = _bgzip(str(tmp_path), "s_pl.bed", _peak_lines(rng, "+", 400))
    names = list(ce.COMMON_HEADER)
    raw = pd.read_csv(gz, sep="\t", header=None).iloc[:, :len(names)]
    raw.to_csv(str(tmp_path / "t.bed"), sep="\t", header=False, index=False)
    here = ce._read_peak_table(str(tmp_path / "t.bed"), 2000)
    there = pool.apply_async(ce._read_peak_table, (str(tmp_path / "t.bed"), 2000)).get()
    pd.testing.assert_frame_equal(here, there, check_exact=True)
