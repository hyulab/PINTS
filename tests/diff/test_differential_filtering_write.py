"""The pooled write of the stratified-filtering table (_ParallelIndexedBed) yields a .gz with the same lines,
and the same tabix lookups, as stratified_filtering, which sorts with bedtools before compressing."""
import gzip
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pandas as pd
import pysam
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
from test_differential import N

SLOW = settings(max_examples=max(5, N // 2), deadline=None, suppress_health_check=list(HealthCheck))
CHROMS = ["chr1", "chr10", "chr2", "chrUn_KI270755v1", "chrX", "HLA-A*01:01"]


@pytest.fixture(scope="module")
def pool():
    with mp.get_context("fork").Pool(3) as p:
        yield p


def _table(rng, n, ties=False):
    chrom = rng.choice(CHROMS, n)
    start = rng.integers(0, 10 ** 6 if not ties else 50, n)
    if not ties:  # unique (chromosome, start)
        keep = ~pd.DataFrame({"c": chrom, "s": start}).duplicated().to_numpy()
        chrom, start = chrom[keep], start[keep]
        n = len(chrom)
    end = start + rng.integers(1, rng.choice([4, 6, 300, 3000]), n)
    f = lambda: np.where(rng.random(n) < 0.05, np.nan, rng.random(n) * 10.0 ** rng.integers(-12, 3, n))
    rows = list(zip(chrom, start, end, ["p%d" % i for i in range(n)], rng.random(n) ** 3, rng.integers(0, 300, n),
                    f(), rng.random(n), f() + 0.5, rng.random(n), f(), start, rng.integers(0, 40, n),
                    rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 90, n)))
    return ce._bins_to_df(rows, "+")


def _lines(gz):
    with gzip.open(gz, "rb") as fh:
        return fh.read()


def _fetch_all(gz):
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    out = []
    for c in tbx.contigs:
        for a in (0, 1000, 250_000, 999_999):
            out.append((c, a, list(tbx.fetch(c, a, a + 20_000))))
    return out


def _both(pool, df, d, kwargs, n_chunks):
    os.makedirs(os.path.join(d, "a"))
    os.makedirs(os.path.join(d, "b"))
    a = ce.stratified_filtering(df.copy(), os.path.join(d, "a", "s_1_pl.bed"), 0.1, **kwargs)
    b = ce._ParallelIndexedBed(pool, df.copy(), os.path.join(d, "b", "s_1_pl.bed"), 0.1, kwargs, n_chunks).get()
    return a, b


@SLOW
@given(st.integers(0, 2 ** 31 - 1), st.sampled_from([1, 2, 7, 64]), st.sampled_from([50, 3000, 20_000]),
       st.booleans())
def test_parallel_write_equals_stratified_filtering(pool, seed, n_chunks, n, disable_small):
    rng = np.random.default_rng(seed)
    df = _table(rng, n)
    kwargs = {"disable_small": disable_small, "disable_qc": False}
    with tempfile.TemporaryDirectory() as d:
        a, b = _both(pool, df, d, kwargs, n_chunks)
        assert os.path.basename(a) == os.path.basename(b)
        assert _lines(a) == _lines(b)
        assert _fetch_all(a) == _fetch_all(b)
        assert os.path.exists(a + ".csi") and os.path.exists(b + ".csi")
        assert not os.path.exists(b[:-3])  # no plain file left behind
        assert sorted(os.listdir(os.path.join(d, "a"))) == sorted(os.listdir(os.path.join(d, "b")))  # .tpt too


@pytest.mark.parametrize("n", [0, 1])
def test_ties_and_tiny_tables_use_the_original_writer(pool, n):
    rng = np.random.default_rng(3)
    df = _table(rng, 4000, ties=True) if n == 1 else _table(rng, 50).iloc[:0]
    if n == 1:
        assert df.duplicated(["chromosome", "start"]).any()
    with tempfile.TemporaryDirectory() as d:
        r = ce._filter_frame_task(df.copy(), os.path.join(d, "x.bed"), 0.1, {})
        assert r[0] == "path"
    if n == 1:
        with tempfile.TemporaryDirectory() as d:
            a, b = _both(pool, df, d, {}, 4)
            assert _lines(a) == _lines(b)


def test_errors_surface_on_get(pool):
    df = _table(np.random.default_rng(5), 100).drop(columns=["mu_1"])
    with tempfile.TemporaryDirectory() as d:
        r = ce._ParallelIndexedBed(pool, df, os.path.join(d, "s.bed"), 0.1, {}, 4)
        with pytest.raises(KeyError):
            r.get()


@pytest.mark.parametrize("size", [0, 1, 0xff00 - 1, 0xff00, 0xff00 + 1, 3 * 0xff00 + 17])
@pytest.mark.parametrize("kind", ["text", "random"])
def test_bgzf_compress_round_trip(size, kind):
    rng = np.random.default_rng(size)
    data = rng.integers(0, 256, size, dtype=np.uint8).tobytes() if kind == "random" else \
        (b"chr1\t100\t200\tp\t0.5\n" * (size // 19 + 1))[:size]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.gz")
        with open(p, "wb") as fh:
            fh.write(ce.bgzf_compress(data[:size // 2]) + ce.bgzf_compress(data[size // 2:]) + ce._BGZF_EOF)
        assert gzip.decompress(open(p, "rb").read()) == data
        with pysam.BGZFile(p, "rb") as fh:
            assert fh.read() == data
