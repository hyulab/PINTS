"""Differential tests for the in-memory coverage store: tracks built by each producer hold exactly what the
reference's .npy dumps hold (values, dtypes, chromosome order, read totals), and no dumps are written."""
import os
import tempfile

import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.io_engine as io
import pints_ref.io_engine as rio
from helpers import make_bam
from test_differential import same, N
from test_differential_parallel import mixed_reads

SLOW = settings(max_examples=max(3, N // 2), deadline=None, suppress_health_check=list(HealthCheck))
LIBS = ["R_5", "R_3", "R1_5", "R1_3", "R2_5", "R2_3", "PROcap", "PROseq", "CoPRO", "mNETseq"]


def _assert_tracks_equal_dumps(tracks, dumps):
    assert list(tracks) == list(dumps)  # same chromosomes, same (header) order
    for c in dumps:
        ref = np.load(dumps[c])
        t = tracks[c]
        assert t.length == ref.shape[0] and t.dtype == ref.dtype, (c, t, ref.dtype)
        assert np.array_equal(t.dense(), ref), c


def _npys(d):
    return [f for f in os.listdir(d) if f.endswith(".npy")]


@SLOW
@given(mixed_reads(), st.sampled_from(LIBS), st.booleans(), st.sampled_from([1, 3]),
       st.sampled_from([(), ("ctg",)]), st.booleans())
def test_bam_tracks_equal_reference_dumps(rr, lib, rc, threads, filters, fast):
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
        pl, mn, total, cache = io._read_signal(bam, lib, "", os.path.join(d, "a"), "p", reverse_complement=rc,
                                               thread_n=threads, filters=filters, fast_bam=fast, as_tracks=True)
        b = rio.get_read_signal(bam, lib, "", os.path.join(d, "b"), "p", reverse_complement=rc, filters=filters)
        same(total, b[2])
        _assert_tracks_equal_dumps(pl, b[0])
        _assert_tracks_equal_dumps(mn, b[1])
        assert cache == [] and _npys(os.path.join(d, "a")) == []


@SLOW
@given(mixed_reads(), st.sampled_from(["R_5", "R1_3", "PROseq"]), st.sampled_from([2, 8]),
       st.sampled_from([300, 4000]), st.integers(0, 2 ** 31 - 1))
def test_bam_region_tracks_equal_reference_dumps(rr, lib, threads, span, seed):
    """Contigs split into regions whose 5' ends spill into neighbouring regions."""
    contigs, reads = rr
    rng = np.random.default_rng(seed)
    for c, L in contigs[:1]:
        for b in range(span, L, span):
            for pos in (b - 25, b - 1, b, b - 30):
                reads.append(dict(contig=c, pos=pos, cigar="30M", reverse=bool(rng.random() < 0.5), mapq=60))
    old = io.BAM_SPLIT_MIN_SPAN
    io.BAM_SPLIT_MIN_SPAN = span
    try:
        with tempfile.TemporaryDirectory() as d:
            bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
            os.makedirs(os.path.join(d, "a")); os.makedirs(os.path.join(d, "b"))
            pl, mn, total, _ = io._read_signal(bam, lib, "", os.path.join(d, "a"), "p", thread_n=threads,
                                               as_tracks=True)
            b = rio.get_read_signal(bam, lib, "", os.path.join(d, "b"), "p")
            same(total, b[2])
            _assert_tracks_equal_dumps(pl, b[0])
            _assert_tracks_equal_dumps(mn, b[1])
            assert _npys(os.path.join(d, "a")) == []
    finally:
        io.BAM_SPLIT_MIN_SPAN = old


@SLOW
@given(mixed_reads(), st.sampled_from(["PROcap", "R2_3"]))
def test_bam_tracks_from_the_npy_cache(rr, lib):
    """Dumps left in the output directory are reused (as tracks) and reported for deletion."""
    contigs, reads = rr
    with tempfile.TemporaryDirectory() as d:
        bam = make_bam(os.path.join(d, "x.bam"), contigs, reads)
        out = os.path.join(d, "a"); os.makedirs(out)
        dumps = io.get_read_signal(bam, lib, "", out, "p")
        before = sorted(_npys(out))
        pl, mn, total, cache = io._read_signal(bam, lib, "", out, "p", as_tracks=True)
        ref = rio.get_read_signal(bam, lib, "", out, "p")  # the reference also reads the cache
        same(total, ref[2])
        _assert_tracks_equal_dumps(pl, dumps[0])
        _assert_tracks_equal_dumps(mn, dumps[1])
        assert sorted(cache) == sorted(list(dumps[0].values()) + list(dumps[1].values()))
        assert sorted(_npys(out)) == before
