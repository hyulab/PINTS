"""Characterisation tests: local-environment refinement (LER), window testing, filtering, pairing."""
import gzip
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
import pytest

import pints.calling_engine as ce
from pints.stats_engine import ZIP, bgIQR, pkIQR, IQR
from helpers import synth_coverage, write_subpeak_tabix, write_candidate_gz

CHR = "chrT"


def _row(s, e, mu, name="p"):
    return (s, e, name, mu, 0.0, "+", 0.1, max(1, e - s), s, 3)


@pytest.fixture
def zip_globals():
    ce.stat_tester = ZIP()
    ce.iqr_obj = bgIQR()
    yield
    ce.stat_tester = None
    ce.iqr_obj = None


# ------------------------------------------------------------ atom_ler
def test_atom_ler_clipping_threshold_and_keys(tmp_path):
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(100, 120, 5.0), _row(150, 160, 0.5),
                                                       _row(190, 260, 3.0)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    q = defaultdict(int)
    se, re, dens = IQR.atom_ler(tbx, CHR, 110, 200, q, peak_in_bg_threshold=1, offset=7)
    assert se == [(0 + 7, 10 + 7), (80 + 7, 90 + 7)]
    assert re == [(110, 120), (190, 200)]
    assert dens == [5.0, 3.0]
    # keys use the *unclipped* length and the text value of mu as written with '%f'
    assert dict(q) == {"20-5.000000": 1, "10-0.500000": 1, "70-3.000000": 1}


def test_atom_ler_negative_query_start_clamped(tmp_path):
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(0, 10, 2.0)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    se, re, _ = IQR.atom_ler(tbx, CHR, -50, 20, defaultdict(int))
    assert re == [(0, 10)] and se == [(0, 10)]


# ------------------------------------------------------------ pkIQR
def test_pkiqr_masks_outlier_density(tmp_path):
    cov = np.zeros(1000, dtype=np.int32)
    cov[300:310] = 9
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(100, 105, 1.5), _row(200, 205, 1.5),
                                                       _row(300, 310, 10.0)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    env, n = pkIQR.remove_peaks_in_local_env(ZIP(), tbx, CHR, 0, 400, 600, 1000, 5, 1, cov, 0.1, {},
                                             peak_threshold=5.)
    assert n == 1
    assert env.shape[0] == 400 + 400 - 10 and env.max() == 0


def test_pkiqr_disable_ler_returns_raw_concat(tmp_path):
    cov = np.arange(1000, dtype=np.int32) % 3
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(100, 105, 1.5)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    env, n = pkIQR.remove_peaks_in_local_env(ZIP(), tbx, CHR, 0, 400, 600, 1000, 5, 1, cov, 0.1, {},
                                             disable_ler=True)
    assert n == 0 and np.array_equal(env, np.concatenate((cov[0:400], cov[600:1000])))


# ------------------------------------------------------------ bgIQR
def _bg_fixture(tmp_path, rng, n_sub):
    cov = synth_coverage(rng, length=20_000, n_peaks=0, edge_peaks=False)
    rows = []
    starts = np.linspace(3000, 17000, n_sub).astype(int)
    for i, s in enumerate(starts):
        h = 30 if i == 0 else 1
        cov[s:s + 10] += h
        rows.append(_row(int(s), int(s + 10), float(h)))
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, rows)
    return cov, pysam.TabixFile(gz, index=gz + ".csi"), rows


@pytest.mark.parametrize("n_sub", [2, 6])
def test_bgiqr_branches_mask_strong_peak_and_fill_cache(tmp_path, rng, n_sub):
    cov, tbx, rows = _bg_fixture(tmp_path, rng, n_sub)
    cache = {}
    env, n = bgIQR.remove_peaks_in_local_env(ZIP(), tbx, CHR, 2500, 9900, 10100, 17500, 5, 1, cov, 0.1, cache,
                                             enable_eler=True, peak_threshold=20.)
    assert n >= 1
    assert "%d-%d" % (rows[0][0], rows[0][1]) in cache and cache["%d-%d" % (rows[0][0], rows[0][1])] == 1
    assert set(cache.values()) <= {0, 1} and n == sum(cache.values())
    assert env.shape[0] == (9900 - 2500) + (17500 - 10100) - 10 * n
    # second call is served from cache and is identical
    env2, n2 = bgIQR.remove_peaks_in_local_env(ZIP(), tbx, CHR, 2500, 9900, 10100, 17500, 5, 1, cov, 0.1, cache,
                                               enable_eler=True, peak_threshold=20.)
    assert n2 == n and np.array_equal(env, env2)


def test_bgiqr_no_candidates_returns_concat(tmp_path, rng):
    cov = synth_coverage(rng, length=5000, n_peaks=0, edge_peaks=False)
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(4900, 4910, 0.5)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    env, n = bgIQR.remove_peaks_in_local_env(ZIP(), tbx, CHR, 0, 1000, 1100, 2000, 5, 1, cov, 0.1, {})
    assert n == 0 and np.array_equal(env, np.concatenate((cov[:1000], cov[1100:2000])))


# ------------------------------------------------------------ check_window
def test_check_window_short_or_empty_window_shortcut(zip_globals, tmp_path):
    cov = np.zeros(1000, dtype=np.int32)
    cov[10:12] = 3
    out = ce.check_window(10, 13, 2, 0, 0, cov, 1, 0, None, CHR, 0.1, {})
    assert out[0] == 1. and out[1] == 6 and out[4] == (0, 0, 0) and out[5] == 1.
    assert out[6] == pytest.approx(2 / 3)
    out = ce.check_window(100, 200, 0, 0, 0, cov, 1, 0, None, CHR, 0.1, {})
    assert out[0] == 1. and out[1] == 0


def test_check_window_strong_peak_significant(zip_globals, tmp_path, rng):
    cov = synth_coverage(rng, length=30_000, n_peaks=0, edge_peaks=False)
    cov[15000:15030] += 40
    gz = write_subpeak_tabix(tmp_path / "sp.bed", CHR, [_row(15000, 15030, 40.0)])
    tbx = pysam.TabixFile(gz, index=gz + ".csi")
    mu, var, pi, _, _ = ZIP().fit(cov[15000:15030])
    p, wv, mu0, pi0, ler, lrtp, dens = ce.check_window(15000, 15030, mu, var, pi, cov, 1, 0.0, tbx, CHR, 0.1, {},
                                                       top_peak_mu=20.)
    assert p < 1e-10 and lrtp < 1e-10 and wv == cov[15000:15030].sum() and len(ler) == 3


# ------------------------------------------------------------ chromosome-level pipeline
@pytest.fixture
def chrom_run(zip_globals, tmp_path, rng):
    cov = synth_coverage(rng, length=80_000, n_peaks=60)
    npy = str(tmp_path / "cov.npy")
    np.save(npy, cov)
    out = str(tmp_path / "s_pl.bed")
    sp = ce.unified_element_definition([{CHR: npy}], CHR, "+", out)
    return cov, npy, out, sp


def test_unified_element_definition_output(chrom_run):
    cov, npy, out, sp = chrom_run
    assert sp.endswith(".gz") and os.path.exists(sp + ".csi")
    rows = [l.rstrip("\n").split("\t") for l in gzip.open(sp, "rt")]
    starts = [int(r[1]) for r in rows]
    assert starts == sorted(starts)
    assert [r[3] for r in rows] == ["%s-%d" % (CHR, i + 1) for i in range(len(rows))]
    for r in rows:
        s, e, summit = int(r[1]), int(r[2]), int(r[4])
        assert 0 <= s < e <= len(cov) and s <= summit < e
        assert summit == s + int(np.argmax(cov[s:e]))


def test_unified_element_definition_empty_returns_none(zip_globals, tmp_path):
    npy = str(tmp_path / "z.npy")
    np.save(npy, np.zeros(1000, dtype=np.int8))
    assert ce.unified_element_definition([{CHR: npy}], CHR, "+", str(tmp_path / "z_pl.bed")) is None


def test_check_window_chromosome_schema(chrom_run):
    cov, npy, out, sp = chrom_run
    df = ce.check_window_chromosome(npy, out, "+", CHR, sp, 0.1)
    assert list(df.columns) == ["chromosome", "start", "end", "name", "pval", "strand", "reads", "mu_0", "pi_0",
                                "mu_1", "pi_1", "var_1", "summit", "summit_val", "ler_1", "ler_2", "ler_3",
                                "non_zeros"]
    assert (df.reads > 0).all() and df.pval.between(0, 1).all()
    n_cand = sum(1 for _ in gzip.open(sp, "rt"))
    assert len(df) <= n_cand
    for _, r in df.iterrows():
        assert r.reads == cov[r.start:r.end].sum()


def test_stratified_filtering_writes_indexed_bed(chrom_run, logger):
    cov, npy, out, sp = chrom_run
    df = ce.check_window_chromosome(npy, out, "+", CHR, sp, 0.1)
    res = ce.stratified_filtering(df, out, 0.1)
    assert res == out + ".gz" and os.path.exists(res + ".csi")
    got = pd.read_csv(res, sep="\t", header=None, names=ce.COMMON_HEADER)
    assert len(got) == len(df) and got.padj.between(0, 1).all()
    assert got.sort_values(["chromosome", "start"]).index.equals(got.index)


# ------------------------------------------------------------ pairing
def _cand_bed(path, rows):
    pd.DataFrame(rows, columns=ce.COMMON_HEADER).to_csv(path, sep="\t", index=False, header=False)
    pysam.tabix_index(path, preset="bed", force=True, csi=True)
    return path + ".gz"


def _rec(s, e, strand, padj, pval, reads, summit, sv):
    return [CHR, s, e, ".", padj, strand, reads, pval, 0, 0, 0, 0, 0, 0, 0, 0, 0, summit, sv]


def test_merge_opposite_peaks_pairs_upstream_minus_peak(tmp_path):
    mn = _cand_bed(str(tmp_path / "mn.bed"), [_rec(800, 810, "-", 0.01, 0.001, 20, 805, 9),
                                              _rec(5000, 5010, "-", 0.01, 0.001, 20, 5005, 9)])
    sig = tmp_path / "sig.bed"
    sig.write_text("\t".join(map(str, _rec(1000, 1010, "+", 0.02, 0.002, 30, 1005, 12))) + "\n")
    div, bid, single = (str(tmp_path / x) for x in ("div.bed", "bid.bed", "single.bed"))
    ce.merge_opposite_peaks(str(sig), mn, div, bid, single, 0.1)
    b = open(bid).read().strip().split("\t")
    assert b[:3] == [CHR, "800", "1010"] and b[-1] == "Stringent(qval)"
    assert float(b[6]) == 50.0 and b[11] == "1005" and b[13] == "805"
    assert open(div).read() == open(bid).read()      # fwd summit - rev summit >= 0
    assert open(single).read() == ""


def test_merge_opposite_peaks_singleton(tmp_path):
    mn = _cand_bed(str(tmp_path / "mn.bed"), [_rec(5000, 5010, "-", 0.5, 0.2, 2, 5005, 1)])
    sig = tmp_path / "sig.bed"
    line = "\t".join(map(str, _rec(1000, 1010, "+", 0.02, 0.002, 30, 1005, 12))) + "\n"
    sig.write_text(line)
    div, bid, single = (str(tmp_path / x) for x in ("div.bed", "bid.bed", "single.bed"))
    ce.merge_opposite_peaks(str(sig), mn, div, bid, single, 0.1)
    assert open(single).read() == line and open(bid).read() == ""


def test_merge_opposite_peaks_relaxed_and_convergent(tmp_path):
    # opposite peak downstream-overlapping (convergent): bidirectional but not divergent
    mn = _cand_bed(str(tmp_path / "mn.bed"), [_rec(1002, 1030, "-", 0.5, 0.3, 4, 1025, 2)])
    sig = tmp_path / "sig.bed"
    sig.write_text("\t".join(map(str, _rec(1000, 1010, "+", 0.02, 0.002, 30, 1005, 12))) + "\n")
    div, bid, single = (str(tmp_path / x) for x in ("div.bed", "bid.bed", "single.bed"))
    ce.merge_opposite_peaks(str(sig), mn, div, bid, single, 0.1)
    b = open(bid).read().strip().split("\t")
    assert b[-1] == "Relaxed" and open(div).read() == ""
