#  PINTS: Peak Identifier for Nascent Transcripts Sequencing
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
"""Command-line options that never reached the engine, --model Poisson, fractional bigWig values and pairs
found on one strand only."""
import logging
import os
import runpy
import sys
import tempfile

import numpy as np
import pandas as pd

import pints.calling_engine as ce
import pints.io_engine as io
import pints.stats_engine as se

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALLER = os.path.join(ROOT, "scripts", "pints_caller")


def _cli_kwargs(monkeypatch, tmp_path, extra):
    """Run scripts/pints_caller's argument parsing with peak_calling stubbed out; return its keyword arguments."""
    captured = {}
    monkeypatch.setattr(ce, "peak_calling", lambda *a, **kw: captured.update(kw))
    bw = tmp_path / "x.bw"
    bw.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["pints_caller", "--bw-pl", str(bw), "--bw-mn", str(bw), "--save-to",
                                      str(tmp_path), "--file-prefix", "t", "--dont-check-updates"] + extra)
    old_hook = sys.excepthook
    try:
        runpy.run_path(CALLER, run_name="__main__")
    finally:
        sys.excepthook = old_hook
    return captured


def test_focused_chrom_reaches_engine(monkeypatch, tmp_path):
    """--focused-chrom is stored as focused_chrom; the engine read only highlight_chromosome."""
    assert ce._highlight_chromosome(_cli_kwargs(monkeypatch, tmp_path, ["--focused-chrom", "chr7"])) == "chr7"
    assert ce._highlight_chromosome(_cli_kwargs(monkeypatch, tmp_path, [])) == "chr1"
    assert ce._highlight_chromosome({"highlight_chromosome": "chr2"}) == "chr2"


def test_stringent_pairs_only_reaches_engine(monkeypatch, tmp_path):
    """--stringent-pairs-only is stored as stringent_pairs_only; the engine read only stringent_only."""
    assert ce._stringent_only(_cli_kwargs(monkeypatch, tmp_path, ["--stringent-pairs-only"])) is True
    assert ce._stringent_only(_cli_kwargs(monkeypatch, tmp_path, [])) is False
    assert ce._stringent_only({"stringent_only": True}) is True


def test_peak_width_reaches_find_peaks(monkeypatch, tmp_path):
    """--peak-width was parsed but never passed to scipy.signal.find_peaks."""
    assert _cli_kwargs(monkeypatch, tmp_path, ["--peak-width", "5"])["peak_width"] == 5
    w = np.zeros(200, dtype=np.int32)
    w[20] = 9  # 1-bp spike
    w[100:130] = np.r_[np.arange(1, 16), np.arange(15, 0, -1)]  # broad peak
    legacy = ce.cut_peaks(w, 0.3, 3)
    assert legacy == ce.cut_peaks(w, 0.3, 3, peak_width=None)
    wide = ce.cut_peaks(w, 0.3, 3, peak_width=5)
    assert any(s <= 20 < e for s, e in legacy) and not any(s <= 20 < e for s, e in wide)
    assert any(s <= 115 < e for s, e in wide)


def test_poisson_model_lrt():
    """Poisson did not implement the abstract lrt(), so --model Poisson raised TypeError."""
    from scipy.stats import poisson, chi2
    rng = np.random.default_rng(3)
    a, b = rng.poisson(2.0, 200), rng.poisson(3.0, 300)
    ll = lambda x: poisson.logpmf(x, x.mean()).sum()
    w, p = se.Poisson().lrt(a, b)
    w_ref = 2 * (ll(a) + ll(b) - ll(np.concatenate([a, b])))
    assert np.isclose(w, w_ref) and np.isclose(p, chi2.sf(w_ref, 1))
    assert se.Poisson().lrt(np.zeros(5), np.zeros(5)) == (0., 1.)
    ps = [se.Poisson().lrt(rng.poisson(2, 100), rng.poisson(2, 300))[1] for _ in range(400)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09  # calibrated under the null


def test_fractional_bigwig_values_are_reported(tmp_path, caplog):
    import pyBigWig
    path = str(tmp_path / "f.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries("chr1", list(range(100, 110)), values=[0.5] * 10, span=1)
    bw.close()
    with caplog.at_level(logging.WARNING):
        cov, _ = io._get_coverage_bw(path, "chr", str(tmp_path), "f")
    assert any("non-integer" in r.getMessage() for r in caplog.records)
    assert np.load(cov["chr1"]).sum() == 0  # values are still truncated, as before


def _peaks(rows, strand):
    out = [dict(chromosome="chr1", start=s, end=e, name=".", padj=padj, strand=strand, reads=reads,
                pval=padj / 10, mu_0=.1, pi_0=.1, mu_1=1., pi_1=.1, var_1=.1, ler_1=0, ler_2=0, ler_3=0,
                non_zeros=3, summit=(s + e) // 2, summit_val=5) for s, e, padj, reads in rows]
    return pd.DataFrame(out, columns=ce.COMMON_HEADER)


def _pair(pl, mn):
    with tempfile.TemporaryDirectory() as d:
        pre = os.path.join(d, "s")
        dfs = {"pl": _peaks(pl, "+"), "mn": _peaks(mn, "-")}
        for lab, df in dfs.items():
            fn = pre + "_%s.bed" % lab
            df.sort_values(["chromosome", "start"]).to_csv(fn, sep="\t", index=False, header=False)
            io.index_bed_file(fn)
        (bid, _, _), _ = ce.pair(pre, dfs, 0.1, False)
        return pd.read_csv(bid, sep="\t", header=None)


def test_pairs_found_on_one_strand_only():
    """Only the forward strand's significant peaks find partners: the pairs were written unmerged and parsed
    with the wrong columns, so the chromosome column held a peak end."""
    b = _pair([(1000, 1050, 1e-5, 50), (9000, 9050, 1e-5, 30)], [(800, 850, 0.9, 40), (8850, 8900, 0.9, 5)])
    assert list(b.iloc[:, 0]) == ["chr1", "chr1"]
    assert [tuple(r) for r in b.iloc[:, 1:3].to_numpy()] == [(800, 1050), (8850, 9050)]


def test_parallel_bigwig_loader_reports_fractional_values(tmp_path, caplog):
    """The block-wise loader used by peak calling warns as _get_coverage_bw does."""
    import pyBigWig
    path = str(tmp_path / "f.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries("chr1", list(range(100, 110)), values=[0.5] * 10, span=1)
    bw.close()
    with caplog.at_level(logging.WARNING):
        total = io._bw_chrom_worker(path, "chr1", 1000, np.int32, str(tmp_path / "f_chr1.npy"))
    assert total == 0 and any("non-integer" in r.getMessage() for r in caplog.records)


def test_peak_width_reaches_blockwise_element_definition(monkeypatch, tmp_path):
    seen = []
    real = ce.cut_peaks
    monkeypatch.setattr(ce, "cut_peaks", lambda *a, **kw: seen.append(kw.get("peak_width")) or real(*a, **kw))
    cov = np.zeros(50_000, dtype=np.int32)
    cov[1000:1030] = 5
    fn = str(tmp_path / "c.npy")
    np.save(fn, cov)
    ce.unified_element_definition([{"chr1": fn}], "chr1", "+", str(tmp_path / "s_pl.bed"), peak_width=5)
    assert seen and set(seen) == {5}
