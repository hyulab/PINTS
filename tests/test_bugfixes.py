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


