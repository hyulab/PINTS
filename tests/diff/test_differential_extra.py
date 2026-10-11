"""Additional differential tests closing hot-path coverage gaps:
multi-chromosome peaks_single_strand (serial vs Pool, empty chromosome, fc_cutoff / QC branches),
pair(keep_sticks=False), and the alpha dry-run (cut_peaks_dry_run) with a synthetic GTF."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

import pints.calling_engine as ce
import pints.stats_engine as se
import pints_ref.calling_engine as rce
import pints_ref.stats_engine as rse
from helpers import synth_coverage
from test_differential import same, file_bytes, N

SLOW = settings(max_examples=max(3, N // 3), deadline=None, suppress_health_check=list(HealthCheck))


def _globals(iqr="bgIQR"):
    ce.stat_tester, rce.stat_tester = se.ZIP(), rse.ZIP()
    ce.iqr_obj, rce.iqr_obj = getattr(se, iqr)(), getattr(rse, iqr)()


def _multichrom_covs(seed):
    rng = np.random.default_rng(seed)
    covs = {}
    for k, L in enumerate((40_000, 23_456, 18_001)):
        covs["chr%d" % (k + 1)] = synth_coverage(rng, length=L, n_peaks=int(rng.integers(5, 40)),
                                                 dtype=np.int16)
    covs["chrE"] = np.zeros(5_000, dtype=np.int8)                  # no signal -> subpeak def None
    return covs


def _run_pss(mod, d, covs, strand, kw):
    files = {}
    for c, v in covs.items():
        files[c] = os.path.join(d, "%s.npy" % c)
        np.save(files[c], v)
    lab = "pl" if strand == "+" else "mn"
    spd = {c: mod.unified_element_definition([files], c, strand, os.path.join(d, "s_%s.bed" % lab))
           for c in files}
    out = mod.peaks_single_strand(files, os.path.join(d, "s_1_%s.bed" % lab), spd, strand, 0.1, **kw)
    return file_bytes(out)


@SLOW
@given(seed=st.integers(0, 2 ** 31 - 1), strand=st.sampled_from(["+", "-"]),
       fc=st.sampled_from([0, 1.5]), qc=st.booleans(), iqr=st.sampled_from(["bgIQR", "pkIQR"]),
       sens=st.booleans())
def test_peaks_single_strand_multichrom(seed, strand, fc, qc, iqr, sens):
    _globals(iqr)
    covs = _multichrom_covs(seed)
    kw = dict(fc_cutoff=fc, disable_qc=qc, sensitive=sens, thread_n=1)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        same(_run_pss(ce, da, covs, strand, kw), _run_pss(rce, db, covs, strand, kw))


def test_peaks_single_strand_pool_equals_serial(monkeypatch):
    """thread_n > 1 must give byte-identical output to the serial path (ref and opt)."""
    # The reference relies on fork-inherited module globals, so pin its Pool to fork
    # where spawn is the default (macOS, Python >= 3.14 on Linux).
    import multiprocessing as mp
    monkeypatch.setattr(rce, "Pool", mp.get_context("fork").Pool)
    _globals()
    covs = _multichrom_covs(42)
    res = []
    for mod in (rce, ce):
        for t in (1, 3):
            with tempfile.TemporaryDirectory() as d:
                res.append(_run_pss(mod, d, covs, "+", dict(thread_n=t)))
    for r in res[1:]:
        same(res[0], r)


@pytest.mark.parametrize("keep_sticks", [False])
def test_pair_keep_sticks(keep_sticks):
    rng = np.random.default_rng(3)
    dfs = {}
    for strand, lab in (("+", "pl"), ("-", "mn")):
        n = 150
        s = np.sort(rng.choice(100_000, n, replace=False)); e = s + rng.integers(1, 60, n)
        padj = np.where(rng.random(n) < .5, rng.uniform(0, .05, n), rng.uniform(0, 1, n))
        dfs[lab] = pd.DataFrame({"chromosome": "chr1", "start": s, "end": e, "name": ".", "padj": padj,
                                 "strand": strand, "reads": rng.integers(1, 50, n), "pval": padj / 3,
                                 "mu_0": .1, "pi_0": .1, "mu_1": 1., "pi_1": .1, "var_1": .1, "ler_1": 0,
                                 "ler_2": 0, "ler_3": 0, "non_zeros": rng.integers(1, 4, n),
                                 "summit": s, "summit_val": rng.integers(1, 9, n)}).loc[:, ce.COMMON_HEADER]
    res = []
    for mod in (ce, rce):
        with tempfile.TemporaryDirectory() as d:
            for lab, df in dfs.items():
                fn = os.path.join(d, "s_%s.bed" % lab)
                df.to_csv(fn, sep="\t", index=False, header=False)
                mod.index_bed_file(fn)
            (b, dv, u), _ = mod.pair(os.path.join(d, "s"), {k: v.copy() for k, v in dfs.items()}, 0.1, False,
                                     keep_sticks=keep_sticks)
            res.append([file_bytes(x) for x in (b, dv, u)])
    same(res[0], res[1])


def _synthetic_gtf(path, chrom, tss_list):
    with open(path, "w") as fh:
        for i, (tss, strand) in enumerate(tss_list):
            s, e = (tss + 1, tss + 3000) if strand == "+" else (tss - 3000, tss + 1)
            attr = 'gene_id "G%d"; transcript_id "T%d"; gene_type "protein_coding"; gene_name "g%d";' % (i, i, i)
            fh.write("%s\tsyn\ttranscript\t%d\t%d\t.\t%s\t.\t%s\n" % (chrom, s, e, strand, attr))


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_cut_peaks_dry_run(seed):
    import warnings
    rng = np.random.default_rng(seed)
    L = 400_000
    pl = np.zeros(L, dtype=np.int16); mn = np.zeros(L, dtype=np.int16)
    tss = []
    for p in rng.choice(np.arange(5_000, L - 5_000, 4_000), 60, replace=False):
        strand = "+" if rng.random() < .5 else "-"
        tgt = pl if strand == "+" else mn
        for off in rng.integers(-60, 60, int(rng.integers(1, 4))):   # clustered TSSs
            w = int(rng.integers(3, 25))
            tgt[p + off:p + off + w] += rng.poisson(rng.integers(5, 40), w).astype(np.int16)
        tss.append((int(p), strand))
    out = []
    for mod in (ce, rce):
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gtf = os.path.join(d, "a.gtf"); _synthetic_gtf(gtf, "chr1", tss)
            fp, fm = os.path.join(d, "p.npy"), os.path.join(d, "m.npy")
            np.save(fp, pl); np.save(fm, mn)
            out.append(mod.cut_peaks_dry_run(gtf, [{"chr1": fp}], [{"chr1": fm}], highlight_chromosome="chr1"))
    same(out[0], out[1])
