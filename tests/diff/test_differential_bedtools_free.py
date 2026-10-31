"""Opt-in paths that used pybedtools, against the reference (pybedtools + bedtools): borrowing elements from
other replicates, boundary extension with a promoter BED, and pints_counter's union of regions."""
import gzip
import importlib.machinery
import importlib.util
import os
import random
import shutil

import pytest

import pints.calling_engine as ce
import pints.extension_engine as ee
import pints_ref.calling_engine as rce
import pints_ref.extension_engine as ree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = os.path.join(ROOT, "tests", "datasets")
pytestmark = pytest.mark.skipif(shutil.which("bedtools") is None, reason="the reference needs bedtools")


def _pair_records(rng, n, chroms, span):
    """Records shaped like *_bidirectional/_divergent_peaks.bed (8 columns), many overlapping and tied."""
    out = []
    for _ in range(n):
        s = rng.randrange(span)
        out.append("\t".join([rng.choice(chroms), str(s), str(s + rng.randrange(1, 600)),
                              rng.choice(["Stringent", "Marginal", "Stringent,Marginal"]),
                              "%.3g" % rng.random(), "%.3g" % rng.random(), str(rng.randrange(1, 90)),
                              str(rng.randrange(1, 90))]) + "\n")
    return out


def _single_records(rng, n, chroms, span):
    """Records shaped like *_unidirectional_peaks.bed (9 columns)."""
    out = []
    for i in range(n):
        s = rng.randrange(span)
        out.append("\t".join([rng.choice(chroms), str(s), str(s + rng.randrange(1, 400)), "Peak%d" % i,
                              "%.3g" % rng.random(), rng.choice("+-"), str(rng.randrange(1, 90)),
                              str(s + 1), str(rng.randrange(1, 30))]) + "\n")
    return out


def _write(path, lines):
    with open(path, "w") as fh:
        fh.writelines(lines)


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


@pytest.mark.parametrize("seed", range(8))
def test_inferring_elements_from_other_reps(seed, tmp_path):
    rng = random.Random(seed)
    chroms = rng.sample(["chr1", "chr2", "chr10", "chrX"], rng.choice([1, 2, 4]))
    span = rng.choice([2_000, 20_000, 200_000])
    n_samples = rng.choice([2, 3])
    files = {}
    for rep in range(1, n_samples + 1):
        for label, maker in (("bidirectional", _pair_records), ("divergent", _pair_records),
                             ("unidirectional", _single_records)):
            n = rng.choice([0, 1, 30, 400]) if rng.random() < 0.2 else rng.choice([30, 400])
            files["x_%d_%s_peaks.bed" % (rep, label)] = maker(rng, n, chroms, span)
    out = {}
    for name, mod in (("new", ce), ("ref", rce)):
        d = tmp_path / name
        d.mkdir()
        for fn, lines in files.items():
            _write(d / fn, lines)
        mod.inferring_elements_from_other_reps(str(d / "x"), n_samples)
        out[name] = {fn: _read(d / fn) for fn in sorted(os.listdir(d))}
    assert out["new"] == out["ref"]


@pytest.mark.parametrize("compressed", [False, True])
def test_extension_with_promoter_bed(compressed, tmp_path):
    rng = random.Random(7)
    span = 50_000_000
    promoters = ["track name=promoters\n", "# housekeeping\n"] + [
        "chr22\t%d\t%d\tp%d\t0\t%s\n" % (s, s + rng.randrange(1, 3000), i, rng.choice("+-"))
        for i, s in enumerate(rng.randrange(span) for _ in range(3000))]
    peaks = {"a_divergent_peaks.bed": _pair_records(rng, 300, ["chr22"], span),
             "a_bidirectional_peaks.bed": _pair_records(rng, 300, ["chr22"], span),
             "a_unidirectional_peaks.bed": _single_records(rng, 300, ["chr22"], span)}
    out = {}
    for name, mod in (("new", ee), ("ref", ree)):
        d = tmp_path / name
        d.mkdir()
        for fn, lines in peaks.items():
            _write(d / fn, lines)
        prom = str(d / ("promoters.bed.gz" if compressed else "promoters.bed"))
        if compressed:
            with gzip.open(prom, "wt") as fh:
                fh.writelines(promoters)
        else:
            _write(prom, promoters)
        mod.main((os.path.join(DATASETS, "pl.chr22.bw"), os.path.join(DATASETS, "mn.chr22.bw")), None,
                 str(d / "a_divergent_peaks.bed"), str(d / "a_bidirectional_peaks.bed"),
                 str(d / "a_unidirectional_peaks.bed"), promoter_bed=prom)
        out[name] = {fn: _read(d / fn) for fn in sorted(os.listdir(d)) if "_element_" in fn}
    assert len(out["new"]) == 6
    distal = [fn for fn in out["new"] if fn.endswith("bp_e.bed")]
    assert all(0 < out["new"][fn].count(b"\n") < out["new"][fn.replace("bp_e.bed", "bp.bed")].count(b"\n")
               for fn in distal)  # the promoters remove some elements, not all
    assert out["new"] == out["ref"]


def _load_counter():
    loader = importlib.machinery.SourceFileLoader("pints_counter_script",
                                                  os.path.join(ROOT, "scripts", "pints_counter"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("max_dist", [0, 1, 250])
def test_counter_prepare_regions(max_dist, tmp_path):
    pybedtools = pytest.importorskip("pybedtools")
    rng = random.Random(max_dist)
    paths = []
    for k, maker in enumerate((_pair_records, _pair_records, _single_records)):
        p = str(tmp_path / ("%d.bed" % k))
        _write(p, maker(rng, 500, ["chr1", "chr2"], 100_000))
        paths.append(p)
    got = _load_counter().prepare_regions(paths[:2], paths[2:], max_dist=max_dist)
    expected = pybedtools.BedTool.cat(*[pybedtools.BedTool(f) for f in paths], postmerge=False).sort().merge(
        d=max_dist).saveas()
    assert [[r[0], int(r[1]), int(r[2])] for r in got] == [[r[0], r.start, r.end] for r in expected]
