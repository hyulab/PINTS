"""Synthetic-data builders used across test modules (no randomness unless an rng is passed)."""
import gzip
import os

import numpy as np
import pysam


# ---------------------------------------------------------------- coverage ----
def synth_coverage(rng, length=60_000, n_peaks=40, bg_rate=0.02, bg_pi=0.9, peak_width=(5, 60),
                   peak_height=(3, 60), dtype=np.int32, edge_peaks=True):
    """Sparse ZIP-like background with planted TSS-like clusters (incl. near chromosome edges)."""
    bg = rng.poisson(bg_rate / (1 - bg_pi) if bg_pi < 1 else 0, size=length)
    bg[rng.random(length) < bg_pi] = 0
    cov = bg.astype(np.int64)
    centers = list(rng.integers(200, length - 200, size=n_peaks))
    if edge_peaks:
        centers += [30, 1200, length - 40, length - 1500]
    for c in centers:
        w = int(rng.integers(*peak_width))
        h = float(rng.uniform(*peak_height))
        s = max(0, c - w // 2)
        e = min(length, s + w)
        x = np.arange(e - s)
        prof = h * np.exp(-0.5 * ((x - (e - s) / 2) / max(1.0, w / 6)) ** 2)
        cov[s:e] += rng.poisson(prof)
    return cov.astype(dtype)


# ---------------------------------------------------------------- tabix -------
def write_subpeak_tabix(path, chrom, rows):
    """rows: iterables of (start, end, name, mu, var, strand, pi, nonzero, summit, summit_val).

    Mirrors the format written by check_window_chromosome (incl. the '%f' formatting).
    """
    path = str(path)
    with open(path, "w") as fh:
        for r in sorted(rows, key=lambda x: (x[0], x[1])):
            s, e, name, mu, var, strand, pi, nz, sm, sv = r
            fh.write("%s\t%d\t%d\t%s\t%f\t%s\t%s\t%f\t%d\t%d\t%d\n" % (chrom, s, e, name, mu, var, strand, pi, nz,
                                                                      sm, sv))
    pysam.tabix_index(path, preset="bed", force=True, csi=True)
    return path + ".gz"


def write_candidate_gz(path, chrom, intervals):
    """Gzipped subpeak definition file as produced by unified_element_definition."""
    with gzip.open(str(path), "wt") as fh:
        for i, (s, e) in enumerate(intervals):
            fh.write("%s\t%d\t%d\t%s-%d\t%d\n" % (chrom, s, e, chrom, i + 1, s))
    return str(path)


# ---------------------------------------------------------------- BAM ---------
def make_bam(path, contigs, reads):
    """Write a coordinate-sorted, indexed BAM.

    contigs : list of (name, length)
    reads   : list of dicts with keys: contig, pos, cigar (str), reverse, mapq, and optional
              read1/read2/paired/secondary/supplementary/unmapped/name
    """
    path = str(path)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": n, "LN": l} for n, l in contigs]}
    tmp = path + ".unsorted.bam"
    names = [n for n, _ in contigs]
    with pysam.AlignmentFile(tmp, "wb", header=header) as out:
        for i, r in enumerate(reads):
            a = pysam.AlignedSegment(out.header)
            a.query_name = r.get("name", "r%d" % i)
            if r.get("contig") is not None:
                a.cigarstring = r["cigar"]
            qlen = (a.infer_query_length() or 30) if r.get("contig") is not None else 30
            a.query_sequence = "A" * qlen
            a.query_qualities = pysam.qualitystring_to_array("I" * qlen)
            flag = 0
            if r.get("paired"):
                flag |= 0x1
            if r.get("read1"):
                flag |= 0x40
            if r.get("read2"):
                flag |= 0x80
            if r.get("reverse"):
                flag |= 0x10
            if r.get("secondary"):
                flag |= 0x100
            if r.get("supplementary"):
                flag |= 0x800
            if r.get("unmapped"):
                flag |= 0x4
            a.flag = flag
            if r.get("contig") is None:
                a.reference_id = -1
                a.reference_start = -1
                a.next_reference_id = -1
                a.next_reference_start = -1
            else:
                a.reference_id = names.index(r["contig"])
                a.reference_start = r["pos"]
            a.mapping_quality = r.get("mapq", 60)
            out.write(a)
    pysam.sort("-o", path, tmp)
    os.remove(tmp)
    pysam.index(path)
    return path


def expected_positions(reads, end, rc, which_read, mapq=30):
    """Independent re-implementation of the PINTS BAM-parsing semantics for test oracles.

    end: '5' or '3'; rc: reverse complement; which_read: None (SE), 'R1' or 'R2'.
    Returns dict (contig, strand) -> list of 0-based positions, and total read count.
    """
    out = {}
    total = 0
    for r in reads:
        if r.get("contig") is None:
            continue
        if r.get("mapq", 60) < mapq:
            continue
        if which_read == "R1" and r.get("read2"):
            continue
        if which_read == "R2" and r.get("read1"):
            continue
        a = pysam.AlignedSegment()
        a.cigarstring = r["cigar"]
        ref_len = a.reference_length
        start = r["pos"]
        stop = start + ref_len  # exclusive
        rev = bool(r.get("reverse"))
        if end == "5":
            pos = stop - 1 if rev else start
        else:
            pos = start if rev else stop - 1
        strand = "-" if rev else "+"
        if rc:
            strand = "+" if strand == "-" else "-"
        out.setdefault((r["contig"], strand), []).append(pos)
        total += 1
    return out, total


def make_bigwig(path, chroms, values):
    """chroms: list of (name, length); values: dict name -> float array (np.nan for gaps)."""
    import pyBigWig
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(list(chroms))
    for name, _ in chroms:
        v = np.asarray(values[name], dtype=np.float64)
        idx = np.where(~np.isnan(v) & (v != 0))[0]
        if idx.size:
            bw.addEntries(name, idx.astype(np.int64).tolist(), values=v[idx].tolist(), span=1)
    bw.close()
    return str(path)
