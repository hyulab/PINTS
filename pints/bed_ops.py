#!/usr/bin/env python
# coding=utf-8
"""
The few bedtools operations PINTS runs on its own BED files, without bedtools or pybedtools.

Each function reproduces, byte for byte, what PINTS got from pybedtools + bedtools 2.31 on Linux:

* `sort_lines` is ``bedtools sort``: chromosomes in byte order (``std::map<string>``), records of a
  chromosome ordered by start with libstdc++'s ``std::sort``, which is not stable. Records with equal
  starts therefore come out in an order that depends on the whole chromosome's input order; `_std_sort`
  replays the algorithm to get it. Chromosomes without tied starts are ordered with numpy.
* `merge_lines` is ``bedtools merge -c ... -o collapse|distinct`` on sorted records.
* `cat_lines` is ``BedTool.cat(..., postmerge=False)``: concatenation, truncated to the smallest field
  count when the files differ in it.

Only what PINTS writes is supported (tab-separated BED records, no header, track or comment lines);
other input raises ValueError instead of being silently treated differently.
"""
import numpy as np

_S_THRESHOLD = 16  # libstdc++ _S_threshold


def _std_sort(idx, key):
    """
    Reorder the list `idx` in place exactly as libstdc++ ``std::sort(first, last, comp)`` would with
    ``comp(a, b) = key[a] < key[b]`` (introsort, median-of-three pivot, heapsort at the depth limit,
    final insertion sort; bits/stl_algo.h and bits/stl_heap.h).
    """
    n = len(idx)
    if n < 2:
        return idx

    def move_median_to_first(result, a, b, c):
        ka, kb, kc = key[idx[a]], key[idx[b]], key[idx[c]]
        if ka < kb:
            if kb < kc:
                m = b
            elif ka < kc:
                m = c
            else:
                m = a
        elif ka < kc:
            m = a
        elif kb < kc:
            m = c
        else:
            m = b
        idx[result], idx[m] = idx[m], idx[result]

    def unguarded_partition(first, last, pivot):
        kp = key[idx[pivot]]
        while True:
            while key[idx[first]] < kp:
                first += 1
            last -= 1
            while kp < key[idx[last]]:
                last -= 1
            if not first < last:
                return first
            idx[first], idx[last] = idx[last], idx[first]
            first += 1

    def adjust_heap(first, hole, length, value):
        top = hole
        child = hole
        while child < (length - 1) // 2:
            child = 2 * (child + 1)
            if key[idx[first + child]] < key[idx[first + child - 1]]:
                child -= 1
            idx[first + hole] = idx[first + child]
            hole = child
        if length & 1 == 0 and child == (length - 2) // 2:
            child = 2 * (child + 1)
            idx[first + hole] = idx[first + child - 1]
            hole = child - 1
        kv = key[value]
        parent = (hole - 1) // 2
        while hole > top and key[idx[first + parent]] < kv:
            idx[first + hole] = idx[first + parent]
            hole = parent
            parent = (hole - 1) // 2
        idx[first + hole] = value

    def heap_sort(first, last):
        length = last - first
        if length >= 2:
            parent = (length - 2) // 2
            while True:
                adjust_heap(first, parent, length, idx[first + parent])
                if parent == 0:
                    break
                parent -= 1
        while last - first > 1:
            last -= 1
            value = idx[last]
            idx[last] = idx[first]
            adjust_heap(first, 0, last - first, value)

    def introsort_loop(first, last, depth):
        while last - first > _S_THRESHOLD:
            if depth == 0:
                heap_sort(first, last)
                return
            depth -= 1
            mid = first + (last - first) // 2
            move_median_to_first(first, first + 1, mid, last - 1)
            cut = unguarded_partition(first + 1, last, first)
            introsort_loop(cut, last, depth)
            last = cut

    def unguarded_linear_insert(pos):
        value = idx[pos]
        kv = key[value]
        nxt = pos - 1
        while kv < key[idx[nxt]]:
            idx[pos] = idx[nxt]
            pos = nxt
            nxt -= 1
        idx[pos] = value

    def insertion_sort(first, last):
        for i in range(first + 1, last):
            value = idx[i]
            if key[value] < key[idx[first]]:
                idx[first + 1:i + 1] = idx[first:i]
                idx[first] = value
            else:
                unguarded_linear_insert(i)

    introsort_loop(0, n, 2 * (n.bit_length() - 1))
    if n > _S_THRESHOLD:
        insertion_sort(0, _S_THRESHOLD)
        for i in range(_S_THRESHOLD, n):
            unguarded_linear_insert(i)
    else:
        insertion_sort(0, n)
    return idx


def read_lines(path):
    """Records of a BED file PINTS wrote, as lines with their newline (blank lines dropped, as bedtools does)."""
    with open(path, newline="") as fh:
        lines = [ln if ln.endswith("\n") else ln + "\n" for ln in fh if ln.strip()]
    for ln in lines[:1]:
        if ln.startswith(("#", "track", "browser")):
            raise ValueError("%s: header lines are not supported" % path)
    return lines


def _fields(lines):
    rows = [ln.rstrip("\n").split("\t") for ln in lines]
    for r in rows:
        if len(r) < 3:
            raise ValueError("not a BED record: %r" % "\t".join(r))
    return rows


def _start_column(row):
    """Column (1-based) bedtools sorts on, from the format it detects in a record: BED when columns 2 and 3 are
    integers, else GFF/GTF when columns 4 and 5 are."""
    if len(row) >= 3 and row[1].lstrip("-").isdigit() and row[2].lstrip("-").isdigit():
        return 2
    if len(row) >= 5 and row[3].lstrip("-").isdigit() and row[4].lstrip("-").isdigit():
        return 4
    raise ValueError("neither a BED nor a GFF record: %r" % "\t".join(row))


def sort_lines(lines, start_column=None):
    """`lines` (BED or GFF/GTF records, told apart by their first record unless `start_column` is given) in
    ``bedtools sort`` order."""
    rows = _fields(lines)
    if start_column is None and rows:
        start_column = _start_column(rows[0])
    by_chrom = {}
    for i, r in enumerate(rows):
        by_chrom.setdefault(r[0], []).append(i)
    out = []
    for chrom in sorted(by_chrom, key=lambda c: c.encode()):
        members = by_chrom[chrom]
        # a zero-length record is loaded as [start - 1, end + 1) and sorted on start + 1: its own start
        starts = np.fromiter((int(rows[i][start_column - 1]) for i in members), dtype=np.int64, count=len(members))
        if np.unique(starts).shape[0] == starts.shape[0]:
            order = np.argsort(starts, kind="stable").tolist()
        else:
            order = _std_sort(list(range(len(members))), starts.tolist())
        out.extend(lines[members[j]] for j in order)
    return out


def merge_lines(lines, columns, operations):
    """
    ``bedtools merge -c columns -o operations`` on sorted BED records: overlapping and book-ended records
    are merged, zero-length ones taken as one base wider on each side; ``collapse`` lists a column's values in record order, ``distinct`` its unique values in
    byte order. Columns are 1-based.
    """
    if len(columns) != len(operations) or not set(operations) <= {"collapse", "distinct"}:
        raise ValueError("unsupported merge operations: %r" % (operations,))
    out = []
    cur = None
    for r in _fields(lines):
        start, end = int(r[1]), int(r[2])
        if start == end:  # bedtools widens zero-length records by one base on each side
            start, end = start - 1, end + 1
        if cur is not None and r[0] == cur[0] and start <= cur[2]:
            cur[2] = max(cur[2], end)
            cur[3].append(r)
            continue
        if cur is not None:
            out.append(_merged_line(cur, columns, operations))
        cur = [r[0], start, end, [r]]
    if cur is not None:
        out.append(_merged_line(cur, columns, operations))
    return out


def _merged_line(cluster, columns, operations):
    chrom, start, end, members = cluster
    if len(members) == 1:  # a record merged with nothing keeps its own coordinates, even if zero-length
        start, end = int(members[0][1]), int(members[0][2])
    fields = [chrom, str(start), str(end)]
    for col, op in zip(columns, operations):
        values = [m[col - 1] for m in members]
        if op == "distinct":
            values = sorted(set(values), key=lambda v: v.encode())
        fields.append(",".join(values))
    return "\t".join(fields) + "\n"


def cat_lines(paths):
    """``BedTool.cat(*paths, postmerge=False)`` for PINTS's own BED files: records of all non-empty files
    in order, cut to the smallest field count among them when the files differ."""
    parts = [read_lines(p) for p in paths]
    parts = [p for p in parts if p]
    counts = set()
    for p in parts:
        n = {ln.count("\t") + 1 for ln in p[:11]}  # pybedtools looks at the first features only
        if len(n) > 1:
            raise ValueError("records with different field counts in one file")
        counts |= n
    lines = [ln for p in parts for ln in p]
    if len(counts) > 1:
        k = min(counts)
        lines = ["\t".join(ln.rstrip("\n").split("\t")[:k]) + "\n" for ln in lines]
    return lines


def is_empty(path):
    """True if the file has no records (`BedTool(path).count() == 0`: blank, header, track and browser lines
    are not records)."""
    with open(path) as fh:
        return not any(ln.strip() and not ln.startswith(("@", "#", "track", "browser")) for ln in fh)


def write_lines(lines, path):
    with open(path, "w", newline="") as fh:
        fh.writelines(lines)
