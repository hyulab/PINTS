# coding=utf-8
"""
In-memory sparse coverage: one `CoverageTrack` per chromosome strand instead of a dense per-base .npy dump.

More than 99.9% of the bases of a dense coverage dump are zero (bench/design/sparse-coverage.md, section 2), so
a track keeps only the non-zero bases, as sorted runs. Readers go through the methods below (`window`, `fill`,
`clear`, `nonzero`, `sum`, `max`, `is_zero`), never through the arrays, so the layout can change behind them.
Pool workers reach tracks through the module-level registry, by key: tasks carry keys, not tracks, and the
workers inherit the registry when the pool forks (tracks are read-only after that, so their pages stay shared).
`track_from_npy` / `track_to_npy` are the only bridge to files.
"""
import mmap

import numpy as np

__all__ = ("CoverageTrack", "track_from_dense", "track_from_npy", "track_to_npy", "register", "lookup", "release",
           "is_key")


def _pos_dtype(length):
    return np.int32 if length <= np.iinfo(np.int32).max else np.int64


class CoverageTrack:
    """
    Coverage of one chromosome strand of `length` bases, as the dense array of dtype `dtype` it stands for.

    The non-zero bases are stored as sorted, non-overlapping runs ``[starts[i], ends[i])`` holding ``values[i]``
    (non-zero, of dtype `dtype`). ``ends`` is None in the points form, where every run has length 1 (BAM 5'
    counts). `window`, `dense`, `sum` and `max` return exactly what the same operation on the dense array
    returns. Integer dtypes only, like every coverage dtype in PINTS (`NP_DT_NAME` and the uint32 BAM counters).

    `zero_track` marks what the file-based code registers with `save_zero_track` (`is_zero_track`): an empty
    track can be either, and some code paths branch on the registration, not on the content.
    """
    __slots__ = ("length", "dtype", "starts", "ends", "values", "zero_track")

    def __init__(self, length, dtype, starts, values, ends=None, zero_track=False):
        self.length = int(length)
        self.zero_track = bool(zero_track)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in "biu":
            raise TypeError("coverage tracks hold integer dtypes, not %s" % self.dtype)
        pd = _pos_dtype(self.length)
        self.starts = np.ascontiguousarray(starts, dtype=pd)
        self.ends = None if ends is None else np.ascontiguousarray(ends, dtype=pd)
        self.values = np.ascontiguousarray(values, dtype=self.dtype)
        if self.starts.shape != self.values.shape or (self.ends is not None and self.ends.shape != self.starts.shape):
            raise ValueError("starts, ends and values must have the same length")

    # ------------------------------------------------------------------ properties
    @property
    def is_zero(self):
        return self.values.shape[0] == 0

    @property
    def nbytes(self):
        return self.starts.nbytes + self.values.nbytes + (0 if self.ends is None else self.ends.nbytes)

    def nonzero(self):
        """(starts, ends, values) of the non-zero runs; ends are starts + 1 in the points form."""
        return self.starts, (self.starts + 1 if self.ends is None else self.ends), self.values

    def _run_lengths(self):
        return None if self.ends is None else (self.ends - self.starts)

    def sum(self):
        """``dense().sum()``: an int64 (uint64 for unsigned dtypes), as numpy accumulates integers."""
        acc = np.uint64 if self.dtype.kind == "u" else np.int64
        lens = self._run_lengths()
        if lens is None:
            return self.values.sum(dtype=acc)
        return (self.values.astype(acc) * lens.astype(acc)).sum(dtype=acc)

    def max(self):
        """``dense().max()``, of dtype `dtype` (ValueError for a zero-length track, as numpy raises)."""
        if self.length == 0:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        lens = self._run_lengths()
        covered = self.values.shape[0] if lens is None else int(lens.sum())
        if self.values.shape[0] == 0:
            return self.dtype.type(0)
        m = self.values.max()
        if covered < self.length and m < 0:
            m = self.dtype.type(0)
        return self.dtype.type(m)

    # ------------------------------------------------------------------ dense views
    def _span(self, a, b):
        """Indices i0:i1 of the runs that intersect [a, b)."""
        if self.ends is None:
            return np.searchsorted(self.starts, a, "left"), np.searchsorted(self.starts, b, "left")
        return np.searchsorted(self.ends, a, "right"), np.searchsorted(self.starts, b, "left")

    def _positions(self, a, b):
        """Positions and values of the non-zero bases in [a, b), ascending."""
        i0, i1 = self._span(a, b)
        if self.ends is None:
            return self.starts[i0:i1], self.values[i0:i1]
        s = np.maximum(self.starts[i0:i1], a).astype(np.int64)
        e = np.minimum(self.ends[i0:i1], b).astype(np.int64)
        lens = e - s
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=self.dtype)
        offsets = np.repeat(s - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        return offsets + np.arange(total, dtype=np.int64), np.repeat(self.values[i0:i1], lens)

    def window(self, a, b):
        """``dense()[a:b]`` (Python slice rules, including negative and out-of-range bounds), as a new array."""
        a, b, _ = slice(a, b).indices(self.length)
        out = np.zeros(max(b - a, 0), dtype=self.dtype)
        if b > a:
            pos, vals = self._positions(a, b)
            out[pos - a] = vals
        return out

    def dense(self):
        return self.window(0, self.length)

    def fill(self, buf, a=0, b=None):
        """Write the non-zero bases in [a, b) into `buf`, a full-length array of this track's dense form that is
        zero there. Pages of a fresh ``np.zeros`` buffer that are never written cost no memory."""
        pos, vals = self._positions(max(a, 0), self.length if b is None else min(b, self.length))
        buf[pos] = vals

    def clear(self, buf, a=0, b=None):
        """Undo `fill(buf, a, b)`: zero the bases it wrote."""
        pos, _ = self._positions(max(a, 0), self.length if b is None else min(b, self.length))
        buf[pos] = 0

    def empty_buffer(self):
        """
        A full-length zero array of the dense dtype for `fill`, in its own anonymous mapping: its pages are only
        allocated when written, and all of them go back to the OS when it is freed. (After freeing a few
        np.zeros buffers of some MB, malloc serves the next ones from its heap, where calloc writes every page and
        freed pages stay with the process: a pool worker kept ~150 MB that way.)
        """
        n = self.length
        return np.frombuffer(mmap.mmap(-1, max(n * self.dtype.itemsize, 1)), dtype=self.dtype, count=n)

    def __repr__(self):
        return "CoverageTrack(length=%d, dtype=%s, runs=%d, %s)" % (
            self.length, self.dtype, self.values.shape[0], "points" if self.ends is None else "runs")

    def __reduce__(self):
        # tracks are reached through the registry; pickling one into a pool task would copy it into every task
        raise TypeError("CoverageTrack is not meant to be pickled; pass its registry key instead")


def zero_track(length, dtype=np.int8):
    """The track `save_zero_track(path, length, dtype)` stands for: all zeros, registered as a zero track."""
    return CoverageTrack(length, dtype, np.empty(0, dtype=np.int64), np.empty(0, dtype=dtype), zero_track=True)


def align(tracks, support=None):
    """
    The tracks cut into common pieces: (starts, ends, [values of each track on each piece]), where the pieces
    are the elementary segments between all run boundaries, in order, and every track is constant on each.
    Bases outside every run are left out. ends is None when every track is in the points form (pieces of
    one base). With `support` (an index into `tracks`), only the pieces where that track is non-zero.
    """
    pieces = [t.nonzero() for t in tracks]
    points = all(t.ends is None for t in tracks)
    if points:
        bounds = np.unique(np.concatenate([p[0] for p in pieces]).astype(np.int64)) if support is None \
            else pieces[support][0].astype(np.int64)
        seg_end = None
    else:
        src = pieces if support is None else [pieces[support]] + [p for k, p in enumerate(pieces) if k != support]
        edges = np.unique(np.concatenate([np.concatenate((p[0], p[1])) for p in src]).astype(np.int64))
        bounds, seg_end = edges[:-1], edges[1:]
    vals = []
    for t, (st, en, va) in zip(tracks, pieces):
        v = np.zeros(bounds.shape[0], dtype=t.dtype)
        if st.shape[0]:
            i = np.searchsorted(st, bounds, "right") - 1  # the run starting at or before each piece
            inside = (i >= 0) & (bounds < en[np.maximum(i, 0)])
            v[inside] = va[i[inside]]
        vals.append(v)
    keep = None
    if support is not None:
        keep = vals[support] != 0
    elif seg_end is not None:
        keep = np.zeros(bounds.shape[0], dtype=bool)
        for v in vals:
            keep |= v != 0
    if keep is not None:
        bounds, vals = bounds[keep], [v[keep] for v in vals]
        seg_end = None if seg_end is None else seg_end[keep]
    return bounds, seg_end, vals


def normalize_track(exp, inp, sf):
    """
    One chromosome of `normalize_using_input` on tracks: ``max(exp - inp * sf, 0)`` in the smallest
    `NP_DT_NAME` dtype that holds its maximum, or None when no dtype does (the chromosome is then dropped,
    as by the file code).

    Computed only where `exp` is non-zero. Elsewhere the dense code computes ``0 - inp * sf``: with `sf`
    finite and >= 0 and `inp` non-negative (coverage after loading: counts, or bigWig values after
    nan_to_num and the sign flip), that is <= 0 and never NaN, so it is set to 0 and stays 0 in every integer
    dtype (a signed zero cannot survive the cast). Those zeros cannot change the maximum's dtype choice either,
    as the maximum is >= 0 whenever any base is off the support. On the support, every element gets the
    same float64 operations as in the dense code. Other `sf`, or tracks of different lengths, use the dense
    code on the dense arrays.
    """
    from .io_engine import NP_DT_RANGE, NP_DT_NAME
    if not (np.isfinite(sf) and sf >= 0) or exp.length != inp.length or exp.length == 0 \
            or (inp.values.shape[0] and inp.values.min() < 0):
        matched_input = inp.dense() * sf
        adjusted_signal = exp.dense() - matched_input
        adjusted_signal[adjusted_signal < 0] = 0
        chrom_max = np.max(adjusted_signal)
        for k, v in enumerate(NP_DT_RANGE):
            if v > chrom_max:
                return track_from_dense(adjusted_signal.astype(NP_DT_NAME[k], copy=False))
        return None
    starts, ends, (e, i) = align([exp, inp], support=0)
    matched_input = i * sf
    adjusted_signal = e - matched_input
    adjusted_signal[adjusted_signal < 0] = 0
    covered = starts.shape[0] if ends is None else int((ends - starts).sum())
    chrom_max = np.max(adjusted_signal) if covered else None
    if covered < exp.length:
        chrom_max = 0. if chrom_max is None else max(chrom_max, 0.)
    for k, v in enumerate(NP_DT_RANGE):
        if v > chrom_max:
            out = adjusted_signal.astype(NP_DT_NAME[k], copy=False)
            nz = out != 0
            return CoverageTrack(exp.length, NP_DT_NAME[k], starts[nz], out[nz], None if ends is None else ends[nz])
    return None


def merge_tracks(tracks):
    """
    Replicate merge of `merge_replicates_bw` on tracks: (merged track, its sum), where the merged track is the
    sum of the inputs in int32 (the first replicate cast to int32, the others added in place, which wraps
    like int32), cast to the dtype `determine_data_type` picks from its maximum, and the sum is ``np.sum`` of
    the int32 sum. Inputs of different lengths raise ValueError (the dense code fails to broadcast).
    """
    from .io_engine import determine_data_type
    n = tracks[0].length
    if any(t.length != n for t in tracks):
        raise ValueError("operands could not be broadcast together with shapes %s" % [(t.length,) for t in tracks])
    bounds, seg_end, vals = align(tracks)
    points = seg_end is None
    total = np.zeros(bounds.shape[0], dtype=np.int64)
    for v in vals:
        total += v
    merged = total.astype(np.int32)  # modular, as repeated in-place int32 additions
    keep = merged != 0
    m32 = CoverageTrack(n, np.int32, bounds[keep], merged[keep], None if points else seg_end[keep])
    s = m32.sum()
    data_type = determine_data_type(m32.max())  # raises for a zero-length chromosome, like np.max
    vals = m32.values.astype(data_type, copy=False)
    nz = vals != 0
    out = CoverageTrack(n, data_type, m32.starts[nz], vals[nz], None if points else m32.ends[nz])
    return out, s


def track_from_dense(arr):
    """Points-form track holding the dense 1-D integer array `arr` (same length and dtype)."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("coverage arrays are 1-D")
    nz = np.flatnonzero(arr)
    return CoverageTrack(arr.shape[0], arr.dtype, nz, arr[nz])


def track_from_npy(path):
    """Track for a 1-D integer .npy coverage dump (read in blocks, never as a whole dense array)."""
    from .io_engine import npy_layout, npy_read, NPY_BLOCK
    lay = npy_layout(path)
    if lay is None:
        return track_from_dense(np.load(path))
    _, dtype, n = lay
    starts, values = [], []
    with open(path, "rb") as fh:
        for a in range(0, n, NPY_BLOCK):
            blk = npy_read(fh, lay, a, min(n, a + NPY_BLOCK))
            nz = np.flatnonzero(blk)
            starts.append(nz + a)
            values.append(blk[nz])
    if not starts:
        return CoverageTrack(n, dtype, np.empty(0, dtype=np.int64), np.empty(0, dtype=dtype))
    return CoverageTrack(n, dtype, np.concatenate(starts), np.concatenate(values))


def track_to_npy(track, path):
    """Write the dense form of `track` to `path` exactly as ``np.save(path, track.dense())`` would."""
    from .io_engine import _npy_create, NPY_BLOCK
    offset = _npy_create(path, track.dtype, track.length)[0]  # np.save's header; data region zero-filled
    if track.is_zero:
        return path
    with open(path, "r+b") as fh:
        for a in range(0, track.length, NPY_BLOCK):
            b = min(track.length, a + NPY_BLOCK)
            i0, i1 = track._span(a, b)
            if i1 <= i0:
                continue
            blk = track.window(a, b)
            fh.seek(offset + a * track.dtype.itemsize)
            fh.write(blk.tobytes())
    return path


# ---------------------------------------------------------------------------------------------------
# Registry: the only way pool workers reach tracks. Workers inherit it when the pool forks.
# ---------------------------------------------------------------------------------------------------
_REGISTRY = {}


class TrackKey(tuple):
    """Registry key of a track: (sample, strand, chromosome). A tuple subclass, so it pickles small."""
    __slots__ = ()

    def __new__(cls, sample, strand, chromosome):
        return tuple.__new__(cls, (sample, strand, chromosome))

    def __reduce__(self):
        return TrackKey, tuple(self)


def is_key(obj):
    return isinstance(obj, TrackKey)


def register(sample, strand, chromosome, track):
    key = TrackKey(sample, strand, chromosome)
    _REGISTRY[key] = track
    return key


def lookup(key):
    return _REGISTRY[key]


def release(keys):
    for k in keys:
        _REGISTRY.pop(k, None)
