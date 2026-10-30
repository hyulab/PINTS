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
    """
    __slots__ = ("length", "dtype", "starts", "ends", "values")

    def __init__(self, length, dtype, starts, values, ends=None):
        self.length = int(length)
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
        """A full-length zero array of the dense dtype for `fill` (its pages are only allocated when written)."""
        return np.zeros(self.length, dtype=self.dtype)

    def __repr__(self):
        return "CoverageTrack(length=%d, dtype=%s, runs=%d, %s)" % (
            self.length, self.dtype, self.values.shape[0], "points" if self.ends is None else "runs")

    def __reduce__(self):
        # tracks are reached through the registry; pickling one into a pool task would copy it into every task
        raise TypeError("CoverageTrack is not meant to be pickled; pass its registry key instead")


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
