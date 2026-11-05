"""`_run_tasks` dispatches the costliest tasks first but returns results in task order."""
from pints.calling_engine import _run_tasks


class _RecordingPool:
    def __init__(self):
        self.dispatched = []

    def starmap(self, func, iterable, chunksize=1):
        out = []
        for args in iterable:
            self.dispatched.append(args[0])
            out.append(func(*args))
        return out


def _square(x):
    return x * x


def test_costliest_first_results_in_order():
    pool = _RecordingPool()
    tasks = [(x,) for x in range(5)]
    assert _run_tasks(pool, _square, tasks, costs=[3, 50, 7, 50, 1]) == [0, 1, 4, 9, 16]
    assert pool.dispatched == [1, 3, 2, 0, 4]  # ties keep task order


def test_inline_without_pool():
    assert _run_tasks(None, _square, [(2,), (3,)], costs=[1, 2]) == [4, 9]
