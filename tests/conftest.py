"""Shared fixtures and helpers for the PINTS optimisation test-suite.

Layout
------
tests/unit         characterisation tests of the working tree (`pints`)
tests/diff         differential tests: `pints` (optimised) vs frozen `pints_ref`, bitwise
tests/reference    frozen copy of upstream PINTS 81b2dd9 as `pints_ref` (imports renamed only), with the
                   one-strand pair merge fix of the bugfix PR

Run with:  pip install pytest hypothesis numba && pytest -m "not slow" tests/unit tests/diff
"""
import os
import sys
import logging

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "reference"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DATA = os.path.join(HERE, "datasets")


@pytest.fixture(scope="session")
def logger():
    return logging.getLogger("pints-tests")


@pytest.fixture
def rng():
    return np.random.default_rng(20261005)


@pytest.fixture(autouse=True)
def _pybedtools_tmp(tmp_path):
    import pybedtools
    pybedtools.set_tempdir(str(tmp_path))
    yield
    pybedtools.cleanup()


def pytest_configure(config):
    config.addinivalue_line("markers", "spec: characterisation of behaviour where the paper is ambiguous")
    config.addinivalue_line("markers", "slow: end-to-end / simulation tests (seconds to minutes)")
