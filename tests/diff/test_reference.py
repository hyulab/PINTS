"""The frozen reference loads as `pints_ref`, separately from the working tree, and agrees with it on the
functions later changes keep bit-identical (here, before any of them)."""
import logging
import os

import numpy as np
import pandas as pd
import pytest

import pints.calling_engine as ce
import pints.stats_engine as se
import pints_ref.calling_engine as rce
import pints_ref.stats_engine as rse

HERE = os.path.dirname(os.path.abspath(__file__))


def test_reference_is_the_frozen_copy():
    ref_dir = os.path.join(os.path.dirname(HERE), "reference", "pints_ref")
    for mod in (rce, rse):
        assert os.path.dirname(os.path.abspath(mod.__file__)) == ref_dir
    for mod in (ce, se):
        assert os.path.dirname(os.path.abspath(mod.__file__)) != ref_dir


@pytest.mark.parametrize("seed", range(5))
def test_independent_filtering_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 2000
    df = pd.DataFrame({"chromosome": "chr1", "start": np.arange(n), "end": np.arange(n) + 5, "name": ".",
                       "pval": rng.beta(0.3, 1, n), "strand": "+", "reads": rng.negative_binomial(1, 0.1, n)})
    log = logging.getLogger("test")
    got = se.independent_filtering(df.copy(), fdr_target=0.1, logger=log)
    ref = rse.independent_filtering(df.copy(), fdr_target=0.1, logger=log)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


@pytest.mark.parametrize("seed", range(5))
def test_zip_fit_matches_reference(seed):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(500) < 0.7, 0, rng.poisson(3, 500))
    assert se.ZIP().fit(x) == rse.ZIP().fit(x)
