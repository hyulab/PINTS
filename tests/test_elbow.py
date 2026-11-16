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
import unittest
import warnings
import numpy as np
from pints.stats_engine import get_elbow


def _elbow_with_np_cross(X, Y):
    """get_elbow as written with np.cross on 2-D vectors (NumPy < 2.5)"""
    sort_index = X.argsort()
    X_s, Y_s = X[sort_index], Y[sort_index]
    p1, p2 = np.array([X_s[0], Y_s[0]]), np.array([X_s[-1], Y_s[-1]])
    D = np.abs(np.cross(p2 - p1, np.column_stack((X_s, Y_s)) - p1) / np.linalg.norm(p2 - p1))
    midx = np.where(D == np.nanmax(D))[0][-1]
    return X_s[midx], Y_s[midx]


class ElbowTestCase(unittest.TestCase):
    def test_knee_of_a_curve(self):
        X = np.linspace(0, 1, 101)
        Y = np.minimum(X * 10, 1)
        self.assertEqual(get_elbow(X, Y), (X[10], Y[10]))

    def test_same_as_np_cross(self):
        rng = np.random.default_rng(29)
        for _ in range(200):
            n = rng.integers(3, 150)
            X, Y = rng.random(n) * rng.choice([1e-3, 1, 1e6]), rng.random(n)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    expected = _elbow_with_np_cross(X, Y)
            except ValueError:
                self.skipTest("this NumPy no longer has np.cross for 2-D vectors")
            self.assertEqual(get_elbow(X, Y), expected)


if __name__ == "__main__":
    unittest.main()
