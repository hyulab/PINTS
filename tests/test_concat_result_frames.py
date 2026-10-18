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
import pandas as pd
from pints.calling_engine import _concat_result_frames

COLUMNS = ("chromosome", "start", "end", "name", "pval", "reads", "mu_0", "pi_0", "mu_1", "pi_1", "var_1",
           "summit", "summit_val", "ler_1", "ler_2", "ler_3", "non_zeros")
ROW = ("chr1", 100, 105, "p1", 0.01, 7, 0.5, 0.1, 2.0, 0.2, 1.5, 102, 3, 1, 0, 1, 5)
INT_COLUMNS = ("start", "end", "reads", "summit", "summit_val", "ler_1", "ler_2", "ler_3", "non_zeros")


class ConcatResultFramesTestCase(unittest.TestCase):
    def setUp(self):
        # the same construction as check_window_chromosome: a chromosome without testable peaks gives an
        # empty frame whose columns are all object dtype
        self.full = pd.DataFrame([ROW, ROW[:1] + (200, 210) + ROW[3:]], columns=COLUMNS)
        self.empty = pd.DataFrame([], columns=COLUMNS)

    def test_integer_columns_stay_integer(self):
        merged = _concat_result_frames([self.full, self.empty, self.full])
        for col in INT_COLUMNS:
            self.assertTrue(pd.api.types.is_integer_dtype(merged[col]), col)

    def test_same_values_and_order(self):
        frames = [self.empty, self.full, self.empty, self.full]
        merged = _concat_result_frames(frames)
        plain = pd.concat(frames)
        self.assertEqual(merged.shape, plain.shape)
        self.assertEqual(merged.astype(object).values.tolist(), plain.astype(object).values.tolist())

    def test_all_empty(self):
        merged = _concat_result_frames([self.empty, self.empty])
        self.assertEqual(merged.shape[0], 0)
        self.assertEqual(tuple(merged.columns), COLUMNS)


if __name__ == "__main__":
    unittest.main()
