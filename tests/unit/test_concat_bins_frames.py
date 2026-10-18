"""Empty per-chromosome result frames must not upcast integer columns to object dtype."""
import pandas as pd

from pints.calling_engine import _bins_to_df, _concat_result_frames

ROW = ("chr1", 100, 105, "p1", 0.01, 7, 0.5, 0.1, 2.0, 0.2, 1.5, 102, 3, 1, 0, 1, 5)


def test_empty_frame_does_not_upcast():
    empty = _bins_to_df([], "+")
    full = _bins_to_df([ROW, ROW[:1] + (200, 210) + ROW[3:]], "+")
    assert pd.concat([full, empty])["reads"].dtype == object  # the pandas behaviour being avoided
    got = _concat_result_frames([empty, full, empty])
    for col in ("start", "end", "reads", "summit", "summit_val", "ler_1", "ler_2", "ler_3", "non_zeros"):
        assert got[col].dtype.kind == "i", col
    pd.testing.assert_frame_equal(got, pd.concat([empty, full, empty]), check_dtype=False)


def test_all_empty_keeps_original_behaviour():
    empty = _bins_to_df([], "-")
    pd.testing.assert_frame_equal(_concat_result_frames([empty, empty]), pd.concat([empty, empty]))
