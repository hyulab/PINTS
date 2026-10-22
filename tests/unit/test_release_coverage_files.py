"""release_coverage_files deletes unused coverage dumps in the background; housekeeping waits for it."""
import os

import numpy as np

import pints.calling_engine as ce


def test_release_keeps_live_paths_and_housekeeping_waits(tmp_path):
    paths = {}
    for name in ("rep0_c1", "rep0_c2", "rep1_c1", "merged_c1", "zero_c2"):
        p = str(tmp_path / (name + ".npy"))
        np.save(p, np.arange(10))
        paths[name] = p
    other = tmp_path / "notes.txt"
    other.write_text("x")
    old = [{"c1": paths["rep0_c1"], "c2": paths["rep0_c2"]}, {"c1": paths["rep1_c1"], "c2": paths["zero_c2"]},
           {"c3": str(other)}]
    live = [{"c1": paths["merged_c1"], "c2": paths["zero_c2"]}]  # a path shared with a live dict is kept
    ce.release_coverage_files(old, live)
    ce.housekeeping()
    assert not ce._background_deletions
    assert sorted(os.listdir(tmp_path)) == ["merged_c1.npy", "notes.txt", "zero_c2.npy"]


def test_release_missing_files_is_silent(tmp_path):
    ce.release_coverage_files([{"c": str(tmp_path / "gone.npy")}])
    ce.housekeeping()
    assert not ce._background_deletions
