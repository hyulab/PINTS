"""--debug needs matplotlib (the "plots" extra) and says so before doing any work."""
import logging
import sys

import pytest

import pints.calling_engine as ce


def test_debug_without_matplotlib_stops_first(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # `import matplotlib` now raises ImportError
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        ce.peak_calling(str(tmp_path / "missing.bam"), output_dir=str(tmp_path), output_diagnostics=True,
                        check_updates=False)
    assert "pyPINTS[plots]" in caplog.text
