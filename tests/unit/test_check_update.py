"""check_update reads PyPI with urllib: retries, the version message, and no exception when offline."""
import io
import json
import logging
import urllib.error

import pints
import pints.calling_engine as ce


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_pypi(monkeypatch, failures, version):
    calls, waits = [], []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        if len(calls) <= failures:
            raise urllib.error.URLError("offline")
        return _Response(json.dumps({"info": {"version": version}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", waits.append)
    return calls, waits


def test_check_update_latest(monkeypatch, caplog):
    calls, waits = _fake_pypi(monkeypatch, 0, pints.__version__)
    with caplog.at_level(logging.INFO):
        ce.check_update()
    assert calls == [("https://pypi.org/pypi/pyPINTS/json", 10)] and waits == []
    assert "latest version" in caplog.text


def test_check_update_retries_then_warns(monkeypatch, caplog):
    calls, waits = _fake_pypi(monkeypatch, 2, "999.0")
    with caplog.at_level(logging.INFO):
        ce.check_update()
    assert len(calls) == 3 and waits == [0, 6, 10]  # two retries, then the 10 s pause after the warning
    assert "out-dated" in caplog.text


def test_check_update_offline_only_logs(monkeypatch, caplog):
    calls, waits = _fake_pypi(monkeypatch, 99, "999.0")
    with caplog.at_level(logging.ERROR):
        ce.check_update()  # does not raise
    assert len(calls) == 4 and waits == [0, 6, 12]
    assert "offline" in caplog.text
