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
"""--epig-annotation: local bigBed files are used without the web server; a failed lookup skips the annotation."""
import filecmp
import io
import json
import logging
import os
import socket
import urllib.error
import urllib.request


import pints.calling_engine as ce

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
LOCAL_BIGBED = os.path.join(DATASETS, "ENCFF948HPU.bigBed")


def _server_down(*args, **kwargs):
    raise urllib.error.HTTPError("https://pints.yulab.org/api/ea/k562", 502, "Bad Gateway", {}, None)


def _server_says(payload):
    return lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())


def _no_network(*args, **kwargs):
    raise AssertionError("the web server should not be contacted")


def test_local_file_is_used_without_the_server(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", _no_network)
    assert ce.get_epig_annotation(LOCAL_BIGBED, "/nonexistent/prefix") == LOCAL_BIGBED


def test_server_error_skips_with_a_warning_and_restores_the_timeout(monkeypatch, caplog):
    monkeypatch.setattr(urllib.request, "urlopen", _server_down)
    before = socket.getdefaulttimeout()
    with caplog.at_level(logging.WARNING, logger="PINTS - Caller"):
        assert ce.get_epig_annotation("k562", "/nonexistent/prefix") == ""
    assert socket.getdefaulttimeout() == before
    assert any(r.levelno == logging.WARNING and "skipping epigenomic annotation" in r.getMessage()
               for r in caplog.records)


def test_unknown_biosample_skips_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(urllib.request, "urlopen", _server_says({"status": 0, "msg": "Unknown biosample"}))
    with caplog.at_level(logging.WARNING, logger="PINTS - Caller"):
        assert ce.get_epig_annotation("not-a-biosample", "/nonexistent/prefix") == ""
    assert any("Unknown biosample" in r.getMessage() for r in caplog.records)


def test_server_answer_with_a_downloaded_file(monkeypatch, tmp_path):
    prefix = str(tmp_path / "ft")
    (tmp_path / "ft_ENCFF948HPU.bigBed").write_bytes(b"already here")
    monkeypatch.setattr(urllib.request, "urlopen",
                        _server_says({"status": 1, "msg": "https://example.invalid/ENCFF948HPU.bigBed"}))
    monkeypatch.setattr(urllib.request, "urlretrieve", _no_network)
    assert ce.get_epig_annotation("k562", prefix) == prefix + "_ENCFF948HPU.bigBed"


def test_failed_download_leaves_no_file(monkeypatch, tmp_path):
    prefix = str(tmp_path / "ft")
    monkeypatch.setattr(urllib.request, "urlopen",
                        _server_says({"status": 1, "msg": "https://example.invalid/ENCFF948HPU.bigBed"}))

    def broken_download(url, filename):
        with open(filename, "wb") as f:
            f.write(b"partial")
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken_download)
    assert ce.get_epig_annotation("k562", prefix) == ""
    assert not os.path.exists(prefix + "_ENCFF948HPU.bigBed")


def test_peak_calling_without_the_annotation_when_the_server_is_down(monkeypatch, tmp_path):
    """With the server down, --epig-annotation k562 gives the same peak files as a run without it."""
    monkeypatch.setattr(urllib.request, "urlopen", _server_down)
    args = dict(input_bam=None, bw_pl=(os.path.join(DATASETS, "pl.chr22.bw"),),
                bw_mn=(os.path.join(DATASETS, "mn.chr22.bw"),), output_dir=str(tmp_path), fdr_target=0.1,
                thread_n=4)
    ce.peak_calling(output_prefix="ea", epig_annotation="k562", **args)
    ce.peak_calling(output_prefix="plain", **args)
    for kind in ("bidirectional", "divergent", "unidirectional"):
        assert filecmp.cmp(tmp_path / f"ea_1_{kind}_peaks.bed", tmp_path / f"plain_1_{kind}_peaks.bed",
                           shallow=False), kind
