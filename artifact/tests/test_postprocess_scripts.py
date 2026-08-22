import json

import pyarrow as pa

from scripts import probe_metadata_share, report_rerun


def test_report_load_ignores_nested_archives_and_partial_files(tmp_path):
    formal = tmp_path / "p2_dataset_c1024.jsonl"
    partial = tmp_path / "p2_dataset_c8192.partial-exit1-stamp.jsonl"
    archive = tmp_path / "pre_fix_archive"
    archive.mkdir()

    formal.write_text(json.dumps({"source": "formal"}) + "\n")
    partial.write_text(json.dumps({"source": "partial"}) + "\n")
    (archive / "p2_dataset_c1024.jsonl").write_text(
        json.dumps({"source": "archive"}) + "\n")

    assert report_rerun.load([str(tmp_path)]) == [{"source": "formal"}]


def test_probe_direct_enforces_requested_arrow_threads(tmp_path, monkeypatch):
    candidate = tmp_path / "layout.parquet"
    candidate.touch()
    seen = {}

    monkeypatch.setattr(probe_metadata_share, "parse_rate", lambda files: None)
    monkeypatch.setattr(probe_metadata_share, "_entries", lambda path: 10_000)

    def fake_direct(path, window, columns):
        seen["cpu"] = pa.cpu_count()
        seen["io"] = pa.io_thread_count()
        return True

    monkeypatch.setattr(probe_metadata_share, "direct", fake_direct)
    rc = probe_metadata_share.main(
        [str(candidate), "--direct", "--threads", "1"])

    assert rc == 0
    assert seen == {"cpu": 1, "io": 1}


def test_probe_direct_failure_is_nonzero(tmp_path, monkeypatch):
    candidate = tmp_path / "layout.parquet"
    candidate.touch()

    monkeypatch.setattr(probe_metadata_share, "parse_rate", lambda files: None)
    monkeypatch.setattr(probe_metadata_share, "_entries", lambda path: 10_000)
    monkeypatch.setattr(
        probe_metadata_share, "direct", lambda path, window, columns: False)

    assert probe_metadata_share.main([str(candidate), "--direct"]) == 2


def test_probe_rejects_nonpositive_threads(tmp_path):
    candidate = tmp_path / "layout.parquet"
    candidate.touch()

    try:
        probe_metadata_share.main(
            [str(candidate), "--direct", "--threads", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--threads=0 should be rejected")
