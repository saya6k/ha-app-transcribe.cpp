"""Unit tests for the convert-worker JSON-lines protocol (no sockets)."""

import io
import json
from pathlib import Path

import pytest

from wyoming_transcribe_cpp import convert_worker
from wyoming_transcribe_cpp.convert import (
    ConversionFailed,
    _client_session,
)


def run_serve(monkeypatch, request: dict, ensure=None):
    if ensure is not None:
        monkeypatch.setattr(convert_worker, "ensure_custom_gguf", ensure)
    out = io.StringIO()
    convert_worker.serve(json.dumps(request), out)
    lines = [json.loads(l) for l in out.getvalue().splitlines()]
    assert lines, "worker must always emit at least a result line"
    return lines


class TestServe:
    def test_success_emits_result_with_gguf_path(self, monkeypatch):
        lines = run_serve(
            monkeypatch,
            {"repo": "a/b", "quantization": "q4_k_m", "models_dir": "/tmp/m"},
            ensure=lambda repo, quant, mdir, token: Path("/tmp/m/custom/x.gguf"),
        )
        result = lines[-1]
        assert result["event"] == "result"
        assert result["ok"] is True
        assert result["gguf"] == "/tmp/m/custom/x.gguf"

    def test_conversion_logs_are_streamed_as_events(self, monkeypatch):
        def ensure(repo, quant, mdir, token):
            import logging

            logging.getLogger("wyoming_transcribe_cpp.convert").info("hello %s", repo)
            return Path("/x.gguf")

        lines = run_serve(
            monkeypatch, {"repo": "a/b", "quantization": "q4_k_m"}, ensure=ensure
        )
        logs = [l for l in lines if l["event"] == "log"]
        assert any("hello a/b" in l["message"] for l in logs)

    def test_failure_emits_error_result(self, monkeypatch):
        def ensure(repo, quant, mdir, token):
            raise RuntimeError("converter exploded")

        lines = run_serve(
            monkeypatch, {"repo": "a/b", "quantization": "q4_k_m"}, ensure=ensure
        )
        result = lines[-1]
        assert result["ok"] is False
        assert "converter exploded" in result["error"]

    def test_malformed_request_emits_error_result(self, monkeypatch):
        out = io.StringIO()
        convert_worker.serve("{not json", out)
        result = json.loads(out.getvalue().splitlines()[-1])
        assert result["ok"] is False

    def test_missing_repo_emits_error_result(self, monkeypatch):
        lines = run_serve(monkeypatch, {"quantization": "q4_k_m"})
        assert lines[-1]["ok"] is False


class TestClientSession:
    def test_returns_path_on_ok_and_relays_logs(self, caplog):
        reader = io.StringIO(
            json.dumps({"event": "log", "level": "info", "message": "step 1"})
            + "\n"
            + json.dumps({"event": "result", "ok": True, "gguf": "/x.gguf"})
            + "\n"
        )
        writer = io.StringIO()
        with caplog.at_level("INFO"):
            path = _client_session(
                reader, writer, {"repo": "a/b", "quantization": "q4_k_m"}
            )
        assert path == Path("/x.gguf")
        assert json.loads(writer.getvalue())["repo"] == "a/b"
        assert any("step 1" in r.message for r in caplog.records)

    def test_raises_on_error_result(self):
        reader = io.StringIO(
            json.dumps({"event": "result", "ok": False, "error": "boom"}) + "\n"
        )
        with pytest.raises(ConversionFailed) as err:
            _client_session(reader, io.StringIO(), {"repo": "a/b"})
        assert "boom" in str(err.value)

    def test_raises_when_worker_hangs_up_early(self):
        with pytest.raises(ConversionFailed):
            _client_session(io.StringIO(""), io.StringIO(), {"repo": "a/b"})
