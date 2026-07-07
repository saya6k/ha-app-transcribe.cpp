"""Bootstrap model resolution: custom_model failures degrade gracefully."""

from pathlib import Path

from wyoming_transcribe_cpp import __main__ as main_mod
from wyoming_transcribe_cpp import convert
from wyoming_transcribe_cpp.__main__ import _parse_args, _resolve_model
from wyoming_transcribe_cpp.convert import ConversionFailed
from wyoming_transcribe_cpp.models import DEFAULT_MODEL, REGISTRY


class TestResolveModel:
    def test_catalog_model_without_custom(self, monkeypatch):
        args = _parse_args([])
        monkeypatch.setattr(
            main_mod, "ensure_gguf",
            lambda m, q, d, t: Path(f"/data/models/{m}.gguf"),
        )
        gguf, name, repo = _resolve_model(args, None)
        assert name == DEFAULT_MODEL
        assert repo == REGISTRY[DEFAULT_MODEL].repo

    def test_custom_model_served_when_conversion_succeeds(self, monkeypatch):
        args = _parse_args(["--custom-model", "me/ft"])
        monkeypatch.setattr(
            convert, "request_conversion",
            lambda *a, **kw: Path("/data/models/custom/me__ft-Q4_K_M.gguf"),
        )
        gguf, name, repo = _resolve_model(args, None)
        assert name == "me/ft"
        assert gguf.name == "me__ft-Q4_K_M.gguf"

    def test_conversion_failure_falls_back_to_catalog(self, monkeypatch, caplog):
        args = _parse_args(["--custom-model", "me/ft"])

        def boom(*a, **kw):
            raise ConversionFailed("converter died with SIGKILL")

        monkeypatch.setattr(convert, "request_conversion", boom)
        monkeypatch.setattr(
            main_mod, "ensure_gguf",
            lambda m, q, d, t: Path(f"/data/models/{m}.gguf"),
        )
        with caplog.at_level("ERROR"):
            gguf, name, repo = _resolve_model(args, None)
        assert name == DEFAULT_MODEL
        assert any("falling back" in r.message for r in caplog.records)

    def test_worker_socket_outage_also_falls_back(self, monkeypatch):
        args = _parse_args(["--custom-model", "me/ft"])

        def boom(*a, **kw):
            raise OSError("connect: no such file or directory")

        monkeypatch.setattr(convert, "request_conversion", boom)
        monkeypatch.setattr(
            main_mod, "ensure_gguf",
            lambda m, q, d, t: Path(f"/data/models/{m}.gguf"),
        )
        gguf, name, repo = _resolve_model(args, None)
        assert name == DEFAULT_MODEL
