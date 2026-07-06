"""Unit tests for the model registry and quantization resolution."""

import pytest

from wyoming_transcribe_cpp.models import (
    DEFAULT_MODEL,
    REGISTRY,
    gguf_cache_path,
    resolve_quant,
)


class TestResolveQuant:
    def test_exact_match(self):
        assert resolve_quant("q4_k_m", ["F16", "Q8_0", "Q4_K_M"]) == "Q4_K_M"

    def test_falls_back_to_nearest_larger(self):
        # Model publishes no K-quants — q4_k_m request lands on Q8_0,
        # the nearest larger available precision.
        assert resolve_quant("q4_k_m", ["F32", "F16", "Q8_0"]) == "Q8_0"

    def test_falls_back_to_f16_when_no_quants(self):
        assert resolve_quant("q8_0", ["F32", "F16"]) == "F16"

    def test_bf16_serves_as_f16_fallback(self):
        assert resolve_quant("f16", ["F32", "BF16"]) == "BF16"

    def test_unknown_request_raises(self):
        with pytest.raises(ValueError):
            resolve_quant("q2_k", ["F16"])

    def test_no_available_quants_raises(self):
        with pytest.raises(ValueError):
            resolve_quant("q4_k_m", [])


class TestRegistry:
    def test_default_model_present(self):
        assert DEFAULT_MODEL in REGISTRY

    def test_entries_have_required_fields(self):
        for name, entry in REGISTRY.items():
            assert entry.repo.startswith("handy-computer/"), name
            assert entry.license, name
            assert isinstance(entry.streaming, bool), name
            assert entry.quants, name  # quant name -> filename
            for quant, filename in entry.quants.items():
                assert filename.endswith(".gguf"), (name, quant)

    def test_default_model_has_q4_k_m(self):
        assert "Q4_K_M" in REGISTRY[DEFAULT_MODEL].quants


class TestCachePath:
    def test_layout_is_models_dir_slash_repo_slug(self, tmp_path):
        p = gguf_cache_path(
            tmp_path, "handy-computer/whisper-large-v3-turbo-gguf",
            "whisper-large-v3-turbo-Q4_K_M.gguf",
        )
        assert p == (
            tmp_path
            / "handy-computer__whisper-large-v3-turbo-gguf"
            / "whisper-large-v3-turbo-Q4_K_M.gguf"
        )
