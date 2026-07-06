"""Unit tests for the custom-model conversion pipeline (pure logic only)."""

import pytest

from wyoming_transcribe_cpp.convert import (
    ConversionUnsupported,
    custom_gguf_path,
    ensure_custom_gguf,
    pick_convert_script,
)


class TestCustomGgufPath:
    def test_layout_under_custom_dir(self, tmp_path):
        p = custom_gguf_path(tmp_path, "someone/my-whisper-ko", "Q4_K_M")
        assert p == tmp_path / "custom" / "someone__my-whisper-ko-Q4_K_M.gguf"


class TestPickConvertScript:
    def test_whisper_supported(self):
        assert pick_convert_script("whisper") == "convert-whisper.py"

    def test_unsupported_architecture_raises_with_hint(self):
        with pytest.raises(ConversionUnsupported) as err:
            pick_convert_script("wav2vec2")
        assert "wav2vec2" in str(err.value)


class TestEnsureCustomGguf:
    def test_cache_hit_skips_conversion(self, tmp_path):
        dest = custom_gguf_path(tmp_path, "someone/my-whisper", "Q4_K_M")
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"GGUF")
        # Would raise (no venv, no network) if the cache check didn't
        # short-circuit before any real work.
        out = ensure_custom_gguf("someone/my-whisper", "q4_k_m", tmp_path, None)
        assert out == dest
