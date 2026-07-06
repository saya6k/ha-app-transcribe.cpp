"""Unit tests for converter-family detection (fixtures only, no network)."""

import json
from pathlib import Path

import pytest

from wyoming_transcribe_cpp.convert import ConversionUnsupported
from wyoming_transcribe_cpp.detect import (
    CONVERT_SCRIPTS,
    FAMILIES,
    RepoProbe,
    detect_family,
)

FIXTURES = Path(__file__).parent / "fixtures" / "detect"


def probe_from_fixture(fam: str) -> RepoProbe:
    d = json.loads((FIXTURES / f"{fam}.json").read_text())
    return RepoProbe(
        repo=d["repo"],
        model_type=d["model_type"],
        architectures=d["architectures"],
        tags=d["tags"],
        files=d["files"],
    )


class TestRegistryIntegrity:
    def test_sixteen_families(self):
        assert len(FAMILIES) == 16

    def test_every_family_has_a_convert_script(self):
        for fam in FAMILIES:
            assert CONVERT_SCRIPTS[fam].startswith("convert-")
            assert CONVERT_SCRIPTS[fam].endswith(".py")

    def test_every_family_has_a_fixture(self):
        for fam in FAMILIES:
            assert (FIXTURES / f"{fam}.json").exists(), fam


class TestDetectFamily:
    @pytest.mark.parametrize("fam", sorted(FAMILIES))
    def test_representative_repo_detected(self, fam):
        assert detect_family(probe_from_fixture(fam)) == fam

    def test_nemo_file_without_hints_defaults_to_parakeet(self):
        probe = RepoProbe(
            repo="someone/my-finetune",
            files=["README.md", "my-finetune.nemo"],
        )
        assert detect_family(probe) == "parakeet"

    def test_nemo_repo_named_canary_routes_to_canary(self):
        probe = RepoProbe(
            repo="someone/canary-flash-ko",
            files=["canary-flash-ko.nemo"],
        )
        assert detect_family(probe) == "canary"

    def test_nemo_repo_with_qwen_hint_routes_to_canary_qwen(self):
        probe = RepoProbe(
            repo="someone/salm-qwen-ft",
            tags=["nemo", "Qwen"],
            files=["config.json", "model.safetensors"],
        )
        assert detect_family(probe) == "canary_qwen"

    def test_funasr_layout_with_qwen_llm_dir_is_funasr_nano(self):
        probe = RepoProbe(
            repo="someone/fun-asr-ft",
            files=[
                "Qwen3-0.6B/config.json", "config.yaml",
                "configuration.json", "model.pt", "multilingual.tiktoken",
            ],
        )
        assert detect_family(probe) == "funasr_nano"

    def test_funasr_layout_without_llm_dir_is_sensevoice(self):
        probe = RepoProbe(
            repo="someone/sensevoice-ft",
            files=["am.mvn", "config.yaml", "configuration.json", "model.pt"],
        )
        assert detect_family(probe) == "sensevoice"

    def test_unknown_model_type_raises_and_lists_families(self):
        probe = RepoProbe(
            repo="someone/llama-3", model_type="llama",
            files=["config.json", "model.safetensors"],
        )
        with pytest.raises(ConversionUnsupported) as err:
            detect_family(probe)
        assert "someone/llama-3" in str(err.value)
        for fam in FAMILIES:
            assert fam in str(err.value)

    def test_empty_probe_raises(self):
        with pytest.raises(ConversionUnsupported):
            detect_family(RepoProbe(repo="someone/empty"))
