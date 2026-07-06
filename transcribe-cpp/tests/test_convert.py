"""Unit tests for the custom-model conversion pipeline (pure logic only)."""

from pathlib import Path

import pytest

from wyoming_transcribe_cpp import convert
from wyoming_transcribe_cpp.convert import (
    CPU_INDEX,
    OUTDIR_FAMILIES,
    ConversionUnsupported,
    converter_cmd,
    custom_gguf_path,
    ensure_custom_gguf,
    parse_env_deps,
    pip_commands,
    venv_dir,
)
from wyoming_transcribe_cpp.detect import CONVERT_SCRIPTS, FAMILIES
from wyoming_transcribe_cpp.weighthash import WeightIdentity, write_sidecar

ENV_FIXTURES = Path(__file__).parent / "fixtures" / "envs"


def fixture_deps(family: str) -> list[str]:
    return parse_env_deps((ENV_FIXTURES / family / "pyproject.toml").read_text())


class TestCustomGgufPath:
    def test_layout_under_custom_dir(self, tmp_path):
        p = custom_gguf_path(tmp_path, "someone/my-whisper-ko", "Q4_K_M")
        assert p == tmp_path / "custom" / "someone__my-whisper-ko-Q4_K_M.gguf"


class TestParseEnvDeps:
    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_every_family_pyproject_parses(self, family):
        deps = fixture_deps(family)
        assert deps, family

    def test_upstream_pins_survive_verbatim(self):
        assert "transformers==5.6.1" in fixture_deps("whisper")
        assert "transformers==4.57.6" in fixture_deps("voxtral")
        assert "funasr==1.3.1" in fixture_deps("sensevoice")

    def test_git_specs_intact(self):
        assert any(
            d.startswith("nemo-toolkit") and "git+https://" in d
            for d in fixture_deps("parakeet")
        )
        assert any("git+https://" in d for d in fixture_deps("gigaam"))


class TestPipCommands:
    def test_torch_bootstrapped_from_cpu_index_only(self):
        cmds = pip_commands("/venv/bin/python3", fixture_deps("whisper"))
        boot = cmds[0]
        assert "--index-url" in boot and CPU_INDEX in boot
        assert "torch" in boot
        assert "torchaudio" not in boot

    def test_torchaudio_joins_bootstrap_when_family_needs_it(self):
        boot = pip_commands("/v/bin/python3", fixture_deps("sensevoice"))[0]
        assert "torchaudio" in boot

    def test_main_install_carries_all_deps_and_cpu_extra_index(self):
        deps = fixture_deps("voxtral")
        main = pip_commands("/v/bin/python3", deps)[1]
        assert "--extra-index-url" in main and CPU_INDEX in main
        for dep in deps:
            assert dep in main

    def test_no_cuda_index_anywhere(self):
        for family in sorted(FAMILIES):
            for cmd in pip_commands("/v/bin/python3", fixture_deps(family)):
                assert not any("cu1" in c or "cuda" in c for c in cmd)


class TestVenvDir:
    def test_one_venv_per_family(self):
        dirs = {venv_dir(f) for f in FAMILIES}
        assert len(dirs) == len(FAMILIES)
        for d in dirs:
            assert str(d).startswith("/data/convert-venv/")


class TestConverterCmd:
    OUT = Path("/data/models/custom/o__m-REF.gguf")

    def test_uniform_family_gets_positional_out_and_revision(self):
        cmd = converter_cmd("whisper", "o/m", self.OUT, revision="abc123")
        assert cmd[1].endswith("convert-whisper.py")
        assert cmd[2:] == ["o/m", str(self.OUT), "--repo-id", "o/m",
                           "--revision", "abc123"]

    def test_nemo_families_have_no_revision_flag(self):
        cmd = converter_cmd("canary", "o/m", self.OUT, revision="abc123")
        assert "--revision" not in cmd

    def test_outdir_families_use_outdir_instead_of_positional(self):
        cmd = converter_cmd("granite_nar", "o/m", self.OUT, revision="r1")
        assert str(self.OUT) not in cmd
        assert "--outdir" in cmd
        outdir = cmd[cmd.index("--outdir") + 1]
        assert outdir.startswith(str(self.OUT.parent))

    def test_gigaam_is_rejected_with_guidance(self):
        with pytest.raises(ConversionUnsupported) as err:
            converter_cmd("gigaam", "o/m", self.OUT)
        assert "curated" in str(err.value)

    @pytest.mark.parametrize(
        "family", sorted(FAMILIES - {"gigaam"})
    )
    def test_every_family_routes_to_its_script(self, family):
        cmd = converter_cmd(family, "o/m", self.OUT)
        assert cmd[1].endswith(CONVERT_SCRIPTS[family])
        assert cmd[0].startswith(str(venv_dir(family)))


IDENT = WeightIdentity(
    repo="someone/my-whisper", revision="abc",
    weight_hashes={"model.safetensors": "d0d0"},
)


class TestEnsureCustomGguf:
    def test_sidecar_cache_hit_skips_everything(self, tmp_path, monkeypatch):
        dest = custom_gguf_path(tmp_path, "someone/my-whisper", "Q4_K_M")
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"GGUF")
        write_sidecar(dest, IDENT, family="whisper", quant="Q4_K_M")
        monkeypatch.setattr(convert, "identity_from_hub", lambda r, t: IDENT)
        out = ensure_custom_gguf("someone/my-whisper", "q4_k_m", tmp_path, None)
        assert out == dest

    def test_hub_unreachable_serves_existing_cache(self, tmp_path, monkeypatch):
        dest = custom_gguf_path(tmp_path, "someone/my-whisper", "Q4_K_M")
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"GGUF")

        def boom(repo, token):
            raise OSError("offline")

        monkeypatch.setattr(convert, "identity_from_hub", boom)
        out = ensure_custom_gguf("someone/my-whisper", "q4_k_m", tmp_path, None)
        assert out == dest

    def test_hub_unreachable_without_cache_raises(self, tmp_path, monkeypatch):
        def boom(repo, token):
            raise OSError("offline")

        monkeypatch.setattr(convert, "identity_from_hub", boom)
        with pytest.raises(OSError):
            ensure_custom_gguf("someone/none", "q4_k_m", tmp_path, None)

    def test_unknown_quant_rejected_early(self, tmp_path):
        with pytest.raises(ValueError):
            ensure_custom_gguf("someone/x", "q17_z", tmp_path, None)
