"""Unit tests for the original-weight identity sidecar + skip decisions."""

import json

from wyoming_transcribe_cpp.weighthash import (
    WeightIdentity,
    plan_action,
    sidecar_path,
    write_sidecar,
)

IDENT = WeightIdentity(
    repo="someone/my-whisper",
    revision="abc123",
    weight_hashes={"model.safetensors": "d0d0" * 16},
    transcribe_ref="d89ecb7",
)


def make_cache(tmp_path, identity=IDENT):
    gguf = tmp_path / "someone__my-whisper-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    write_sidecar(gguf, identity, family="whisper", quant="Q4_K_M")
    return gguf


class TestSidecar:
    def test_sidecar_sits_next_to_gguf(self, tmp_path):
        gguf = tmp_path / "x-Q4_K_M.gguf"
        assert sidecar_path(gguf) == tmp_path / "x-Q4_K_M.json"

    def test_sidecar_records_identity_family_and_quant(self, tmp_path):
        gguf = make_cache(tmp_path)
        data = json.loads(sidecar_path(gguf).read_text())
        assert data["repo"] == "someone/my-whisper"
        assert data["revision"] == "abc123"
        assert data["weight_hashes"] == IDENT.weight_hashes
        assert data["family"] == "whisper"
        assert data["quant"] == "Q4_K_M"


class TestPlanAction:
    def test_missing_gguf_converts(self, tmp_path):
        action, reason = plan_action(tmp_path / "missing.gguf", IDENT)
        assert action == "convert"
        assert "no cached" in reason

    def test_matching_sidecar_serves_cache(self, tmp_path):
        gguf = make_cache(tmp_path)
        action, reason = plan_action(gguf, IDENT)
        assert action == "serve"
        assert "cache hit" in reason

    def test_changed_weight_hash_reconverts(self, tmp_path):
        gguf = make_cache(tmp_path)
        current = WeightIdentity(
            repo=IDENT.repo, revision="def456",
            weight_hashes={"model.safetensors": "beef" * 16},
            transcribe_ref=IDENT.transcribe_ref,
        )
        action, reason = plan_action(gguf, current)
        assert action == "convert"
        assert "weights changed" in reason

    def test_readme_only_revision_bump_still_serves(self, tmp_path):
        gguf = make_cache(tmp_path)
        current = WeightIdentity(
            repo=IDENT.repo, revision="def456",
            weight_hashes=IDENT.weight_hashes,
            transcribe_ref=IDENT.transcribe_ref,
        )
        action, _ = plan_action(gguf, current)
        assert action == "serve"

    def test_new_transcribe_ref_reconverts(self, tmp_path):
        gguf = make_cache(tmp_path)
        current = WeightIdentity(
            repo=IDENT.repo, revision=IDENT.revision,
            weight_hashes=IDENT.weight_hashes,
            transcribe_ref="0000000",
        )
        action, reason = plan_action(gguf, current)
        assert action == "convert"
        assert "transcribe.cpp" in reason

    def test_unknown_transcribe_ref_is_not_compared(self, tmp_path):
        gguf = make_cache(tmp_path)
        current = WeightIdentity(
            repo=IDENT.repo, revision=IDENT.revision,
            weight_hashes=IDENT.weight_hashes,
            transcribe_ref="",
        )
        action, _ = plan_action(gguf, current)
        assert action == "serve"

    def test_legacy_gguf_without_sidecar_serves_flagged(self, tmp_path):
        gguf = tmp_path / "legacy-Q4_K_M.gguf"
        gguf.write_bytes(b"GGUF")
        action, reason = plan_action(gguf, IDENT)
        assert action == "serve"
        assert "legacy" in reason

    def test_corrupt_sidecar_reconverts(self, tmp_path):
        gguf = make_cache(tmp_path)
        sidecar_path(gguf).write_text("{not json")
        action, reason = plan_action(gguf, IDENT)
        assert action == "convert"
        assert "sidecar" in reason
