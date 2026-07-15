"""Unit tests for model_options.py (fake tc/model stubs, no native lib)."""

import logging

import pytest

from wyoming_transcribe_cpp import model_options as mo


class _WhisperRunOptions:
    def __init__(self, *, initial_prompt=None, condition_on_prev_tokens=None,
                 temperature=None, temperature_inc=None,
                 compression_ratio_thold=None, logprob_thold=None,
                 no_speech_thold=None, max_prev_context_tokens=None,
                 seed=None, max_initial_timestamp=None):
        self.initial_prompt = initial_prompt
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.temperature = temperature
        self.temperature_inc = temperature_inc
        self.compression_ratio_thold = compression_ratio_thold
        self.logprob_thold = logprob_thold
        self.no_speech_thold = no_speech_thold
        self.max_prev_context_tokens = max_prev_context_tokens
        self.seed = seed
        self.max_initial_timestamp = max_initial_timestamp


class _ParakeetStreamOptions:
    def __init__(self, *, att_context_right=None):
        self.att_context_right = att_context_right


class _ParakeetBufferedStreamOptions:
    def __init__(self, *, left_ms=None, chunk_ms=None, right_ms=None):
        self.left_ms = left_ms
        self.chunk_ms = chunk_ms
        self.right_ms = right_ms


class _VoxtralRealtimeStreamOptions:
    def __init__(self, *, num_delay_tokens=None, min_decode_interval_ms=None):
        self.num_delay_tokens = num_delay_tokens
        self.min_decode_interval_ms = min_decode_interval_ms


class _MoonshineStreamingOptions:
    def __init__(self, *, min_decode_interval_ms=None):
        self.min_decode_interval_ms = min_decode_interval_ms


class _FakeTC:
    WhisperRunOptions = _WhisperRunOptions
    ParakeetStreamOptions = _ParakeetStreamOptions
    ParakeetBufferedStreamOptions = _ParakeetBufferedStreamOptions
    VoxtralRealtimeStreamOptions = _VoxtralRealtimeStreamOptions
    MoonshineStreamingOptions = _MoonshineStreamingOptions


class _FakeModel:
    """model.accepts() says yes only to instances of the configured class(es)."""

    def __init__(self, accepted_classes=()):
        self._accepted_classes = tuple(accepted_classes)

    def accepts(self, instance) -> bool:
        return isinstance(instance, self._accepted_classes)


@pytest.fixture(autouse=True)
def _capture_logs(caplog):
    caplog.set_level(logging.INFO, logger="wyoming_transcribe_cpp.model_options")


# --- parse_config -----------------------------------------------------------

def test_parse_config_happy_path():
    raw = '[{"name": "att_context_right", "value": "6"}, {"name": "seed", "value": "42"}]'
    assert mo.parse_config(raw) == {"att_context_right": "6", "seed": "42"}


def test_parse_config_empty_list():
    assert mo.parse_config("[]") == {}


def test_parse_config_malformed_json_is_ignored(caplog):
    assert mo.parse_config("not json") == {}
    assert "not valid JSON" in caplog.text


def test_parse_config_single_entry_bare_object():
    # Supervisor drops the [...] wrapper when model_options has exactly one
    # row, so bashio::config emits a bare object instead of a one-item array.
    assert mo.parse_config('{"name": "att_context_right", "value": "13"}') == {
        "att_context_right": "13"
    }


def test_parse_config_non_array_non_object_is_ignored(caplog):
    assert mo.parse_config('"just a string"') == {}
    assert "must be a JSON array" in caplog.text


def test_parse_config_entry_missing_value_is_skipped(caplog):
    raw = '[{"name": "att_context_right"}, {"name": "seed", "value": "1"}]'
    assert mo.parse_config(raw) == {"seed": "1"}
    assert "missing name/value" in caplog.text


# --- warn_unrecognized_keys ---------------------------------------------------

def test_warn_unrecognized_keys_warns_on_typo(caplog):
    mo.warn_unrecognized_keys({"att_context_rght": "6"})
    assert "not a recognized transcribe.cpp option" in caplog.text


def test_warn_unrecognized_keys_silent_for_known_field(caplog):
    mo.warn_unrecognized_keys({"att_context_right": "6"})
    assert caplog.text == ""


# --- build_family_extension ---------------------------------------------------

def test_build_family_extension_stream_slot_happy_path(caplog):
    model = _FakeModel(accepted_classes=(_ParakeetStreamOptions,))
    ext = mo.build_family_extension(_FakeTC, model, "stream", {"att_context_right": "6"})
    assert isinstance(ext, _ParakeetStreamOptions)
    assert ext.att_context_right == 6  # coerced to int
    # A successfully applied option must be visible at INFO -- the
    # user-facing signal that model_options actually took effect.
    assert "applying ParakeetStreamOptions" in caplog.text


def test_build_family_extension_run_slot_happy_path():
    model = _FakeModel(accepted_classes=(_WhisperRunOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "run",
        {"initial_prompt": "kitchen light", "temperature": "0.2"},
    )
    assert isinstance(ext, _WhisperRunOptions)
    assert ext.initial_prompt == "kitchen light"
    assert ext.temperature == pytest.approx(0.2)


def test_build_family_extension_drops_keys_not_on_the_class():
    # left_ms belongs to ParakeetBufferedStreamOptions, not ParakeetStreamOptions.
    model = _FakeModel(accepted_classes=(_ParakeetStreamOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "stream", {"att_context_right": "6", "left_ms": "200"},
    )
    assert isinstance(ext, _ParakeetStreamOptions)
    assert ext.att_context_right == 6


def test_build_family_extension_bad_coercion_dropped_siblings_survive(caplog):
    model = _FakeModel(accepted_classes=(_ParakeetStreamOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "stream", {"att_context_right": "not-a-number"},
    )
    # No valid kwargs survived -> this candidate is skipped entirely -> None.
    assert ext is None
    assert "is not a valid value" in caplog.text


def test_build_family_extension_returns_none_when_no_candidate_accepted(caplog):
    model = _FakeModel(accepted_classes=())  # accepts nothing
    ext = mo.build_family_extension(_FakeTC, model, "stream", {"att_context_right": "6"})
    assert ext is None
    assert "does not accept" in caplog.text
    # Correct name+value that just doesn't apply here is routine, not a
    # mistake — INFO, not WARNING (see module docstring's severity split).
    assert all(r.levelno == logging.INFO for r in caplog.records)


def test_build_family_extension_returns_none_when_no_keys_match_any_candidate():
    model = _FakeModel(accepted_classes=(_ParakeetStreamOptions,))
    ext = mo.build_family_extension(_FakeTC, model, "stream", {"unrelated_key": "x"})
    assert ext is None


def test_build_family_extension_name_collision_resolved_by_accepts(caplog):
    # min_decode_interval_ms exists on both Voxtral and Moonshine; accepts()
    # (not key order) determines which one is actually used. Regression
    # test: the first, wrong-family candidate (Voxtral, tried before
    # Moonshine) must NOT log a false "not accepted" warning just because
    # Moonshine ends up being the one that's actually used.
    model = _FakeModel(accepted_classes=(_MoonshineStreamingOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "stream", {"min_decode_interval_ms": "80"},
    )
    assert isinstance(ext, _MoonshineStreamingOptions)
    assert ext.min_decode_interval_ms == 80
    assert "not accepted" not in caplog.text
    assert "does not accept" not in caplog.text


def test_build_family_extension_bad_value_warns_even_if_another_key_succeeds(caplog):
    # att_context_right (Parakeet-only, invalid value) fails to coerce and
    # must still be reported even though left_ms (a different key, for a
    # different class) succeeds and gets returned.
    model = _FakeModel(accepted_classes=(_ParakeetBufferedStreamOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "stream",
        {"att_context_right": "not-a-number", "left_ms": "200"},
    )
    assert isinstance(ext, _ParakeetBufferedStreamOptions)
    assert ext.left_ms == 200
    assert "is not a valid value" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# --- condition_on_prev_tokens boolean coercion -------------------------------

@pytest.mark.parametrize("raw_value,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True), ("TRUE", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_condition_on_prev_tokens_recognizes_common_tokens(raw_value, expected):
    model = _FakeModel(accepted_classes=(_WhisperRunOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "run", {"condition_on_prev_tokens": raw_value},
    )
    assert ext.condition_on_prev_tokens is expected


def test_condition_on_prev_tokens_invalid_value_warns_instead_of_silently_false(caplog):
    model = _FakeModel(accepted_classes=(_WhisperRunOptions,))
    ext = mo.build_family_extension(
        _FakeTC, model, "run", {"condition_on_prev_tokens": "treu"},
    )
    # Unlike before: a typo must not silently become False with no trace.
    assert ext is None
    assert "is not a valid value" in caplog.text


# --- resolve_spec_k_drafts ---------------------------------------------------

def test_resolve_spec_k_drafts_default_is_minus_one():
    assert mo.resolve_spec_k_drafts({}) == -1


def test_resolve_spec_k_drafts_valid_value(caplog):
    assert mo.resolve_spec_k_drafts({"spec_k_drafts": "4"}) == 4
    assert "applying spec_k_drafts=4" in caplog.text


def test_resolve_spec_k_drafts_zero_means_off():
    assert mo.resolve_spec_k_drafts({"spec_k_drafts": "0"}) == 0


def test_resolve_spec_k_drafts_non_numeric_falls_back(caplog):
    assert mo.resolve_spec_k_drafts({"spec_k_drafts": "abc"}) == -1
    assert "is not an int" in caplog.text


def test_resolve_spec_k_drafts_below_minus_one_falls_back(caplog):
    assert mo.resolve_spec_k_drafts({"spec_k_drafts": "-5"}) == -1
    assert "must be -1" in caplog.text
