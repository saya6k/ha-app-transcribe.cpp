"""Per-model runtime knobs (``model_options`` addon config) -> transcribe_cpp.

Upstream exposes family-specific run/stream tuning (Whisper prompt/decode
options, Parakeet streaming right-context, buffered-streaming L/C/R windows,
Voxtral-realtime delay, Moonshine streaming decode interval, speculative
decode draft count) through typed ``FamilyExtension`` subclasses and the
``spec_k_drafts`` kwarg. This module maps the addon's flat ``{name: value}``
config (see ``parse_config``) onto whichever of those the *loaded* model
actually accepts.

Everything here is pure Python: ``tc`` (the ``transcribe_cpp`` module) and
``model`` are passed in rather than imported, so this logic is testable
without the compiled native library — see ``tests/test_model_options.py``.

A value that doesn't apply to the loaded model is dropped, never raised —
but never at DEBUG either: the addon's default ``log_level`` is ``info``
(see ``__main__.py``'s ``_LOG_LEVEL_MAP``), which filters DEBUG out, and a
dropped option needs to show up without the user raising verbosity first.
Severity is split by how likely it is to be a genuine mistake:

- An unrecognized key name, or a recognized key whose *value* fails to
  coerce (a typo, wrong type) — WARNING. The name/value is wrong.
- A recognized key/value pair that simply doesn't apply to the loaded
  model's family (e.g. a Parakeet-only key set while a Whisper model is
  loaded) — INFO. The name/value are both fine; they're just for a
  different model, which is routine when one options list is reused
  across several model choices.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)

# slot -> candidate FamilyExtension class names, in the upstream
# bindings/python/src/transcribe_cpp/__init__.py module (pinned at
# TRANSCRIBE_REF). A loaded model belongs to exactly one family, so at most
# one candidate per slot ever passes model.accepts() — order doesn't affect
# correctness, only which candidate is tried (and warned about) first.
_EXTENSION_CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "run": ("WhisperRunOptions",),
    "stream": (
        "ParakeetStreamOptions",
        "ParakeetBufferedStreamOptions",
        "VoxtralRealtimeStreamOptions",
        "MoonshineStreamingOptions",
    ),
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _coerce_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ValueError(f"not a recognized boolean: {raw_value!r}")


# Coercion per upstream kwarg name (shared across classes — e.g.
# min_decode_interval_ms appears on both MoonshineStreamingOptions and
# VoxtralRealtimeStreamOptions with the same int type, so one entry covers
# both). Config values always arrive as strings (HA's {name, value} list).
FIELD_COERCE: dict[str, Callable[[str], Any]] = {
    # WhisperRunOptions
    "initial_prompt": str,
    "condition_on_prev_tokens": _coerce_bool,
    "temperature": float,
    "temperature_inc": float,
    "compression_ratio_thold": float,
    "logprob_thold": float,
    "no_speech_thold": float,
    "max_prev_context_tokens": int,
    "seed": int,
    "max_initial_timestamp": float,
    # ParakeetStreamOptions
    "att_context_right": int,
    # ParakeetBufferedStreamOptions
    "left_ms": int,
    "chunk_ms": int,
    "right_ms": int,
    # VoxtralRealtimeStreamOptions / MoonshineStreamingOptions
    "num_delay_tokens": int,
    "min_decode_interval_ms": int,
    # Session.run() kwarg, not a FamilyExtension field
    "spec_k_drafts": int,
}


def parse_config(raw: str) -> dict[str, str]:
    """Parse the addon's ``--model-options-json`` flag into a flat dict.

    ``raw`` is normally the JSON array of ``{"name": ..., "value": ...}``
    objects that ``bashio::config 'model_options'`` emits (HA stores
    list-of-object options as JSON already — see the ``run`` script). A
    single-row config comes through as a bare JSON object instead of a
    one-item array — Supervisor drops the ``[...]`` wrapper when the list
    has exactly one entry — so that shape is accepted too. Never raises:
    malformed input is warning-logged and treated as no options.
    """
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        _LOGGER.warning("model_options is not valid JSON (%r); ignoring", raw)
        return {}
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        _LOGGER.warning("model_options must be a JSON array; got %s; ignoring",
                        type(items).__name__)
        return {}
    options: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            _LOGGER.warning("model_options entry %r missing name/value; ignoring", item)
            continue
        options[str(item["name"])] = str(item["value"])
    return options


def warn_unrecognized_keys(options: dict[str, str]) -> None:
    """Warn once for any key that isn't a known transcribe.cpp option at all.

    ``build_family_extension`` only warns about keys once it's checking a
    specific candidate class — a key that matches *no* class in *any* slot
    (a typo, or a field from a family transcribe.cpp doesn't have) would
    otherwise never be mentioned anywhere. FIELD_COERCE's key set is exactly
    every real field name across every candidate class plus spec_k_drafts,
    so anything outside it is unrecognized regardless of the loaded model.
    """
    for key in options:
        if key not in FIELD_COERCE:
            _LOGGER.warning(
                "model_options key %r is not a recognized transcribe.cpp option; ignoring",
                key,
            )


def build_family_extension(tc: Any, model: Any, slot: str, options: dict[str, str]) -> Any | None:
    """Build the family extension (if any) the loaded model accepts for ``slot``.

    Two passes, deliberately: a bad *value* for a recognized key is a config
    mistake (WARNING, always — regardless of which class it was tried
    against), but "this key doesn't apply to the loaded model's family" is
    routine — the same options list is often reused across several models —
    so it's only reported (at INFO) once every same-slot candidate has been
    tried and none were accepted. Trying candidates one at a time and
    warning immediately on rejection would misfire: two classes can share a
    field name (e.g. ``min_decode_interval_ms`` on both
    ``VoxtralRealtimeStreamOptions`` and ``MoonshineStreamingOptions``), so
    the first, wrong-family candidate would otherwise log a false "not
    accepted" warning even though a later candidate goes on to succeed.
    """
    candidates: list[tuple[str, Any, dict[str, Any]]] = []
    for cls_name in _EXTENSION_CLASS_NAMES.get(slot, ()):
        cls = getattr(tc, cls_name, None)
        if cls is None:
            continue
        params = inspect.signature(cls.__init__).parameters
        kwargs: dict[str, Any] = {}
        for key, raw_value in options.items():
            if key not in params:
                continue
            coerce = FIELD_COERCE.get(key, str)
            try:
                kwargs[key] = coerce(raw_value)
            except (TypeError, ValueError):
                _LOGGER.warning(
                    "model_options: %s=%r is not a valid value for %s (%s); ignoring",
                    key, raw_value, cls_name, coerce.__name__,
                )
        if kwargs:
            candidates.append((cls_name, cls, kwargs))

    for cls_name, cls, kwargs in candidates:
        instance = cls(**kwargs)
        if model.accepts(instance):
            _LOGGER.info("model_options: applying %s (%s)", cls_name, kwargs)
            return instance

    for cls_name, cls, kwargs in candidates:
        _LOGGER.info(
            "model_options: loaded model does not accept %s (%s) — not an "
            "error, these options just don't apply to this model; ignoring",
            cls_name, kwargs,
        )
    return None


def resolve_spec_k_drafts(options: dict[str, str]) -> int:
    """Resolve the offline-only ``spec_k_drafts`` Session.run() kwarg.

    -1 (family default) unless the user set a valid ``spec_k_drafts``
    (an int >= -1, matching upstream's own validation). Never raises;
    ``Session.stream()`` ignores this — streaming always uses -1 upstream.
    """
    raw_value = options.get("spec_k_drafts")
    if raw_value is None:
        return -1
    try:
        value = int(raw_value)
    except ValueError:
        _LOGGER.warning("model_options: spec_k_drafts=%r is not an int; ignoring", raw_value)
        return -1
    if value < -1:
        _LOGGER.warning(
            "model_options: spec_k_drafts=%r must be -1 (family default), 0 (off), "
            "or a positive draft length; ignoring", raw_value,
        )
        return -1
    _LOGGER.info("model_options: applying spec_k_drafts=%d", value)
    return value
