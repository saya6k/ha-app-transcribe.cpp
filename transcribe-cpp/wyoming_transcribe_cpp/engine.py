"""transcribe.cpp binding wrapper.

The upstream ``transcribe_cpp`` package is pure ctypes; it finds our built
``libtranscribe.so`` through the ``TRANSCRIBE_LIBRARY`` env var (set in the
Dockerfile). Import is deferred to ``TranscribeEngine`` so unit tests of the
pure-Python modules never touch the native library.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from . import model_options as mo

if TYPE_CHECKING:
    import numpy as np

_LOGGER = logging.getLogger(__name__)


class TranscribeEngine:
    """One loaded GGUF model + a single decode session (serialized by caller)."""

    def __init__(
        self, model_path: str, *, n_threads: int | None = None,
        model_options: dict[str, str] | None = None,
    ) -> None:
        import transcribe_cpp as tc

        self._tc = tc
        _LOGGER.info(
            "Loading %s (binding %s, native %s)",
            model_path, getattr(tc, "__version__", "?"), tc.native_version(),
        )
        self._model = tc.Model(model_path)
        caps = self._model.capabilities  # property on the binding
        self.sample_rate: int = caps.native_sample_rate
        self.languages: tuple[str, ...] = caps.languages
        self.supports_streaming: bool = caps.supports_streaming
        if n_threads is None:
            n_threads = min(os.cpu_count() or 4, 16)
        self._session = self._model.session(n_threads=n_threads)

        # Resolved once here (not per-utterance): the extension objects hold
        # plain Python values and are safe to reuse across every run()/
        # stream() call, and this is where warnings about unsupported/
        # unrecognized model_options belong — once at startup, not spammed
        # on every utterance.
        options = model_options or {}
        mo.warn_unrecognized_keys(options)
        self._run_family = mo.build_family_extension(tc, self._model, "run", options)
        self._stream_family = mo.build_family_extension(tc, self._model, "stream", options)
        self._spec_k_drafts = mo.resolve_spec_k_drafts(options)
        # Only take the defensive retry-without-family path when something
        # non-default was actually resolved — otherwise every call (the
        # overwhelming common case: no model_options configured at all)
        # would have any unrelated UnsupportedRequest/InvalidArgument (e.g.
        # a genuinely empty PCM buffer) mislabeled as a model_options
        # rejection and silently retried instead of surfacing as-is.
        self._run_has_extras = self._run_family is not None or self._spec_k_drafts != -1
        self._stream_has_extras = self._stream_family is not None

        _LOGGER.info(
            "Model ready: arch=%s variant=%s backend=%s sample_rate=%d "
            "languages=%d streaming=%s",
            self._model.arch, self._model.variant, self._model.backend,
            self.sample_rate, len(self.languages), self.supports_streaming,
        )

    def transcribe(self, pcm: "np.ndarray", language: str | None) -> str:
        """Batch-decode one utterance (float32 mono at ``sample_rate``)."""
        language = self._normalize_language(language)
        if not self._run_has_extras:
            result = self._session.run(pcm, language=language)
            return result.text.strip()
        try:
            result = self._session.run(
                pcm, language=language, spec_k_drafts=self._spec_k_drafts,
                family=self._run_family,
            )
        except (self._tc.UnsupportedRequest, self._tc.InvalidArgument) as err:
            _LOGGER.warning(
                "model_options rejected at run time (%s); retrying without them", err
            )
            result = self._session.run(pcm, language=language)
        return result.text.strip()

    def create_stream(self, language: str | None) -> "TranscribeStream":
        """Begin a streaming decode (requires ``supports_streaming``)."""
        language = self._normalize_language(language)
        if not self._stream_has_extras:
            return TranscribeStream(self._session.stream(language=language))
        try:
            stream = self._session.stream(language=language, family=self._stream_family)
        except (self._tc.UnsupportedRequest, self._tc.InvalidArgument) as err:
            _LOGGER.warning(
                "model_options rejected at stream time (%s); retrying without them", err
            )
            stream = self._session.stream(language=language)
        return TranscribeStream(stream)

    def warmup(self) -> None:
        """Fault weights in from disk with a short silent decode."""
        import numpy as np

        self.transcribe(np.zeros(self.sample_rate, dtype=np.float32), None)

    def _normalize_language(self, language: str | None) -> str | None:
        """Map a pipeline language tag onto what the model advertises."""
        if not language:
            return None
        lang = language.split("-")[0].lower()
        if self.languages and lang not in self.languages:
            _LOGGER.debug(
                "Language %r not advertised by model; letting it auto-detect", lang
            )
            return None
        return lang

    def close(self) -> None:
        self._session.close()
        self._model.close()


class TranscribeStream:
    """One in-flight streaming decode (a session runs at most one)."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def feed(self, pcm: "np.ndarray") -> str:
        """Feed samples; return the current hypothesis (committed+tentative)."""
        self._stream.feed(pcm)
        return self._stream.text().display

    def finalize(self) -> str:
        """Flush the decoder and return the final text.

        ``full`` is the complete final hypothesis; ``display``
        (committed+tentative) can lag it after a late revision.
        """
        self._stream.finalize()
        return self._stream.text().full.strip()

    def close(self) -> None:
        """Reset the stream so the session can start the next one."""
        self._stream.reset()
