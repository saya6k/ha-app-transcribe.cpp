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

if TYPE_CHECKING:
    import numpy as np

_LOGGER = logging.getLogger(__name__)


class TranscribeEngine:
    """One loaded GGUF model + a single decode session (serialized by caller)."""

    def __init__(self, model_path: str, *, n_threads: int | None = None) -> None:
        import transcribe_cpp as tc

        self._tc = tc
        _LOGGER.info(
            "Loading %s (binding %s, native %s)",
            model_path, getattr(tc, "__version__", "?"), tc.native_version(),
        )
        self._model = tc.Model(model_path)
        caps = self._model.capabilities()
        self.sample_rate: int = caps.native_sample_rate
        self.languages: tuple[str, ...] = caps.languages
        self.supports_streaming: bool = caps.supports_streaming
        if n_threads is None:
            n_threads = min(os.cpu_count() or 4, 16)
        self._session = self._model.session(n_threads=n_threads)
        _LOGGER.info(
            "Model ready: arch=%s variant=%s backend=%s sample_rate=%d "
            "languages=%d streaming=%s",
            self._model.arch, self._model.variant, self._model.backend,
            self.sample_rate, len(self.languages), self.supports_streaming,
        )

    def transcribe(self, pcm: "np.ndarray", language: str | None) -> str:
        """Batch-decode one utterance (float32 mono at ``sample_rate``)."""
        result = self._session.run(pcm, language=self._normalize_language(language))
        return result.text.strip()

    def create_stream(self, language: str | None) -> "TranscribeStream":
        """Begin a streaming decode (requires ``supports_streaming``)."""
        stream = self._session.stream(language=self._normalize_language(language))
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
        """Flush the decoder and return the final text."""
        self._stream.finalize()
        return self._stream.text().display.strip()

    def close(self) -> None:
        """Reset the stream so the session can start the next one."""
        self._stream.reset()
