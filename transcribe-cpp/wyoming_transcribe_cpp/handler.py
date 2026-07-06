"""Wyoming protocol handler.

Streaming-capable models (``engine.supports_streaming``) decode incrementally:
TranscriptStart -> TranscriptChunk deltas -> TranscriptStop + final Transcript.
Batch models buffer the utterance and decode once at AudioStop, emitting only
the final Transcript (their Info advertises no transcript streaming).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .audio import pcm16_to_float32
from .diarize import render_tagged
from .streaming import decode_mode, text_delta

if TYPE_CHECKING:
    from argparse import Namespace

    from .diarize import Diarizer
    from .engine import TranscribeEngine, TranscribeStream
    from .enhance import Enhancer

_LOGGER = logging.getLogger(__name__)

# One decode session shared across connections — serialize access.
_ASR_LOCK = asyncio.Lock()


class TranscribeHandler(AsyncEventHandler):
    """Wyoming ASR handler backed by a transcribe.cpp session."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: Namespace,
        engine: TranscribeEngine,
        enhancer: Enhancer | None = None,
        diarizer: Diarizer | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self._args = cli_args
        self._engine = engine
        self._enhancer = enhancer
        self._diarizer = diarizer
        self._mode = decode_mode(
            engine.supports_streaming,
            enhancement=enhancer is not None,
            diarization=diarizer is not None,
        )
        self._language: str | None = cli_args.language
        self._chunks: list[np.ndarray] | None = None
        self._stream: TranscribeStream | None = None
        self._last_partial = ""
        self._n_samples = 0
        self._t0 = 0.0

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.wyoming_info_event)
                return True

            if event.type == "transcribe":
                lang = (event.data or {}).get("language")
                if lang:
                    self._language = lang
                _LOGGER.debug("Transcribe request (language=%s)", self._language)
                return True

            if AudioStart.is_type(event.type):
                _LOGGER.debug("Utterance start")
                self._n_samples = 0
                self._t0 = time.monotonic()
                if self._mode == "stream":
                    self._close_stream()
                    async with _ASR_LOCK:
                        self._stream = self._engine.create_stream(self._language)
                    self._last_partial = ""
                    await self.write_event(
                        TranscriptStart(language=self._language).event()
                    )
                else:
                    self._chunks = []
                return True

            if AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                samples = pcm16_to_float32(chunk.audio)
                self._n_samples += samples.size
                if self._stream is not None:
                    loop = asyncio.get_running_loop()
                    async with _ASR_LOCK:
                        partial = await loop.run_in_executor(
                            None, self._stream.feed, samples
                        )
                    delta = text_delta(self._last_partial, partial)
                    if delta:
                        await self.write_event(TranscriptChunk(text=delta).event())
                    self._last_partial = partial
                else:
                    if self._chunks is None:
                        self._chunks = []
                    self._chunks.append(samples)
                return True

            if AudioStop.is_type(event.type):
                await self._finalize()
                return True

            return True
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            _LOGGER.debug("Client disconnected")
            self._close_stream()
            self._chunks = None
            return False
        except Exception:
            _LOGGER.exception("Unexpected error in handle_event")
            self._close_stream()
            self._chunks = None
            await self.write_event(
                Event("error", {"text": "Internal server error", "code": "internal"})
            )
            return False

    async def _finalize(self) -> None:
        loop = asyncio.get_running_loop()
        if self._stream is not None:
            try:
                async with _ASR_LOCK:
                    text = await loop.run_in_executor(None, self._stream.finalize)
            finally:
                self._close_stream()
            await self.write_event(TranscriptStop().event())
        elif self._chunks:
            pcm = np.concatenate(self._chunks)
            self._chunks = None
            async with _ASR_LOCK:
                if self._enhancer is not None:
                    pcm = await loop.run_in_executor(
                        None, self._enhancer.denoise, pcm, self._engine.sample_rate
                    )
                if self._diarizer is not None:
                    _, segments = await loop.run_in_executor(
                        None, self._engine.transcribe_result, pcm, self._language
                    )
                    diar = await loop.run_in_executor(
                        None, self._diarizer.diarize, pcm
                    )
                    text = render_tagged(segments, diar)
                else:
                    text = await loop.run_in_executor(
                        None, self._engine.transcribe, pcm, self._language
                    )
        else:
            _LOGGER.debug("AudioStop without audio")
            await self.write_event(Transcript(text="").event())
            return

        audio_s = self._n_samples / float(self._engine.sample_rate)
        wall = time.monotonic() - self._t0
        _LOGGER.info(
            "Utterance: %.1fs audio, wall=%.2fs, RTF=%.2f -> %r",
            audio_s, wall, wall / audio_s if audio_s else 0.0, text,
        )
        await self.write_event(
            Transcript(text=text, language=self._language).event()
        )
