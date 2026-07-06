"""Wyoming protocol handler — batch decode.

Audio chunks are buffered per utterance and decoded once at AudioStop.
(Streaming partials for streaming-capable models land in Phase 2.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .audio import pcm16_to_float32

if TYPE_CHECKING:
    from argparse import Namespace

    from .engine import TranscribeEngine

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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self._args = cli_args
        self._engine = engine
        self._language: str | None = cli_args.language
        self._chunks: list[np.ndarray] | None = None
        self._t0 = 0.0

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
                self._chunks = []
                self._t0 = time.monotonic()
                return True

            if AudioChunk.is_type(event.type):
                if self._chunks is None:
                    self._chunks = []
                chunk = AudioChunk.from_event(event)
                self._chunks.append(pcm16_to_float32(chunk.audio))
                return True

            if AudioStop.is_type(event.type):
                await self._finalize()
                return True

            return True
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            _LOGGER.debug("Client disconnected")
            self._chunks = None
            return False
        except Exception:
            _LOGGER.exception("Unexpected error in handle_event")
            self._chunks = None
            await self.write_event(
                Event("error", {"text": "Internal server error", "code": "internal"})
            )
            return False

    async def _finalize(self) -> None:
        if not self._chunks:
            _LOGGER.debug("AudioStop without audio")
            await self.write_event(Transcript(text="").event())
            return
        pcm = np.concatenate(self._chunks)
        self._chunks = None
        loop = asyncio.get_running_loop()
        async with _ASR_LOCK:
            text = await loop.run_in_executor(
                None, self._engine.transcribe, pcm, self._language
            )
        audio_s = pcm.size / float(self._engine.sample_rate)
        wall = time.monotonic() - self._t0
        _LOGGER.info(
            "Utterance: %.1fs audio, wall=%.2fs, RTF=%.2f -> %r",
            audio_s, wall, wall / audio_s if audio_s else 0.0, text,
        )
        await self.write_event(
            Transcript(text=text, language=self._language).event()
        )
