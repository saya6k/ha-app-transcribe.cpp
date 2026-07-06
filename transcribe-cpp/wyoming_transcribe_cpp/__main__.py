"""Entry point: download GGUF, load engine, serve Wyoming."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import MODELS_DIR, PORT, QUANT_ALIASES
from .engine import TranscribeEngine
from .handler import TranscribeHandler
from .models import DEFAULT_MODEL, REGISTRY, ensure_gguf

_LOGGER = logging.getLogger(__name__)


def _build_info(model: str, engine: TranscribeEngine) -> Info:
    repo = REGISTRY[model].repo
    return Info(
        asr=[
            AsrProgram(
                name="Transcribe (cpp)",
                description="transcribe.cpp GGUF ASR on ggml",
                attribution=Attribution(
                    name="handy-computer",
                    url="https://github.com/handy-computer/transcribe.cpp",
                ),
                installed=True,
                version=__version__,
                supports_transcript_streaming=engine.supports_streaming,
                models=[
                    AsrModel(
                        name=model,
                        languages=list(engine.languages),
                        attribution=Attribution(
                            name="handy-computer",
                            url=f"https://huggingface.co/{repo}",
                        ),
                        installed=True,
                        description=None,
                        version=None,
                    )
                ],
            )
        ],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe (cpp) Wyoming server")
    parser.add_argument("--uri", default=f"tcp://0.0.0.0:{PORT}")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=sorted(REGISTRY),
        help="Catalog model to download and serve",
    )
    parser.add_argument(
        "--quantization", default="q4_k_m", choices=sorted(QUANT_ALIASES),
        help="GGUF weight precision (falls back to nearest larger published)",
    )
    parser.add_argument("--model-dir", default=MODELS_DIR)
    parser.add_argument(
        "--language", default=None,
        help="Fallback language when the client doesn't specify one",
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    gguf = ensure_gguf(args.model, args.quantization, args.model_dir, token)
    _LOGGER.info("Model file ready: %s", gguf)

    engine = TranscribeEngine(str(gguf))
    try:
        _LOGGER.info("Warming up ...")
        engine.warmup()

        wyoming_info = _build_info(args.model, engine)
        server = AsyncServer.from_uri(args.uri)
        _LOGGER.info("Starting server on %s", args.uri)
        server_task = asyncio.create_task(
            server.run(partial(TranscribeHandler, wyoming_info, args, engine))
        )
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, server_task.cancel)
        loop.add_signal_handler(signal.SIGTERM, server_task.cancel)
        try:
            await server_task
        except asyncio.CancelledError:
            _LOGGER.info("Server stopped")
    finally:
        engine.close()


def run() -> None:
    try:
        asyncio.run(main())
    except Exception:
        _LOGGER.exception("Fatal error during bootstrap")
        raise SystemExit(1)


if __name__ == "__main__":
    run()
