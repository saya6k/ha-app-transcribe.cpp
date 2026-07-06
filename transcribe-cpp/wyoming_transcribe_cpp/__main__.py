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


def _build_info(
    model: str, repo: str, engine: TranscribeEngine, streaming: bool
) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="Transcribe.cpp",
                description="transcribe.cpp GGUF ASR on ggml",
                attribution=Attribution(
                    name="handy-computer",
                    url="https://github.com/handy-computer/transcribe.cpp",
                ),
                installed=True,
                version=__version__,
                supports_transcript_streaming=streaming,
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


_LOG_LEVEL_MAP = {
    "trace": logging.DEBUG,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe.cpp Wyoming server")
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
        "--custom-model", default="",
        help="HF repo id of a fine-tuned checkpoint to convert+quantize "
             "on-device (overrides --model)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Fallback language when the client doesn't specify one",
    )
    parser.add_argument(
        "--speech-enhancement", action="store_true",
        help="Denoise each utterance with GTCRN before decoding",
    )
    parser.add_argument(
        "--diarization", action="store_true",
        help="Tag the final transcript with [Speaker N] labels",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=0,
        help="Diarization cluster count hint (0 = auto)",
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument(
        "--log-level", default="info", choices=sorted(_LOG_LEVEL_MAP),
        help="Log verbosity (HA log level names)",
    )
    return parser.parse_args(argv)


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=_LOG_LEVEL_MAP.get(args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    if args.custom_model:
        from .convert import ensure_custom_gguf

        gguf = ensure_custom_gguf(
            args.custom_model, args.quantization, args.model_dir, token
        )
        model_name = args.custom_model
        model_repo = args.custom_model
    else:
        gguf = ensure_gguf(args.model, args.quantization, args.model_dir, token)
        model_name = args.model
        model_repo = REGISTRY[args.model].repo
    _LOGGER.info("Model file ready: %s", gguf)

    engine = TranscribeEngine(str(gguf))
    try:
        _LOGGER.info("Warming up ...")
        engine.warmup()

        enhancer = None
        if args.speech_enhancement:
            from .enhance import Enhancer

            _LOGGER.info("Speech enhancement enabled — loading GTCRN ...")
            enhancer = Enhancer(args.model_dir)

        diarizer = None
        if args.diarization:
            from .diarize import Diarizer

            _LOGGER.info("Diarization enabled — loading sherpa-onnx models ...")
            diarizer = Diarizer(args.model_dir, max_speakers=args.max_speakers)

        from .streaming import decode_mode

        streaming = (
            decode_mode(
                engine.supports_streaming,
                enhancement=enhancer is not None,
                diarization=diarizer is not None,
            )
            == "stream"
        )
        wyoming_info = _build_info(model_name, model_repo, engine, streaming)
        server = AsyncServer.from_uri(args.uri)
        _LOGGER.info("Starting server on %s", args.uri)
        server_task = asyncio.create_task(
            server.run(
                partial(
                    TranscribeHandler, wyoming_info, args, engine,
                    enhancer, diarizer,
                )
            )
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
