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
from .enhance import DEFAULT_SIZE as FASTENHANCER_DEFAULT_SIZE
from .enhance import SIZES as FASTENHANCER_SIZES
from .engine import TranscribeEngine
from .handler import TranscribeHandler
from .model_options import parse_config as parse_model_options
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
        help="Denoise the audio with FastEnhancer before decoding",
    )
    parser.add_argument(
        "--fastenhancer-size", default=None, choices=sorted(FASTENHANCER_SIZES),
        help="FastEnhancer model size (default: base)",
    )
    parser.add_argument(
        "--model-options-json", default="[]",
        help="JSON array of {name, value} pairs — per-model runtime knobs "
             "(e.g. att_context_right, initial_prompt); ignored for models "
             "that don't support them",
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument(
        "--log-level", default="info", choices=sorted(_LOG_LEVEL_MAP),
        help="Log verbosity (HA log level names)",
    )
    return parser.parse_args(argv)


def _resolve_model(args, token):
    """Return (gguf, model_name, model_repo) for the server to load.

    Conversion runs in the unprivileged convert-worker; this process
    (transcribe user) only talks to it over the unix socket. A failed
    conversion is a configuration error: log the reason and let the
    exception stop the add-on cleanly (the s6 finish script halts the
    container), instead of silently serving a model the user did not
    ask for.
    """
    if args.custom_model:
        from . import convert

        try:
            gguf = convert.request_conversion(
                args.custom_model, args.quantization, args.model_dir, token
            )
        except (convert.ConversionFailed, OSError) as err:
            _LOGGER.error(
                "custom_model %s could not be converted: %s — stopping. "
                "Fix custom_model (or clear it to use the catalog model) "
                "and start the add-on again.",
                args.custom_model, err,
            )
            raise
        return gguf, args.custom_model, args.custom_model
    gguf = ensure_gguf(args.model, args.quantization, args.model_dir, token)
    return gguf, args.model, REGISTRY[args.model].repo


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=_LOG_LEVEL_MAP.get(args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    gguf, model_name, model_repo = _resolve_model(args, token)
    _LOGGER.info("Model file ready: %s", gguf)

    engine = TranscribeEngine(
        str(gguf), model_options=parse_model_options(args.model_options_json)
    )
    try:
        _LOGGER.info("Warming up ...")
        engine.warmup()

        enhancer = None
        if args.speech_enhancement:
            from .enhance import Enhancer

            size = args.fastenhancer_size or FASTENHANCER_DEFAULT_SIZE
            _LOGGER.info(
                "Speech enhancement enabled — loading FastEnhancer (%s) ...", size
            )
            enhancer = Enhancer(args.model_dir, size)
            if engine.sample_rate != enhancer.sample_rate:
                _LOGGER.warning(
                    "Model runs at %d Hz but FastEnhancer needs %d Hz — "
                    "disabling speech enhancement",
                    engine.sample_rate, enhancer.sample_rate,
                )
                enhancer = None

        wyoming_info = _build_info(
            model_name, model_repo, engine, engine.supports_streaming
        )
        server = AsyncServer.from_uri(args.uri)
        _LOGGER.info("Starting server on %s", args.uri)
        server_task = asyncio.create_task(
            server.run(
                partial(
                    TranscribeHandler, wyoming_info, args, engine, enhancer,
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
