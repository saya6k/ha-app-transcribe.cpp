"""Per-connection conversion handler behind s6-ipcserverd.

s6-ipcserverd spawns this module with the accepted unix socket as
stdin/stdout, so the protocol is plain JSON lines on std streams:

    in : {"repo": ..., "quantization": ..., "hf_token": ..., "models_dir": ...}
    out: {"event": "log", "level": ..., "message": ...}   (streamed)
         {"event": "result", "ok": true, "gguf": ...}
         {"event": "result", "ok": false, "error": ...}

It runs as the unprivileged ``converter`` user and is the only process
allowed to write /data/convert-venv and /data/models/custom.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TextIO

from .const import MODELS_DIR
from .convert import ensure_custom_gguf


class _JsonLineHandler(logging.Handler):
    def __init__(self, out: TextIO) -> None:
        super().__init__()
        self._out = out

    def emit(self, record: logging.LogRecord) -> None:
        line = json.dumps({
            "event": "log",
            "level": record.levelname.lower(),
            "message": self.format(record),
        })
        self._out.write(line + "\n")
        self._out.flush()


def _result(out: TextIO, payload: dict) -> None:
    out.write(json.dumps({"event": "result", **payload}) + "\n")
    out.flush()


def serve(request_line: str, out: TextIO) -> None:
    handler = _JsonLineHandler(out)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        try:
            request = json.loads(request_line)
            repo = request["repo"]
            quantization = request.get("quantization", "q4_k_m")
        except (ValueError, KeyError, TypeError) as err:
            _result(out, {"ok": False, "error": f"bad request: {err}"})
            return
        try:
            gguf = ensure_custom_gguf(
                repo,
                quantization,
                request.get("models_dir") or MODELS_DIR,
                request.get("hf_token") or None,
            )
        except Exception as err:  # noqa: BLE001 — full reason to the client
            logging.getLogger(__name__).exception("conversion failed")
            _result(out, {"ok": False, "error": f"{type(err).__name__}: {err}"})
            return
        _result(out, {"ok": True, "gguf": str(gguf)})
    finally:
        root.removeHandler(handler)


def main() -> None:
    serve(sys.stdin.readline(), sys.stdout)


if __name__ == "__main__":
    main()
