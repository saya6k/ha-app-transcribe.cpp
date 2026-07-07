"""Speech enhancement via FastEnhancer (opt-in).

FastEnhancer (MIT, https://github.com/aask1357/fastenhancer) is a streaming
neural denoiser running on onnxruntime CPU; the selected size's ONNX model
(22k-1.1M parameters, 16 kHz, DNS-Challenge trained) is downloaded to
/data/models/enhance/ on first enable, never baked into the image.

The exported model is stateful: each call takes one hop of samples plus the
cache_in_* tensors and returns one enhanced hop plus updated caches, so it
denoises frame-by-frame with (n_fft - hop) samples of algorithmic latency
(16-26 ms). Streaming decodes stay streaming with enhancement on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_N_FFT = 512  # all sizes; only the hop differs (read from the model input)

_RELEASE_URL = (
    "https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0"
)
SIZES = {
    "tiny": "t",
    "base": "b",
    "small": "s",
    "medium": "m",
    "large": "l",
}
DEFAULT_SIZE = "base"


def ensure_enhance_model(models_dir: str | Path, size: str) -> Path:
    name = f"fastenhancer_{SIZES[size]}.onnx"
    dest = Path(models_dir) / "enhance" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".onnx.part")
        _LOGGER.info("Downloading FastEnhancer (%s) model ...", size)
        urlretrieve(f"{_RELEASE_URL}/{name}", tmp)  # noqa: S310 - fixed https URL
        tmp.rename(dest)
    return dest


class EnhanceStream:
    """Frame-by-frame denoising state for one utterance."""

    def __init__(self, session) -> None:
        self._session = session
        self.hop: int = session.get_inputs()[0].shape[1]
        self._state = {
            i.name: np.zeros(i.shape, dtype=np.float32)
            for i in session.get_inputs()
            if i.name.startswith("cache_in_")
        }
        self._buf = np.empty(0, dtype=np.float32)

    def _run(self, frames: np.ndarray) -> np.ndarray:
        """Denoise hop-aligned samples (n*hop) frame by frame."""
        out = []
        for idx in range(0, frames.size, self.hop):
            self._state["wav_in"] = frames[np.newaxis, idx : idx + self.hop]
            result = self._session.run(None, self._state)
            out.append(result[0][0])
            for j, cache in enumerate(result[1:]):
                self._state[f"cache_in_{j}"] = cache
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Denoise the full hops available; buffer the remainder."""
        pcm = np.concatenate([self._buf, pcm])
        n = pcm.size - pcm.size % self.hop
        self._buf = pcm[n:]
        return self._run(pcm[:n])

    def flush(self) -> np.ndarray:
        """Zero-pad to drain the buffered remainder and the model latency."""
        tail = np.concatenate(
            [self._buf, np.zeros(_N_FFT, dtype=np.float32)]
        )
        self._buf = np.empty(0, dtype=np.float32)
        return self._run(tail[: tail.size - tail.size % self.hop])


class Enhancer:
    """FastEnhancer session shared across connections (16 kHz only)."""

    def __init__(self, models_dir: str | Path, size: str = DEFAULT_SIZE) -> None:
        import onnxruntime

        model = ensure_enhance_model(models_dir, size)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.sample_rate: int = SAMPLE_RATE

    def create_stream(self) -> EnhanceStream:
        return EnhanceStream(self._session)

    def denoise(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise one buffered utterance (the batch decode path)."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"FastEnhancer requires 16 kHz, got {sample_rate}")
        stream = self.create_stream()
        out = np.concatenate([stream.process(pcm), stream.flush()])
        latency = _N_FFT - stream.hop
        return out[latency : latency + pcm.size]
