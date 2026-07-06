"""Speech enhancement via sherpa-onnx GTCRN (opt-in).

GTCRN (MIT) is a ~24k-parameter denoiser running on onnxruntime CPU; the
model is downloaded to /data/models/enhance/ on first enable, never baked
into the image. It denoises the whole buffered utterance at AudioStop
(decode_mode forces batch while enhancement is on).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

if TYPE_CHECKING:
    import numpy as np

_LOGGER = logging.getLogger(__name__)

GTCRN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)


def ensure_enhance_model(models_dir: str | Path) -> Path:
    dest = Path(models_dir) / "enhance" / "gtcrn_simple.onnx"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".onnx.part")
        _LOGGER.info("Downloading GTCRN model ...")
        urlretrieve(GTCRN_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.rename(dest)
    return dest


class Enhancer:
    """GTCRN denoiser over one buffered utterance."""

    def __init__(self, models_dir: str | Path) -> None:
        import sherpa_onnx

        model = ensure_enhance_model(models_dir)
        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=str(model)
                ),
            )
        )
        self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        self.sample_rate: int = self._denoiser.sample_rate

    def denoise(self, pcm: "np.ndarray", sample_rate: int) -> "np.ndarray":
        """Denoise float32 mono PCM; returns audio at the input rate."""
        import numpy as np

        denoised = self._denoiser.run(pcm, sample_rate=sample_rate)
        out = np.asarray(denoised.samples, dtype=np.float32)
        if denoised.sample_rate != sample_rate:
            # GTCRN runs at 16 kHz; ASR models here are 16 kHz too, so this
            # is defensive only (linear resample, good enough for ASR input).
            duration = out.size / float(denoised.sample_rate)
            n = int(round(duration * sample_rate))
            out = np.interp(
                np.linspace(0.0, out.size - 1, n),
                np.arange(out.size),
                out,
            ).astype(np.float32)
        return out
