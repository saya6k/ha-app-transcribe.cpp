"""Speaker diarization via sherpa-onnx (opt-in, models downloaded on enable).

Models (both permissively licensed, fetched to /data/models/diarization/):
- segmentation: pyannote segmentation-3.0 (MIT), ONNX export from the
  sherpa-onnx release assets
- embedding: 3D-Speaker CAM++ zh_en advanced (Apache-2.0)

Tag rendering is pure logic (unit-tested); the sherpa-onnx import happens
only inside :class:`Diarizer`.
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from urllib.request import urlretrieve

if TYPE_CHECKING:
    import numpy as np

_LOGGER = logging.getLogger(__name__)

_ASSETS = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = (
    f"{_ASSETS}/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    f"{_ASSETS}/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)


@dataclass(frozen=True)
class DiarSegment:
    start: float
    end: float
    speaker: int  # 0-based


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def render_tagged(
    asr_segments: Sequence[tuple[float, float, str]],
    diar_segments: Sequence[DiarSegment],
) -> str:
    """Merge ASR segments with speaker turns into '[Speaker N] text …'.

    Each ASR segment is attributed to the speaker with the largest temporal
    overlap; zero-span segments (models without timestamps) fall back to the
    speaker who talks the most overall. Without diarization output the plain
    text is returned untouched.
    """
    if not diar_segments:
        return " ".join(text for _, _, text in asr_segments).strip()

    totals: dict[int, float] = {}
    for d in diar_segments:
        totals[d.speaker] = totals.get(d.speaker, 0.0) + (d.end - d.start)
    dominant = max(totals, key=lambda s: totals[s])

    parts: list[str] = []
    current: int | None = None
    for t0, t1, text in asr_segments:
        overlaps = {
            d.speaker: _overlap(t0, t1, d.start, d.end) for d in diar_segments
        }
        best = max(overlaps.values(), default=0.0)
        speaker = (
            max(overlaps, key=lambda s: overlaps[s]) if best > 0.0 else dominant
        )
        if speaker != current:
            parts.append(f"[Speaker {speaker + 1}]")
            current = speaker
        parts.append(text.strip())
    return " ".join(p for p in parts if p).strip()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _LOGGER.info("Downloading %s ...", url)
    urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
    tmp.rename(dest)


def ensure_diarization_models(models_dir: str | Path) -> tuple[Path, Path]:
    """Download segmentation + embedding models if missing; return paths."""
    base = Path(models_dir) / "diarization"
    seg_model = base / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if not seg_model.exists():
        archive = base / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(base, filter="data")
        archive.unlink()
    if not seg_model.exists():
        raise FileNotFoundError(f"Segmentation model missing after extract: {seg_model}")
    emb_model = base / EMBEDDING_URL.rsplit("/", 1)[1]
    if not emb_model.exists():
        _download(EMBEDDING_URL, emb_model)
    return seg_model, emb_model


class Diarizer:
    """sherpa-onnx offline speaker diarization on one buffered utterance."""

    def __init__(self, models_dir: str | Path, max_speakers: int = 0) -> None:
        import sherpa_onnx

        seg_model, emb_model = ensure_diarization_models(models_dir)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model)
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=max_speakers if max_speakers > 0 else -1
            ),
        )
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        self.sample_rate = self._sd.sample_rate

    def diarize(self, pcm: "np.ndarray") -> list[DiarSegment]:
        """Run diarization on float32 mono PCM at ``self.sample_rate``."""
        result = self._sd.process(pcm).sort_by_start_time()
        return [
            DiarSegment(start=s.start, end=s.end, speaker=s.speaker)
            for s in result
        ]
