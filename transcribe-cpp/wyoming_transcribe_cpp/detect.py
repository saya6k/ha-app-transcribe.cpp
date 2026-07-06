"""Detect which upstream converter family a HF checkpoint belongs to.

Detection uses HF Hub *metadata only* (model_type/architectures/tags/file
listing) — the hub API exposes these even for gated repos whose files
need a token, so family routing never downloads a weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .convert import ConversionUnsupported

_LOGGER = logging.getLogger(__name__)

# family -> upstream converter script (scripts/convert-*.py at the pinned
# TRANSCRIBE_REF). Family slugs equal upstream scripts/envs/<family> dirs.
CONVERT_SCRIPTS = {
    "canary": "convert-canary.py",
    "canary_qwen": "convert-canary-qwen.py",
    "cohere": "convert-cohere.py",
    "funasr_nano": "convert-funasr_nano.py",
    "gigaam": "convert-gigaam.py",
    "granite": "convert-granite.py",
    "granite_nar": "convert-granite_nar.py",
    "medasr": "convert-medasr.py",
    "moonshine": "convert-moonshine.py",
    "moonshine_streaming": "convert-moonshine_streaming.py",
    "parakeet": "convert-parakeet.py",
    "qwen3_asr": "convert-qwen3_asr.py",
    "sensevoice": "convert-sensevoice.py",
    "voxtral": "convert-voxtral.py",
    "voxtral_realtime": "convert-voxtral_realtime.py",
    "whisper": "convert-whisper.py",
}

FAMILIES = frozenset(CONVERT_SCRIPTS)

# HF config.json model_type -> family, verified against one representative
# repo per family (tests/fixtures/detect/). NeMo-format and FunASR-format
# repos publish no usable model_type and fall through to file heuristics.
_MODEL_TYPE_MAP = {
    "whisper": "whisper",
    "moonshine": "moonshine",
    "moonshine_streaming": "moonshine_streaming",
    "qwen3_asr": "qwen3_asr",
    "voxtral": "voxtral",
    "voxtral_realtime": "voxtral_realtime",
    "granite_speech": "granite",
    "granite_speech_nar": "granite_nar",
    "gigaam": "gigaam",
    "fastconformer": "canary",
    "lasr_ctc": "medasr",
    "cohere_asr": "cohere",
}


@dataclass
class RepoProbe:
    """Hub metadata snapshot a detection decision is made from."""

    repo: str
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def probe_from_hub(repo: str, token: str | None) -> RepoProbe:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, token=token)
    config = getattr(info, "config", None) or {}
    return RepoProbe(
        repo=repo,
        model_type=config.get("model_type", "") or "",
        architectures=list(config.get("architectures", []) or []),
        tags=list(info.tags or []),
        files=sorted(s.rfilename for s in (info.siblings or [])),
    )


def detect_family(probe: RepoProbe) -> str:
    fam = _detect(probe)
    if fam is None:
        raise ConversionUnsupported(
            f"could not detect a supported model family for "
            f"{probe.repo!r} (model_type={probe.model_type!r}). "
            f"Supported families: {', '.join(sorted(FAMILIES))}."
        )
    _LOGGER.info("Detected %s as family %r", probe.repo, fam)
    return fam


def _detect(probe: RepoProbe) -> str | None:
    if probe.model_type in _MODEL_TYPE_MAP:
        return _MODEL_TYPE_MAP[probe.model_type]

    if any(a.startswith("CohereAsr") for a in probe.architectures):
        return "cohere"

    files = set(probe.files)
    # FunASR-native checkpoints (SenseVoice, Fun-ASR-Nano): YAML config +
    # torch.save() blob instead of HF config.json/safetensors.
    if {"config.yaml", "configuration.json", "model.pt"} <= files:
        has_llm = any(
            f.startswith("Qwen") or f == "multilingual.tiktoken"
            for f in files
        )
        return "funasr_nano" if has_llm else "sensevoice"

    # NeMo exports: a .nemo archive, or hub metadata tagged nemo (the
    # canary-qwen SALM repo ships safetensors + nemo tag, no .nemo file).
    is_nemo = any(f.endswith(".nemo") for f in files) or any(
        t.lower() == "nemo" for t in probe.tags
    )
    if is_nemo:
        blob = " ".join(probe.tags).lower() + " " + probe.repo.lower()
        if "qwen" in blob:
            return "canary_qwen"
        if "canary" in blob:
            return "canary"
        return "parakeet"

    return None


# Catalog slug -> family. The curated registry keys are exactly the
# upstream variant slugs, so they double as the --variant pool for
# converters that validate the variant (whisper, moonshine, voxtral,
# parakeet, canary).
def _slug_family(slug: str) -> str | None:
    if slug.startswith("canary-qwen"):
        return "canary_qwen"
    if slug.startswith("canary"):
        return "canary"
    if slug.startswith("cohere"):
        return "cohere"
    if slug.startswith("fun-asr"):
        return "funasr_nano"
    if slug.startswith("gigaam"):
        return "gigaam"
    if slug.startswith("granite"):
        return "granite_nar" if slug.endswith("-nar") else "granite"
    if slug == "medasr":
        return "medasr"
    if slug.startswith("moonshine-streaming"):
        return "moonshine_streaming"
    if slug.startswith("moonshine"):
        return "moonshine"
    if slug.startswith(("parakeet", "nemotron")):
        return "parakeet"
    if slug.startswith("qwen3-asr"):
        return "qwen3_asr"
    if slug.startswith("sensevoice"):
        return "sensevoice"
    if slug.startswith("voxtral"):
        return "voxtral_realtime" if "realtime" in slug else "voxtral"
    if slug.startswith("whisper") or slug == "breeze-asr-25":
        return "whisper"
    return None


def variant_slugs(family: str) -> list[str]:
    from .models import REGISTRY

    return sorted(s for s in REGISTRY if _slug_family(s) == family)


def derive_variant(family: str, probe: RepoProbe) -> str | None:
    """Base-variant slug for a fine-tune, or None when underivable.

    Preference: the repo's ``base_model:*`` hub tag, then the longest
    catalog slug embedded in the repo name. None simply omits --variant;
    validating converters will then reject truly unidentifiable repos
    with their own message.
    """
    pool = variant_slugs(family)
    for tag in probe.tags:
        if tag.startswith("base_model:"):
            name = tag.rsplit("/", 1)[-1].lower()
            if name in pool:
                return name
    hay = probe.repo.rsplit("/", 1)[-1].lower().replace("_", "-")
    matches = [s for s in pool if s in hay]
    if matches:
        return max(matches, key=len)
    return None
