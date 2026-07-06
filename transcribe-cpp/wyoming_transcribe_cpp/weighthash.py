"""Original-weight identity: skip reconversion when nothing relevant changed.

Identity of a custom model = the sha256 of every LFS file in the source
repo (weights, tokenizer blobs) plus the transcribe.cpp ref the converter
came from. Both come from HF Hub *metadata* (``files_metadata=True``) —
no weight download is needed to decide. The repo revision is recorded for
logs but a README-only revision bump does not retrigger conversion.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightIdentity:
    repo: str
    revision: str
    weight_hashes: dict[str, str] = field(default_factory=dict)
    transcribe_ref: str = ""


def current_transcribe_ref() -> str:
    return os.environ.get("TRANSCRIBE_REF", "")


def identity_from_hub(repo: str, token: str | None) -> WeightIdentity:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, token=token, files_metadata=True)
    hashes = {
        s.rfilename: s.lfs.sha256
        for s in (info.siblings or [])
        if s.lfs is not None
    }
    return WeightIdentity(
        repo=repo,
        revision=info.sha or "",
        weight_hashes=hashes,
        transcribe_ref=current_transcribe_ref(),
    )


def sidecar_path(gguf: Path) -> Path:
    return gguf.with_suffix(".json")


def write_sidecar(
    gguf: Path, identity: WeightIdentity, family: str, quant: str
) -> None:
    data = {**asdict(identity), "family": family, "quant": quant}
    sidecar_path(gguf).write_text(json.dumps(data, indent=1) + "\n")


def plan_action(gguf: Path, current: WeightIdentity) -> tuple[str, str]:
    """Return ("serve"|"convert", reason) for a cached GGUF path."""
    if not gguf.exists():
        return "convert", f"no cached GGUF at {gguf}"

    sidecar = sidecar_path(gguf)
    if not sidecar.exists():
        return "serve", (
            "legacy cache entry without weight-identity sidecar — serving "
            "as-is; delete the GGUF to force reconversion"
        )
    try:
        cached = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return "convert", f"unreadable sidecar {sidecar} — reconverting"

    if cached.get("weight_hashes") != current.weight_hashes:
        return "convert", (
            f"upstream weights changed for {current.repo} "
            f"(revision {cached.get('revision')} -> {current.revision})"
        )
    if (
        current.transcribe_ref
        and cached.get("transcribe_ref")
        and cached["transcribe_ref"] != current.transcribe_ref
    ):
        return "convert", (
            "transcribe.cpp converter updated "
            f"({cached['transcribe_ref']} -> {current.transcribe_ref})"
        )
    if cached.get("revision") != current.revision:
        _LOGGER.info(
            "%s revision moved to %s but weight files are identical — "
            "keeping cache", current.repo, current.revision,
        )
    return "serve", f"cache hit for {current.repo} ({gguf.name})"
