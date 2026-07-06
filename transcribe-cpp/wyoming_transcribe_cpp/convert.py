"""Custom fine-tuned model conversion: HF checkpoint -> GGUF -> quantized.

torch is never in the image. On first use we create a persistent venv under
/data/convert-venv with torch-cpu + the upstream converter deps, then run
the upstream ``convert-<family>.py`` (shipped in the image at
/usr/local/share/transcribe-cpp/scripts) and ``transcribe-quantize``.
Conversion is CPU-only and slow by design; progress goes to the app log.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from .const import QUANT_ALIASES
from .models import resolve_quant

_LOGGER = logging.getLogger(__name__)

SCRIPTS_DIR = Path("/usr/local/share/transcribe-cpp/scripts")
VENV_DIR = Path("/data/convert-venv")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# HF config.json model_type -> upstream converter. Whisper is the supported
# fine-tune path for v1; other families need per-family deps/validation and
# are added deliberately, not by default.
CONVERT_SCRIPTS = {
    "whisper": "convert-whisper.py",
}

# Mirrors upstream scripts/envs/whisper/pyproject.toml (torch pulled from the
# CPU wheel index to stay small and image-free).
CONVERT_DEPS = [
    "torch>=2.2",
    "gguf>=0.10",
    "huggingface-hub>=0.30",
    "librosa>=0.10",
    "safetensors>=0.4",
    "soundfile>=0.12",
    "numpy>=1.26",
    "transformers==5.6.1",
]


class ConversionUnsupported(Exception):
    pass


def custom_gguf_path(models_dir: str | Path, repo: str, quant: str) -> Path:
    """Cache location: <models_dir>/custom/<owner>__<name>-<QUANT>.gguf."""
    return Path(models_dir) / "custom" / f"{repo.replace('/', '__')}-{quant}.gguf"


def pick_convert_script(model_type: str) -> str:
    script = CONVERT_SCRIPTS.get(model_type)
    if script is None:
        raise ConversionUnsupported(
            f"custom_model architecture {model_type!r} is not supported for "
            f"on-device conversion yet (supported: {sorted(CONVERT_SCRIPTS)}). "
            "Convert it on a workstation with upstream transcribe.cpp instead."
        )
    return script


def _detect_model_type(repo: str, token: str | None) -> str:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=repo, filename="config.json", token=token)
    model_type = json.loads(Path(config_path).read_text()).get("model_type", "")
    _LOGGER.info("custom_model %s: model_type=%r", repo, model_type)
    return model_type


def _venv_python() -> Path:
    return VENV_DIR / "bin" / "python3"


def _run(cmd: list[str], **kwargs) -> None:
    _LOGGER.info("Running: %s", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def _ensure_venv() -> None:
    if _venv_python().exists():
        return
    _LOGGER.info(
        "Bootstrapping conversion venv at %s (torch-cpu — this downloads a "
        "few hundred MB once and is reused afterwards)", VENV_DIR,
    )
    _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    _run([
        _venv_python(), "-m", "pip", "install", "--no-cache-dir",
        "--extra-index-url", TORCH_CPU_INDEX, *CONVERT_DEPS,
    ])


def ensure_custom_gguf(
    repo: str, quantization: str, models_dir: str | Path, token: str | None
) -> Path:
    """Convert + quantize ``repo`` (HF id) unless already cached."""
    quant = QUANT_ALIASES.get(quantization.lower())
    if quant is None:
        raise ValueError(f"Unknown quantization: {quantization!r}")
    dest = custom_gguf_path(models_dir, repo, quant)
    if dest.exists():
        _LOGGER.info("Custom model cached: %s", dest)
        return dest

    script = pick_convert_script(_detect_model_type(repo, token))
    _ensure_venv()

    dest.parent.mkdir(parents=True, exist_ok=True)
    f32 = dest.with_name(dest.name.replace(f"-{quant}.gguf", "-F32.gguf"))
    env = None
    if token:
        import os

        env = dict(os.environ, HF_TOKEN=token)
    if not f32.exists():
        _LOGGER.info("Converting %s to F32 GGUF (torch-cpu, slow) ...", repo)
        _run([_venv_python(), SCRIPTS_DIR / script, repo, f32], env=env)
    _LOGGER.info("Quantizing to %s ...", quant)
    resolved = resolve_quant(quantization, ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"])
    _run(["transcribe-quantize", f32, dest, "--quant", resolved])
    f32.unlink()
    return dest
