"""Custom fine-tuned model conversion: HF checkpoint -> GGUF -> quantized.

torch is never in the image. Each converter family gets its own lazy venv
under /data/convert-venv/<family> because upstream's per-family dep pins
conflict (e.g. whisper wants transformers==5.6.1, voxtral ==4.57.6). The
dep list is parsed from the upstream ``scripts/envs/<family>/pyproject.toml``
shipped in the image — never duplicated here. torch always resolves from
the CPU wheel index; GPU wheels are out of scope.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from .const import QUANT_ALIASES
from .models import resolve_quant
from .weighthash import identity_from_hub, plan_action, write_sidecar

_LOGGER = logging.getLogger(__name__)

SHARE_DIR = Path("/usr/local/share/transcribe-cpp")
SCRIPTS_DIR = SHARE_DIR / "scripts"
ENVS_DIR = SHARE_DIR / "envs"
VENV_ROOT = Path("/data/convert-venv")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class ConversionUnsupported(Exception):
    pass


def custom_gguf_path(models_dir: str | Path, repo: str, quant: str) -> Path:
    """Cache location: <models_dir>/custom/<owner>__<name>-<QUANT>.gguf."""
    return Path(models_dir) / "custom" / f"{repo.replace('/', '__')}-{quant}.gguf"


def parse_env_deps(pyproject_text: str) -> list[str]:
    return tomllib.loads(pyproject_text)["project"]["dependencies"]


def env_deps(family: str) -> list[str]:
    return parse_env_deps((ENVS_DIR / family / "pyproject.toml").read_text())


def venv_dir(family: str) -> Path:
    return VENV_ROOT / family


def _venv_python(family: str) -> Path:
    return venv_dir(family) / "bin" / "python3"


def pip_commands(python: str | Path, deps: list[str]) -> list[list[str]]:
    """torch first from the CPU-only index, then the family deps from PyPI.

    Installing torch (and torchaudio when the family needs it) up front
    from the CPU index keeps pip from ever resolving the CUDA-bundled
    PyPI linux wheels; the later resolve sees torch already satisfied.
    """
    torch_pkgs = ["torch"]
    if any(d.split()[0].startswith("torchaudio") for d in deps):
        torch_pkgs.append("torchaudio")
    base = [str(python), "-m", "pip", "install", "--no-cache-dir"]
    return [
        [*base, "--index-url", CPU_INDEX, *torch_pkgs],
        [*base, "--extra-index-url", CPU_INDEX, *deps],
    ]


def _run(cmd: list[str], **kwargs) -> None:
    _LOGGER.info("Running: %s", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def ensure_family_venv(family: str) -> Path:
    """Create /data/convert-venv/<family> on first use; reuse afterwards."""
    python = _venv_python(family)
    if python.exists():
        return python
    _LOGGER.info(
        "Bootstrapping %s conversion venv at %s (torch-cpu — downloads a "
        "few hundred MB once, more for NeMo families, and is reused "
        "afterwards)", family, venv_dir(family),
    )
    _run([sys.executable, "-m", "venv", str(venv_dir(family))])
    for cmd in pip_commands(python, env_deps(family)):
        _run(cmd)
    return python


def converter_cmd(
    family: str, repo: str, out_path: Path
) -> list[str]:
    """Uniform upstream CLI: <script> <repo> <out.gguf> --repo-id <repo>."""
    from .detect import CONVERT_SCRIPTS

    script = SCRIPTS_DIR / CONVERT_SCRIPTS[family]
    if family == "gigaam":
        # gigaam's converter ignores the repo and downloads official
        # weights keyed by --variant-key; fine-tune import is not a thing
        # upstream supports. Curated gigaam models come prebuilt instead.
        raise ConversionUnsupported(
            "gigaam checkpoints cannot be imported as custom_model "
            "(upstream converter only fetches official GigaAM weights); "
            "pick a gigaam model from the curated catalog instead."
        )
    return [
        str(_venv_python(family)), str(script), repo, str(out_path),
        "--repo-id", repo,
    ]


def ensure_custom_gguf(
    repo: str, quantization: str, models_dir: str | Path, token: str | None
) -> Path:
    """Convert + quantize ``repo`` (HF id) unless the cache is current."""
    quant = QUANT_ALIASES.get(quantization.lower())
    if quant is None:
        raise ValueError(f"Unknown quantization: {quantization!r}")
    dest = custom_gguf_path(models_dir, repo, quant)

    try:
        identity = identity_from_hub(repo, token)
    except Exception as err:
        if dest.exists():
            _LOGGER.warning(
                "HF Hub unreachable (%s) — serving cached %s unverified",
                err, dest,
            )
            return dest
        raise
    action, reason = plan_action(dest, identity)
    _LOGGER.info("custom_model %s: %s", repo, reason)
    if action == "serve":
        return dest

    from .detect import detect_family, probe_from_hub

    family = detect_family(probe_from_hub(repo, token))
    python = ensure_family_venv(family)

    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = dest.with_name(dest.name.replace(f"-{quant}.gguf", "-REF.gguf"))
    env = dict(os.environ, HF_TOKEN=token) if token else None
    if not ref.exists():
        _LOGGER.info(
            "Converting %s (family %s) to reference GGUF (torch-cpu, "
            "slow) ...", repo, family,
        )
        _run(converter_cmd(family, repo, ref), env=env)
    _LOGGER.info("Quantizing to %s ...", quant)
    resolved = resolve_quant(
        quantization, ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    )
    _run(["transcribe-quantize", ref, dest, "--quant", resolved])
    ref.unlink()
    write_sidecar(dest, identity, family=family, quant=quant)
    return dest
