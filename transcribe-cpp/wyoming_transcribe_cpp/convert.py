"""Custom fine-tuned model conversion: HF checkpoint -> GGUF -> quantized.

torch is never in the image. Each converter family gets its own lazy venv
under /data/convert-venv/<family> because upstream's per-family dep pins
conflict (e.g. whisper wants transformers==5.6.1, voxtral ==4.57.6). The
dep list is parsed from the upstream ``scripts/envs/<family>/pyproject.toml``
shipped in the image — never duplicated here. torch always resolves from
the CPU wheel index; GPU wheels are out of scope.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import TextIO

from .const import QUANT_ALIASES
from .models import resolve_quant
from .weighthash import identity_from_hub, plan_action, write_sidecar

_LOGGER = logging.getLogger(__name__)

SHARE_DIR = Path("/usr/local/share/transcribe-cpp")
SCRIPTS_DIR = SHARE_DIR / "scripts"
ENVS_DIR = SHARE_DIR / "envs"
VENV_ROOT = Path("/data/convert-venv")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CONVERT_SOCK = Path("/run/transcribe-cpp/convert.sock")


class ConversionUnsupported(Exception):
    pass


class ConversionFailed(Exception):
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


_GIT_DEP = re.compile(
    r"git\+https://github\.com/(?P<org>[^/]+)/(?P<repo>[^/@]+?)"
    r"(?:\.git)?@(?P<ref>[A-Za-z0-9._-]+)$"
)


def _degit(dep: str) -> str:
    """Rewrite a github git+ spec to the equivalent commit tarball.

    The image ships no git; pip installs pinned github sources from
    /archive/<ref>.tar.gz instead — identical content, no VCS needed.
    """
    if "git+" not in dep:
        return dep
    m = _GIT_DEP.search(dep)
    if m is None:
        raise ConversionUnsupported(
            f"cannot install VCS dependency without git in the image: {dep!r}"
        )
    head = dep.split("@", 1)[0].strip()
    return (
        f"{head} @ https://github.com/{m['org']}/{m['repo']}"
        f"/archive/{m['ref']}.tar.gz"
    )


def pip_commands(python: str | Path, deps: list[str]) -> list[list[str]]:
    """torch first from the CPU-only index, then the family deps from PyPI.

    Installing torch (and torchaudio when the family needs it) up front
    from the CPU index keeps pip from ever resolving the CUDA-bundled
    PyPI linux wheels; the later resolve sees torch already satisfied.
    """
    deps = [_degit(d) for d in deps]
    torch_pkgs = ["torch"]
    if any(d.split()[0].startswith("torchaudio") for d in deps):
        torch_pkgs.append("torchaudio")
    base = [str(python), "-m", "pip", "install", "--no-cache-dir"]
    return [
        [*base, "--index-url", CPU_INDEX, *torch_pkgs],
        [*base, "--extra-index-url", CPU_INDEX, *deps],
    ]


def _run(cmd: list[str], env: dict | None = None) -> None:
    """Run a tool, streaming its output through the app log line by line.

    The worker's stdout is the client's socket, so subprocess output must
    never hit it raw — everything goes through logging.
    """
    _LOGGER.info("Running: %s", " ".join(map(str, cmd)))
    proc = subprocess.Popen(
        [str(c) for c in cmd], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _LOGGER.info("| %s", line.rstrip())
    _check_returncode(proc.wait(), cmd)


def _check_returncode(returncode: int, cmd: list) -> None:
    if returncode == 0:
        return
    if returncode < 0:
        import signal as _signal

        name = _signal.Signals(-returncode).name
        hint = (
            " — the kernel OOM-killer usually sends SIGKILL; NeMo-family "
            "conversion needs several GB of free RAM"
            if name == "SIGKILL" else ""
        )
        raise ConversionFailed(
            f"converter died with {name}{hint} (cmd: {cmd[1]})"
        )
    raise subprocess.CalledProcessError(returncode, cmd)


# Debian trixie's Python 3.13 still has wheel gaps in the NeMo dep tree
# (editdistance ships no cp313 wheels at all) and the image carries no C
# toolchain to build sdists — NeMo-family venvs therefore run on a
# managed CPython 3.12 (python-build-standalone, sha256-pinned),
# downloaded once into /data.
PY312_FAMILIES = frozenset({"parakeet", "canary", "canary_qwen"})
PY312_DIR = VENV_ROOT / ".cpython312"
_PBS_RELEASE = "20260623"
_PBS_VERSION = "3.12.13"
_PBS_SHA256 = {
    "aarch64": "b85154b9c7ca9de3f85f2c9f032d503151db16ef198de86b885fc61890c075ed",
    "x86_64": "10a452caac7041357805f0c19a60576df53f1ab06d1abfc9200f1f0157cb3bd1",
}


def _machine() -> str:
    import platform

    return platform.machine()


def _download(url: str, dest: Path) -> None:
    import urllib.request

    urllib.request.urlretrieve(url, dest)  # noqa: S310 — pinned https URL


def ensure_python312() -> Path:
    """Fetch the managed CPython 3.12 into /data on first use."""
    python = PY312_DIR / "python" / "bin" / "python3.12"
    if python.exists():
        return python
    machine = _machine()
    sha = _PBS_SHA256.get(machine)
    if sha is None:
        raise ConversionUnsupported(
            f"no pinned CPython 3.12 build for architecture {machine!r}"
        )
    url = (
        "https://github.com/astral-sh/python-build-standalone/releases/"
        f"download/{_PBS_RELEASE}/cpython-{_PBS_VERSION}%2B{_PBS_RELEASE}-"
        f"{machine}-unknown-linux-gnu-install_only_stripped.tar.gz"
    )
    _LOGGER.info("Downloading managed CPython %s (%s) ...", _PBS_VERSION, machine)
    PY312_DIR.mkdir(parents=True, exist_ok=True)
    tarball = PY312_DIR / "cpython.tar.gz"
    _download(url, tarball)
    import hashlib
    import tarfile

    actual = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if actual != sha:
        tarball.unlink()
        raise ConversionUnsupported(
            f"managed CPython download failed its checksum ({actual})"
        )
    with tarfile.open(tarball) as tf:
        tf.extractall(PY312_DIR, filter="data")
    tarball.unlink()
    return python


def _venv_ready_marker(family: str) -> Path:
    return venv_dir(family) / ".ready"


def ensure_family_venv(family: str) -> Path:
    """Create /data/convert-venv/<family> on first use; reuse afterwards.

    The .ready marker lands only after every install step succeeded — a
    crash mid-bootstrap leaves no half-venv to be silently reused.
    """
    python = _venv_python(family)
    if _venv_ready_marker(family).exists():
        return python
    if venv_dir(family).exists():
        _LOGGER.warning(
            "Rebuilding incomplete %s venv at %s", family, venv_dir(family)
        )
        shutil.rmtree(venv_dir(family))
    _LOGGER.info(
        "Bootstrapping %s conversion venv at %s (torch-cpu — downloads a "
        "few hundred MB once, more for NeMo families, and is reused "
        "afterwards)", family, venv_dir(family),
    )
    base = (
        ensure_python312() if family in PY312_FAMILIES
        else Path(sys.executable)
    )
    _run([str(base), "-m", "venv", str(venv_dir(family))])
    for cmd in pip_commands(python, env_deps(family)):
        _run(cmd)
    venv_dir(family).mkdir(parents=True, exist_ok=True)
    _venv_ready_marker(family).touch()
    return python


# granite_nar/medasr take --outdir instead of a positional output path;
# their product is moved to the REF path afterwards.
OUTDIR_FAMILIES = frozenset({"granite_nar", "medasr"})
# Families whose converter accepts --revision (pins the checkout to the
# revision recorded in the weight-identity sidecar). The NeMo trio and
# cohere/parakeet resolve their own downloads and have no such flag.
_REVISION_FAMILIES = frozenset({
    "whisper", "moonshine", "moonshine_streaming", "qwen3_asr",
    "sensevoice", "funasr_nano", "granite", "granite_nar", "medasr",
    "voxtral", "voxtral_realtime",
})
# How the base variant reaches each converter — they differ:
#   --variant flag                      (most transformers-side families)
#   out_path.parent.name                (parakeet: slug = output dir name)
#   slug_from_repo_id(--repo-id)        (canary, canary_qwen)
_VARIANT_FLAG_FAMILIES = frozenset({
    "whisper", "moonshine", "moonshine_streaming", "qwen3_asr",
    "sensevoice", "funasr_nano", "granite", "voxtral", "voxtral_realtime",
})
SLUGDIR_FAMILIES = frozenset({"parakeet"})
_REPOID_VARIANT_FAMILIES = frozenset({"canary", "canary_qwen"})


def _outdir_for(out_path: Path) -> Path:
    return out_path.parent / (out_path.stem + ".outdir")


def _pick_nemo_file(files: list[str]) -> str | None:
    return next((f for f in sorted(files) if f.endswith(".nemo")), None)


def converter_cmd(
    family: str,
    repo: str,
    out_path: Path,
    revision: str | None = None,
    variant: str | None = None,
    model_spec: str | None = None,
) -> list[str]:
    """Upstream CLI: <script> <repo> [<out.gguf>] --repo-id <repo> [...]"""
    from .detect import CONVERT_SCRIPTS

    if family == "gigaam":
        # gigaam's converter ignores the repo and downloads official
        # weights keyed by --variant-key; fine-tune import is not a thing
        # upstream supports. Curated gigaam models come prebuilt instead.
        raise ConversionUnsupported(
            "gigaam checkpoints cannot be imported as custom_model "
            "(upstream converter only fetches official GigaAM weights); "
            "pick a gigaam model from the curated catalog instead."
        )
    if family in SLUGDIR_FAMILIES | _REPOID_VARIANT_FAMILIES and not variant:
        raise ConversionUnsupported(
            f"{family} conversion needs a recognizable base variant to "
            "dispatch the converter. Add a base_model tag to the HF repo "
            "(e.g. base_model:nvidia/parakeet-tdt-0.6b-v2) or include the "
            "base catalog slug in the repo name."
        )
    cmd = [
        str(_venv_python(family)),
        str(SCRIPTS_DIR / CONVERT_SCRIPTS[family]),
        model_spec or repo,
    ]
    if family in OUTDIR_FAMILIES:
        cmd += ["--repo-id", repo, "--outdir", str(_outdir_for(out_path))]
    elif family in SLUGDIR_FAMILIES:
        cmd += [
            str(out_path.parent / variant / out_path.name), "--repo-id", repo,
        ]
    elif family in _REPOID_VARIANT_FAMILIES:
        # slug_from_repo_id(--repo-id) picks the variant profile; the
        # bare catalog slug passes through it unchanged.
        cmd += [str(out_path), "--repo-id", variant]
    else:
        cmd += [str(out_path), "--repo-id", repo]
    if revision and family in _REVISION_FAMILIES:
        cmd += ["--revision", revision]
    if variant and family in _VARIANT_FLAG_FAMILIES:
        cmd += ["--variant", variant]
    return cmd


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

    from .detect import derive_variant, detect_family, probe_from_hub

    probe = probe_from_hub(repo, token)
    family = detect_family(probe)
    variant = derive_variant(family, probe)
    if variant:
        _LOGGER.info("Using base variant %r for %s", variant, repo)
    ensure_family_venv(family)

    # NeMo's from_pretrained only recognizes conventionally-named .nemo
    # files and caches them in its own throwaway directory (re-downloaded
    # on every retry). Fetch the archive through the standard hub cache
    # ourselves and hand the converter a local path instead.
    model_spec = None
    if family in SLUGDIR_FAMILIES | _REPOID_VARIANT_FAMILIES:
        nemo_file = _pick_nemo_file(probe.files)
        if nemo_file:
            from huggingface_hub import hf_hub_download

            _LOGGER.info("Fetching %s from %s ...", nemo_file, repo)
            model_spec = hf_hub_download(
                repo, nemo_file,
                revision=identity.revision or None, token=token,
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = dest.with_name(dest.name.replace(f"-{quant}.gguf", "-REF.gguf"))
    env = dict(os.environ, HF_TOKEN=token) if token else None
    if not ref.exists():
        _LOGGER.info(
            "Converting %s (family %s) to reference GGUF (torch-cpu, "
            "slow) ...", repo, family,
        )
        if family in SLUGDIR_FAMILIES and variant:
            (ref.parent / variant).mkdir(parents=True, exist_ok=True)
        _run(
            converter_cmd(
                family, repo, ref, identity.revision, variant, model_spec
            ),
            env=env,
        )
        if family in OUTDIR_FAMILIES:
            outdir = _outdir_for(ref)
            produced = sorted(outdir.glob("**/*.gguf"))
            if len(produced) != 1:
                raise ConversionFailed(
                    f"expected exactly one GGUF under {outdir}, "
                    f"found {len(produced)}"
                )
            produced[0].replace(ref)
            shutil.rmtree(outdir)
        elif family in SLUGDIR_FAMILIES and variant:
            (ref.parent / variant / ref.name).replace(ref)
            shutil.rmtree(ref.parent / variant)
    _LOGGER.info("Quantizing to %s ...", quant)
    resolved = resolve_quant(
        quantization, ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    )
    _run(["transcribe-quantize", ref, dest, "--quant", resolved])
    ref.unlink()
    write_sidecar(dest, identity, family=family, quant=quant)
    return dest


# ---- socket client (runs in the server process as the transcribe user) ----


def _client_session(reader: TextIO, writer: TextIO, request: dict) -> Path:
    """Send one request, relay worker log events, return the GGUF path."""
    writer.write(json.dumps(request) + "\n")
    writer.flush()
    for line in reader:
        event = json.loads(line)
        if event.get("event") == "log":
            _LOGGER.log(
                logging.getLevelNamesMapping().get(
                    event.get("level", "info").upper(), logging.INFO
                ),
                "[convert] %s", event.get("message", ""),
            )
        elif event.get("event") == "result":
            if event.get("ok"):
                return Path(event["gguf"])
            raise ConversionFailed(event.get("error", "unknown error"))
    raise ConversionFailed("conversion worker closed the connection early")


def request_conversion(
    repo: str,
    quantization: str,
    models_dir: str | Path,
    token: str | None,
    socket_path: Path = CONVERT_SOCK,
    connect_timeout: float = 30.0,
) -> Path:
    """Ask the convert-worker (unix socket, unprivileged) for a GGUF."""
    deadline = time.monotonic() + connect_timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect(str(socket_path))
            break
        except OSError:
            if time.monotonic() >= deadline:
                sock.close()
                raise
            time.sleep(0.5)
    try:
        with sock.makefile("r") as reader, sock.makefile("w") as writer:
            return _client_session(reader, writer, {
                "repo": repo,
                "quantization": quantization,
                "models_dir": str(models_dir),
                "hf_token": token or "",
            })
    finally:
        sock.close()
