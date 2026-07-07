"""Run convert-parakeet.py in its lowest-memory configuration.

Two independent peaks OOM-kill the converter on small-RAM (8 GB) hosts:

1. NeMo restore_from instantiates the full model before loading weights
   (~2x the RSS of a direct archive read). Upstream's supported switch
   is the ``prefer_direct_load`` variant-profile flag — flipped here on
   every profile.
2. Upstream's ``_load_nemo_archive_directly`` reads the whole
   model_weights.ckpt into a bytes buffer and then torch.load()s a
   BytesIO of it, holding archive bytes AND tensors simultaneously
   (~2x the state dict). The streaming replacement below hands the
   (seekable) tar member file straight to torch.load — same result,
   one copy less.

The shipped script stays verbatim: it is loaded as a module (its
__main__ guard keeps it inert), patched in memory, and main() is
called. Runs under the family venv's python; stdlib only.
"""

import importlib.util
import sys
from pathlib import Path


def _streaming_load_nemo_archive(nemo_path):
    """Drop-in for upstream _load_nemo_archive_directly, minus the
    whole-checkpoint bytes copy."""
    import tarfile

    import torch
    import yaml

    with tarfile.open(nemo_path) as tf:
        names = tf.getnames()

        def _member(suffix: str) -> str:
            for n in names:
                if n.endswith(suffix):
                    return n
            raise FileNotFoundError(
                f"{nemo_path}: missing entry ending with {suffix!r}"
            )

        cfg = yaml.safe_load(tf.extractfile(_member("model_config.yaml")).read())
        sd = torch.load(
            tf.extractfile(_member("model_weights.ckpt")),
            map_location="cpu", weights_only=False,
        )
        sp_proto = None
        for n in names:
            if n.endswith("_tokenizer.model") or n.endswith("/tokenizer.model"):
                sp_proto = tf.extractfile(n).read()
                break
        if sp_proto is None:
            raise FileNotFoundError(f"{nemo_path}: no SPM tokenizer.model entry")

    return cfg, sd, sp_proto


def main() -> None:
    script = sys.argv[1]
    spec = importlib.util.spec_from_file_location("convert_parakeet", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_parakeet"] = module
    spec.loader.exec_module(module)
    for profile in module.VARIANT_PROFILES.values():
        profile["prefer_direct_load"] = True
    if hasattr(module, "_load_nemo_archive_directly"):
        module._load_nemo_archive_directly = _streaming_load_nemo_archive
    raise SystemExit(module.main([script, *sys.argv[2:]]))


if __name__ == "__main__":
    main()
