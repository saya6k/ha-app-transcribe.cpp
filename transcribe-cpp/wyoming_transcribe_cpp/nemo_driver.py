"""Run convert-parakeet.py with upstream's prefer_direct_load forced.

NeMo restore_from instantiates the full model before loading weights —
peak RSS is roughly twice the direct-archive read, and on small-RAM
hosts (8 GB) the kernel OOM-killer takes the process before the
converter's own restore_from-failure fallback can engage. The variant
profile flag ``prefer_direct_load`` is upstream's supported switch for
exactly this load path; this driver loads the shipped script as a
module (its __main__ guard keeps it inert), flips the flag on every
profile, and calls its main() — the script file itself stays verbatim.

Runs under the family venv's python; stdlib only.
"""

import importlib.util
import sys


def main() -> None:
    script = sys.argv[1]
    spec = importlib.util.spec_from_file_location("convert_parakeet", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for profile in module.VARIANT_PROFILES.values():
        profile["prefer_direct_load"] = True
    raise SystemExit(module.main([script, *sys.argv[2:]]))


if __name__ == "__main__":
    main()
