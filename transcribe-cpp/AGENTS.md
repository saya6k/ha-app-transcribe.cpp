# Agent notes — transcribe-cpp

- Upstream is pinned by `ARG TRANSCRIBE_REF` in the Dockerfile. When bumping:
  regenerate the model registry from the new checkout —
  `python3 scripts/gen_registry.py /path/to/transcribe.cpp` — which rewrites
  `wyoming_transcribe_cpp/registry.json`, the `model: list(...)` line in
  `config.yaml`, and the model table in `DOCS.md`. A unit test fails if they
  drift.
- The Python binding is pure ctypes; it finds our built `libtranscribe.so`
  via the `TRANSCRIBE_LIBRARY` env var (set in the runtime stage). Binding
  attributes `capabilities`, `arch`, `variant`, `backend` are *properties*.
- Streaming vs batch is decided per session by `streaming.decode_mode()`:
  enhancement (whole-utterance GTCRN) and diarization (needs Result segment
  timestamps) both force batch.
- Local builds/tests on the maintainer's Mac use Apple Container CLI
  (`container build` / `container run`), not Docker. bashio reads options
  from the Supervisor API, so local runs bypass s6 with
  `--entrypoint python3 … -m wyoming_transcribe_cpp <flags>`.
- Unit tests: `python3 -m pytest tests/` (pure logic only — no native lib,
  no network). Container smoke recipes are in `../tasks/notes.md`.
- Runtime image layer budget: ≤ 5 layers on top of base-debian (currently 4:
  apt, COPY /usr/local, COPY rootfs, ldconfig+chmod).
