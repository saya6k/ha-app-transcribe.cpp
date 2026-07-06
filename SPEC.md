# SPEC — Transcribe.cpp Home Assistant App

Status: READY FOR REVIEW
Date: 2026-07-06

## 1. Objective

Package [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT,
C++ GGUF ASR engine, 16 model families / 60+ variants) as a Home Assistant app
(formerly add-on) named **Transcribe (cpp)**, following the established
`ha-app-*` conventions. Structural template: `ha-app-nemo-asr-cpp`
(Dockerfile/CI shape) + `ha-app-nemotron-asr-c` (streaming handler +
enhancer bridge).

Target user: Home Assistant users running Assist voice pipelines who want
fast local CPU speech-to-text with the full transcribe.cpp model catalog,
optional noise suppression, optional speaker labels, and the ability to
bring their own fine-tuned checkpoint (converted + quantized to GGUF
on-device).

### Core features

1. **Wyoming STT server** on port `10370/tcp`, using transcribe.cpp built
   from a pinned commit SHA as a shared library with its official Python
   binding.
   - **Streaming included in v1**: follow the `wyoming_nemotron_asr_c`
     handler pattern — `TranscriptStart` → `TranscriptChunk` deltas →
     `TranscriptStop` + final `Transcript`.
   - Not every transcribe.cpp family decodes incrementally. Streaming
     partials are emitted for streaming-capable families (Parakeet,
     Qwen3-ASR streaming variants, …); batch-only families (e.g. Whisper)
     fall back to final-`Transcript`-only. The engine wrapper exposes
     `supports_streaming` per model and the handler branches on it.
2. **Supervisor discovery** (`discovery: [wyoming]` + s6 `discovery`
   one-shot), so HA auto-configures the Wyoming integration.
3. **Audio enhancement** (`speech_enhancement`, default off):
   **FastEnhancer** streaming pre-filter, reusing the proven
   `libfastenhancer.so` ctypes bridge + Dockerfile build step from
   `ha-app-nemotron-asr-c` (256-sample frames, 95 KB weights — works
   inside the streaming path, unlike offline denoisers).
   - License gate before first release: confirm upstream
     `aask1357/fastenhancer` and `kdrkdrkdr/fastenhancer.c.wasm` are
     MIT/BSD/Apache; the wasm port repo currently shows no LICENSE file.
     If it fails the gate, fall back to GTCRN via sherpa-onnx (MIT model,
     Apache-2.0 code) applied at utterance end (final transcript only).
4. **Diarization** (`diarization`, default off): sherpa-onnx offline
   speaker diarization (pyannote `segmentation-3.0` ONNX, MIT +
   3D-Speaker CAM++ embedding, Apache-2.0). Runs on the buffered
   utterance at `AudioStop`; speaker labels are rendered as inline tags
   in the **final** `Transcript` only (`[Speaker 1] … [Speaker 2] …`) —
   streamed partials stay untagged.
   - **Review result (voiceprint delegation):** for Assist utterances
     (single speaker per request) built-in diarization adds little; the
     recommended setup is chaining `ha-app-voiceprint` in front
     (`HA → voiceprint:10350 → this app:10370`), which yields *named*
     speakers from enrollment instead of anonymous "Speaker N". DOCS.md
     documents this chain as the preferred speaker-attribution path.
     Built-in diarization is for long-form / multi-speaker audio where a
     per-utterance verifier cannot help. Diarization models are
     downloaded at runtime only when enabled — never baked into the
     image; `sherpa-onnx` (pip) is the only added dependency.
5. **Model management via config.yaml only** — no exec into container:
   - `model`: curated dropdown generated from a registry
     (`models.py`) that covers **every** handy-computer GGUF family
     supported by the pinned transcribe.cpp commit (Whisper, Parakeet,
     Canary, Qwen3-ASR, Moonshine, and the remaining families). The
     registry records per model: HF repo, license, languages,
     streaming-capable flag. Default: **Whisper large-v3-turbo**.
   - `custom_model`: Hugging Face repo ID of a fine-tuned checkpoint.
     When set, it overrides `model` and triggers the convert pipeline.
   - `quantization`: `q4_k_m|q5_k_m|q6_k|q8_0|f16`, default `q4_k_m`.
     Applies to curated models (pick matching prebuilt GGUF from HF) and
     to converted custom models (run `transcribe-quantize`).
6. **On-device GGUF conversion + quantization** (torch CPU, slow-OK):
   - Uses transcribe.cpp's own `convert-*.py` scripts + the
     `transcribe-quantize` tool from the same pinned commit.
   - torch-cpu is **not** baked into the image. On first conversion the
     app pip-installs torch-cpu + conversion deps into a venv under
     `/data/convert-venv` (persistent, reused). Keeps the image small
     and layers minimal; conversion is a rare, offline-tolerant
     operation whose progress is logged to the app log.
   - Converted GGUFs cached under `/data/models/<repo>__<quant>.gguf`
     (excluded from backups via `backup_exclude`). Re-conversion skipped
     when the cache entry exists.

### Acceptance criteria

- Image builds for amd64 and aarch64 (HA builder in CI; Apple Container
  CLI locally — this Mac has no Docker).
- App starts, announces via discovery, is selectable as STT in an Assist
  pipeline; a spoken test phrase returns a transcript.
- A streaming-capable model emits `TranscriptChunk` deltas before
  `AudioStop`; Whisper (batch) still returns a correct final transcript.
- With `speech_enhancement: true`, noisy WAV transcribes at least as
  well as without (manual smoke check, not a gate).
- With `diarization: true`, a 2-speaker WAV returns `[Speaker N]` tags
  in the final transcript.
- Setting `custom_model` to a fine-tuned Whisper HF repo produces a
  quantized GGUF in `/data/models` and the app serves it.
- All licenses in the image/runtime downloads are MIT/BSD/Apache-2.0;
  third-party notices listed in DOCS.md (FastEnhancer license confirmed
  or fallback taken).
- Runtime image stays in the nemo-asr-cpp layer budget: ≤ 5 layers on
  top of `base-debian` (apt RUN, `COPY --from=builder /usr/local`,
  `COPY rootfs /`, ldconfig/chmod RUN, HEALTHCHECK).

## 2. Commands

```sh
# Lint (same toolchain as sibling repos)
hadolint transcribe-cpp/Dockerfile
shellcheck transcribe-cpp/rootfs/etc/s6-overlay/s6-rc.d/*/run
yamllint .
markdownlint '**/*.md'

# Local image build — this Mac has no Docker; use Apple Container CLI
container build --arg BUILD_FROM=ghcr.io/home-assistant/base-debian:trixie \
  -t local/transcribe-cpp transcribe-cpp/

# Python unit tests
cd transcribe-cpp && python3 -m pytest tests/

# Local run (Wyoming on 10370)
container run --rm -p 10370:10370 -v "$PWD/test-data:/data" local/transcribe-cpp
```

CI (GitHub Actions) keeps using the HA builder / docker as in sibling repos.

## 3. Project structure

Mirror the sibling repos:

```text
ha-app-transcribe.cpp/
├── README.md
├── .hadolint.yaml / .shellcheckrc / .yamllint / .markdownlint.yaml
├── .github/workflows/{ci.yml,build.yml,release-drafter.yml}
│     # build.yml → ghcr.io/saya6k/app-transcribe-cpp, per-arch tags,
│     # repository-dispatch "app-released" to ha-apps (store repo)
└── transcribe-cpp/                  # app slug: transcribe-cpp
    ├── config.yaml                  # version: dev, discovery: wyoming,
    │                                # ports: 10370/tcp, backup_exclude
    ├── Dockerfile                   # 2-stage; ARG TRANSCRIBE_REF +
    │                                # ARG FASTENHANCER_REF (pinned SHAs);
    │                                # builder compiles libtranscribe(+ggml)
    │                                # and libfastenhancer, pip deps → /usr/local
    ├── DOCS.md / .README.j2 / AGENTS.md
    ├── apparmor.txt
    ├── translations/{en,ko}.yaml
    ├── pyproject.toml
    ├── rootfs/etc/s6-overlay/s6-rc.d/
    │     ├── transcribe-cpp/{run,finish,type,…}
    │     └── discovery/{run,type,up,…}
    ├── wyoming_transcribe_cpp/
    │     ├── __main__.py            # arg parsing, model resolve, server boot
    │     ├── handler.py             # streaming AsyncEventHandler
    │     │                          #   (nemotron-asr-c pattern)
    │     ├── engine.py              # transcribe.cpp binding wrapper,
    │     │                          #   supports_streaming per model
    │     ├── enhancer.py            # FastEnhancer ctypes bridge (ported)
    │     ├── diarize.py             # sherpa-onnx diarization → inline tags
    │     ├── convert.py             # /data venv bootstrap + HF→GGUF→quant
    │     ├── models.py              # full model registry + cache paths
    │     └── const.py
    └── tests/
```

### config.yaml options/schema (draft)

```yaml
options:
  model: Whisper large-v3-turbo
  quantization: q4_k_m
  custom_model: ''
  language: auto
  speech_enhancement: false
  diarization: false
  hf_token: ''
schema:
  model: list(<all curated models from models.py registry>)
  quantization: list(q4_k_m|q5_k_m|q6_k|q8_0|f16)
  custom_model: str?
  language: str
  speech_enhancement: bool
  diarization: bool
  max_speakers: int(1,8)?      # diarization hint
  hf_token: password?
  debug_logging: bool?
```

## 4. Code style

- Match sibling repos: plain Python 3 (Debian trixie system python);
  runtime deps limited to `wyoming`, `huggingface_hub`, `numpy`,
  `sherpa-onnx`; comments explain constraints, not narration.
- Dockerfile: `SHELL bash -o pipefail`, pinned upstream commit SHAs via
  `ARG`, `GGML_NATIVE=OFF` for portable kernels, `strip
  --strip-unneeded` on built `.so`, single `COPY --from=builder
  /usr/local /usr/local` handoff.
- s6 scripts: bashio, shellcheck-clean.
- Conventional-Commit titles (release-drafter labeling depends on it).

## 5. Testing strategy

1. **Unit (pytest, no model download):** delta computation for
   `TranscriptChunk` streaming; diarization tag rendering from segment
   lists; convert-cache path/skip logic; option parsing; model registry
   integrity (every entry has repo/license/streaming flag).
2. **CI lint gate:** hadolint/shellcheck/yamllint/markdownlint (ci.yml
   copied from siblings).
3. **Build gate:** per-arch image build in build.yml (HA builder
   action); local pre-push smoke via Apple Container CLI.
4. **Manual smoke (release checklist in DOCS/PR template):** Wyoming
   `describe` healthcheck; Assist end-to-end phrase with a streaming
   model (partials visible) and with Whisper (batch); 2-speaker WAV with
   diarization; one fine-tune conversion run on amd64.

## 6. Boundaries

**Always**
- Keep every runtime dependency MIT/BSD/Apache-2.0; record each model +
  license pair in DOCS.md/models.py before adding it to the dropdown.
- Download models/weights at runtime into `/data` (image stays
  weight-free).
- Preserve the sibling-repo file conventions and CI shape.

**Ask first**
- Adding a third-party component with any other license (incl. gated HF
  models such as pyannote.audio's official pipelines).
- Baking torch or any model weights into the image.
- Changing the Wyoming port, slug, or GHCR image name once released.
- GPU (Vulkan/CUDA) variants.

**Never**
- GPL/AGPL/non-commercial-licensed code or models.
- Extra privileges/maps not needed by a feature.
- Publishing mutable version tags consumed by the store; ha-apps pins
  exact versions via repository-dispatch.

## Decisions log

- 2026-07-06 — Default model: Whisper large-v3-turbo @ `q4_k_m`.
- 2026-07-06 — Streaming partials in v1, per `ha-app-nemotron-asr-c`
  pattern; batch fallback for non-streaming families.
- 2026-07-06 — Curated list must cover *all* transcribe.cpp-supported
  handy-computer model families, not a subset.
- 2026-07-06 — Speaker attribution for Assist: delegate to
  `ha-app-voiceprint` chaining (documented); built-in sherpa-onnx
  diarization kept as opt-in for multi-speaker audio.
- 2026-07-06 — Enhancement: FastEnhancer bridge reused from
  nemotron-asr-c, pending license confirmation (fallback: GTCRN via
  sherpa-onnx).
- 2026-07-06 — Local builds use Apple Container CLI (no Docker on this
  Mac); CI uses HA builder.
- 2026-07-06 — FastEnhancer license gate FAILED (C port has no LICENSE);
  enhancement uses the pre-approved GTCRN/sherpa-onnx fallback, applied at
  utterance end (partials disabled while enhancement is on).
- 2026-07-06 — Model dropdown: all 65 catalog entries except cc-by-nc-4.0
  `canary-1b`; 'other'-licensed models included with license shown in DOCS
  (weights are user-downloaded at runtime, never redistributed).
- 2026-07-06 — `TRANSCRIBE_REF=d89ecb75062e8457681c563994675dc60e31db80`;
  binding is pure-ctypes, loaded via `TRANSCRIBE_LIBRARY` env pointing at
  our built `libtranscribe.so`; streaming detected at runtime via
  `Model.capabilities().supports_streaming`.
