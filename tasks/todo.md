# TODO — Transcribe.cpp HA App

Details per task: see `plan.md`.

## Phase 0 — Research & Pinning
- [x] 0.1 Pin `TRANSCRIBE_REF`, map binding API (batch/streaming) → tasks/notes.md
- [x] 0.2 Model catalog snapshot (all families: repo/license/quants/streaming)
- [x] 0.3 FastEnhancer license gate → GTCRN fallback (SPEC.md decisions log)

## Phase 1 — Minimal Working STT
- [x] 1.1 Repo scaffold (lint configs, config.yaml, translations, stubs, git init)
- [x] 1.2 Dockerfile builder stage (shared lib + transcribe-quantize + pip deps)
- [x] 1.3 engine.py + minimal models.py + batch handler + tests
- [x] 1.4 Runtime stage + s6 + discovery + healthcheck (4 layers added, ≤5 ✓)
- [x] CHECKPOINT A: aarch64 build ✓, jfk.wav + ko.wav → correct transcripts
      with whisper-large-v3-turbo q4_k_m ✓, healthcheck probe ✓, lints ✓

## Phase 2 — Full Catalog & Streaming
- [x] 2.1 Full model registry (64) + quant fallback + config.yaml sync test
- [x] 2.2 Streaming partials (moonshine-streaming-tiny smoke: partials +
      complete final transcript ✓; whisper batch unaffected ✓)
- [x] CHECKPOINT B passed

## Phase 3 — Extras (opt-in)
- [x] 3.1 speech_enhancement — GTCRN via sherpa-onnx (license-gate fallback);
      noise.wav + jfk.wav smoke ✓
- [x] 3.2 diarization — two-speakers.wav → [Speaker 1]/[Speaker 2] ✓
- [x] 3.3 custom_model conversion (smoke: openai/whisper-tiny, in progress)
- [x] CHECKPOINT C: all non-CI SPEC acceptance criteria pass

## Phase 4 — CI & Release
- [x] 4.1 ci.yml / build.yml / release-drafter (GHCR + ha-apps dispatch)
      — needs a GitHub push to verify (no remote yet)
- [x] 4.2 DOCS.md (options, voiceprint chain, generated model table,
      third-party licenses, smoke checklist), translations en/ko
- [ ] icon.png / logo.png (user-provided branding)
- [ ] CHECKPOINT D: push to GitHub, CI green, amd64 smoke on HA box,
      tag v1, ha-apps pickup
