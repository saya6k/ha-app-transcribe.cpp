# TODO — Transcribe.cpp HA App

Details per task: see `plan.md`.

## Phase 0 — Research & Pinning
- [x] 0.1 Pin `TRANSCRIBE_REF`, map binding API (batch/streaming) → tasks/notes.md
- [x] 0.2 Model catalog snapshot (all families: repo/license/quants/streaming)
- [x] 0.3 FastEnhancer license gate → GTCRN fallback (SPEC.md decisions log)

## Phase 1 — Minimal Working STT
- [ ] 1.1 Repo scaffold (lint configs, config.yaml, translations, stubs, git init)
- [ ] 1.2 Dockerfile builder stage (shared lib + transcribe-quantize + pip deps)
- [ ] 1.3 engine.py + minimal models.py + batch handler + tests
- [ ] 1.4 Runtime stage + s6 + discovery + healthcheck (≤5 layers)
- [ ] CHECKPOINT A: local build + WAV→transcript smoke + lints + commit + review

## Phase 2 — Full Catalog & Streaming
- [ ] 2.1 Full model registry + quant selection + config.yaml list sync test
- [ ] 2.2 Streaming partials (nemotron-asr-c pattern, supports_streaming gate)
- [ ] CHECKPOINT B: partials with streaming model, batch unaffected + review

## Phase 3 — Extras (opt-in)
- [ ] 3.1 speech_enhancement (FastEnhancer or GTCRN per 0.3)
- [ ] 3.2 diarization ([Speaker N] tags, voiceprint chain in DOCS)
- [ ] 3.3 custom_model conversion (torch-cpu /data venv, cache)
- [ ] CHECKPOINT C: SPEC acceptance criteria (non-CI) all pass + review

## Phase 4 — CI & Release
- [ ] 4.1 ci.yml / build.yml / release-drafter (GHCR + ha-apps dispatch)
- [ ] 4.2 DOCS.md, .README.j2, translations, license table, smoke checklist
- [ ] CHECKPOINT D: amd64 smoke on HA box, tag v1, ha-apps pickup
