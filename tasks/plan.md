# Implementation Plan: Transcribe.cpp Home Assistant App

Source spec: `../SPEC.md` (READY FOR REVIEW, 2026-07-06)

## Overview

Build the `transcribe-cpp` HA app in vertical slices: first a minimal
working Wyoming STT path (default Whisper model, batch decode) that builds
and runs locally via Apple Container CLI, then layer on the full model
catalog, streaming partials, enhancement, diarization, custom-model
conversion, and finally CI/release wiring. Every phase ends with a bootable
image.

## Architecture Decisions (from SPEC)

- Template repos: `ha-app-nemo-asr-cpp` (Dockerfile/CI/s6 shape),
  `ha-app-nemotron-asr-c` (streaming handler + FastEnhancer bridge).
- 2-stage Dockerfile, pinned `TRANSCRIBE_REF` SHA, runtime ≤ 5 layers.
- transcribe.cpp official Python binding over a shared lib;
  `transcribe-quantize` built in the same builder stage.
- torch-cpu never in the image — lazy venv under `/data/convert-venv`.
- Licenses: MIT/BSD/Apache-2.0 only; FastEnhancer needs a license gate.

## Dependency Graph

```
P0 research (SHA pin, binding API, model catalog, FastEnhancer license)
  │
  ├── T1 repo scaffold ──────────────┐
  ├── T2 Dockerfile builder stage ───┤
  │       │                          │
  │       └── T3 engine.py (batch)   │
  │               │                  │
  │      T4 models.py (minimal) ─────┤
  │               │                  │
  │       T5 handler.py (batch) + s6/discovery + runtime stage
  │               │
  │        [CHECKPOINT A: bootable STT image]
  │               │
  │      T6 full catalog + quant selection
  │      T7 streaming partials
  │        [CHECKPOINT B]
  │      T8 enhancement (license gate → FastEnhancer | GTCRN)
  │      T9 diarization
  │      T10 conversion pipeline (custom_model)
  │        [CHECKPOINT C]
  └───── T11 CI workflows + release docs
         [CHECKPOINT D: release candidate]
```

---

## Phase 0 — Research & Pinning (read-only, no image yet)

### Task 0.1: Pin upstream and map the binding API

**Description:** Clone transcribe.cpp at HEAD, record the commit SHA to use
as `TRANSCRIBE_REF`. Identify: CMake flags for a shared-lib build, the
Python binding's import path and whether it loads a system `.so` or builds
its own, the batch + streaming decode API surface, and how
`transcribe-quantize` and `scripts/convert-*.py` are invoked.

**Acceptance criteria:**
- [ ] `TRANSCRIBE_REF` SHA chosen and written into `tasks/notes.md`
- [ ] Binding call sequence for batch decode documented (load → transcribe)
- [ ] Streaming API presence per model family documented (or confirmed absent)

**Verification:** notes reviewed against upstream source, not README claims.
**Dependencies:** None. **Scope:** S (notes only).

### Task 0.2: Snapshot the model catalog + licenses

**Description:** Enumerate handy-computer HF org GGUF repos; for each
family record repo id, sizes, quant variants available, language coverage,
license, streaming capability. This becomes the `models.py` registry data.

**Acceptance criteria:**
- [ ] Table in `tasks/notes.md` covering every supported family (SPEC: all families)
- [ ] Default `Whisper large-v3-turbo` repo + `q4_k_m` artifact confirmed to exist
- [ ] Any non-MIT/BSD/Apache model flagged (excluded or ask-first)

**Verification:** each repo id resolves on HF.
**Dependencies:** None. **Scope:** S.

### Task 0.3: FastEnhancer license gate

**Description:** Check LICENSE of `aask1357/fastenhancer` (upstream) and
`kdrkdrkdr/fastenhancer.c.wasm` (C port). Decide: FastEnhancer (pass) or
GTCRN-via-sherpa-onnx fallback (fail/unclear). Record decision in SPEC
Decisions log.

**Acceptance criteria:**
- [ ] Both licenses documented with links
- [ ] Decision recorded in SPEC.md Decisions log

**Verification:** license files quoted in notes.
**Dependencies:** None. **Scope:** XS.
**Risk note:** wasm port currently shows no LICENSE file — fallback is pre-approved by SPEC.

---

## Phase 1 — Minimal Working STT (vertical slice: HA hears one phrase)

### Task 1.1: Repo scaffold

**Description:** Create repo skeleton mirroring `ha-app-nemo-asr-cpp`:
lint configs, `transcribe-cpp/` app dir with `config.yaml` (minimal
options: model/quantization/language/hf_token/debug_logging),
`apparmor.txt`, `translations/{en,ko}.yaml`, `pyproject.toml`, DOCS.md +
.README.j2 stubs, `git init` + first commit.

**Acceptance criteria:**
- [ ] `yamllint .` / `markdownlint` pass
- [ ] `config.yaml` schema valid (slug `transcribe-cpp`, port 10370, discovery wyoming, backup_exclude)

**Verification:** lint commands from SPEC §2.
**Dependencies:** none. **Scope:** M.

### Task 1.2: Dockerfile builder stage

**Description:** Builder stage on `base-debian:trixie`: fetch pinned
`TRANSCRIBE_REF`, build shared lib (+ggml, `GGML_NATIVE=OFF`) and
`transcribe-quantize` binary, strip `.so`s into `/usr/local/lib`, pip
install `wyoming`, `huggingface_hub`, `numpy` + the transcribe python
binding + our package into `/usr/local`. Keep convert scripts from
upstream in `/usr/local/share/transcribe/scripts/` for Task 10.

**Acceptance criteria:**
- [ ] `container build` completes the builder stage on this Mac (aarch64)
- [ ] hadolint passes

**Verification:** `container build --target builder …` (or full build once runtime exists).
**Dependencies:** 0.1, 1.1. **Scope:** M (1 file, but the riskiest one).

### Task 1.3: engine.py + models.py (minimal) + batch handler

**Description:** `engine.py` wraps the binding (load model path, batch
transcribe PCM16 mono 16k, `supports_streaming` flag stubbed to False).
`models.py` with just the default Whisper entry + `/data/models` cache
path + HF download via `huggingface_hub`. `handler.py` batch-only Wyoming
handler (buffer chunks, transcribe at AudioStop) + `__main__.py`
bootstrapping the server from env/options.

**Acceptance criteria:**
- [ ] pytest unit tests pass for option parsing and cache-path logic (no model download)
- [ ] Handler emits a final `Transcript` for a WAV pushed via wyoming client

**Verification:** `python3 -m pytest tests/`; manual wyoming round-trip inside container (Task 1.4).
**Dependencies:** 0.1, 1.1. **Scope:** M.

### Task 1.4: Runtime stage + s6 + discovery + healthcheck

**Description:** Runtime stage (apt: python3, libgomp1, libstdc++6,
ca-certificates, netcat-traditional; `COPY --from=builder /usr/local`;
`COPY rootfs /`; ldconfig+chmod RUN; HEALTHCHECK describe|nc). s6 services
`transcribe-cpp` and `discovery` ported from nemo-asr-cpp.

**Acceptance criteria:**
- [ ] Full `container build` succeeds; runtime adds ≤ 5 layers over base
- [ ] `container run` + wyoming `describe` returns service info; test WAV → transcript with default model
- [ ] shellcheck/hadolint pass

**Verification:** SPEC §2 commands; layer count via `container images inspect`.
**Dependencies:** 1.2, 1.3. **Scope:** M.

### CHECKPOINT A — Minimal STT works
- [ ] Local image builds (Apple Container CLI, aarch64)
- [ ] WAV → correct transcript via Wyoming with Whisper large-v3-turbo q4_k_m
- [ ] Lints green; commit; human review before Phase 2

---

## Phase 2 — Full Catalog & Streaming

### Task 2.1: Full model registry + quantization selection

**Description:** Populate `models.py` from Task 0.2 data (every family),
including per-model license + streaming flag; resolve `quantization`
option to the matching HF artifact (fallback rules when a quant is
missing: nearest larger, warn). Generate the `list(...)` for config.yaml
schema and translations from the registry (keep in sync test).

**Acceptance criteria:**
- [ ] Registry integrity test: every entry has repo/license/streaming/quants
- [ ] config.yaml `model:` list matches registry exactly (unit test)
- [ ] Switching model+quant in options loads the right file (manual, 2 models)

**Verification:** pytest + one non-Whisper model smoke in container.
**Dependencies:** 0.2, CHECKPOINT A. **Scope:** M.

### Task 2.2: Streaming partials

**Description:** Port the `wyoming_nemotron_asr_c` handler pattern:
`TranscriptStart` → `TranscriptChunk` deltas → `TranscriptStop` + final
`Transcript`. Engine exposes real `supports_streaming` per family (from
0.1 findings); batch families keep Phase-1 behavior.

**Acceptance criteria:**
- [ ] Delta-computation unit tests pass (prefix and non-prefix cases)
- [ ] Streaming family emits partials before AudioStop (manual)
- [ ] Whisper still returns final-only, no orphaned Start/Stop events

**Verification:** pytest + manual wyoming session logs.
**Dependencies:** 2.1. **Scope:** M.

### CHECKPOINT B — Catalog + streaming
- [ ] Partials visible with a streaming model; batch models unaffected
- [ ] Commit; human review

---

## Phase 3 — Audio Pipeline Extras (each opt-in, default off)

### Task 3.1: Speech enhancement

**Description:** Per Task 0.3 decision: (a) FastEnhancer — port
`enhancer.py` bridge + Dockerfile build step (`FASTENHANCER_REF`) from
nemotron-asr-c, wired before engine input in both batch and streaming
paths; or (b) GTCRN via sherpa-onnx applied to the buffered utterance
(final transcript only). `speech_enhancement: bool` option.

**Acceptance criteria:**
- [ ] Option off → byte-identical pipeline behavior (unit test on pass-through)
- [ ] Option on → noisy WAV transcribes no worse (manual smoke)
- [ ] License notice added to DOCS.md

**Verification:** pytest + container smoke with noisy sample.
**Dependencies:** 0.3, CHECKPOINT B. **Scope:** M.

### Task 3.2: Diarization

**Description:** `diarize.py`: sherpa-onnx offline diarization (pyannote
segmentation-3.0 ONNX + CAM++ embedding), models downloaded to
`/data/models/diarization/` on first enable. Buffer utterance PCM at
AudioStop when `diarization: true`, map segments to `[Speaker N]` inline
tags in the final `Transcript` only. Add `max_speakers` option. DOCS.md
section documenting the `ha-app-voiceprint` chain as the preferred
Assist path.

**Acceptance criteria:**
- [ ] Tag-rendering unit tests (segment → tagged text, incl. overlaps/ordering)
- [ ] 2-speaker WAV → `[Speaker 1]`/`[Speaker 2]` tags (manual, container)
- [ ] Off by default; no model download unless enabled

**Verification:** pytest + container smoke with 2-speaker sample.
**Dependencies:** CHECKPOINT B. **Scope:** M.

### Task 3.3: Custom model conversion (torch-cpu)

**Description:** `convert.py`: when `custom_model` is set — bootstrap
`/data/convert-venv` (pip torch-cpu + upstream conversion deps, logged
progress), download HF checkpoint, run the matching upstream
`convert-*.py`, then `transcribe-quantize` to the selected quant, cache as
`/data/models/<repo>__<quant>.gguf`, then serve it. Skip everything when
cache hit. Clear log errors for unsupported architectures.

**Acceptance criteria:**
- [ ] Cache/skip + naming logic unit-tested (no torch in tests)
- [ ] One real fine-tuned Whisper HF repo converts end-to-end in container and serves (manual; slow OK)
- [ ] Image size unchanged (torch not in image)

**Verification:** pytest + one full conversion run logged.
**Dependencies:** CHECKPOINT B (independent of 3.1/3.2). **Scope:** M.

### CHECKPOINT C — Feature-complete
- [ ] All SPEC acceptance criteria except CI/release ones pass
- [ ] Commit; human review

---

## Phase 4 — CI & Release

### Task 4.1: CI workflows

**Description:** Port `ci.yml` (lints + pytest), `build.yml` (HA builder →
ghcr.io/saya6k/app-transcribe-cpp per-arch, manifest, dispatch
`app-released`/`app-released-beta` to ha-apps), `release-drafter*` from
voiceprint, adjusting names/slug/image.

**Acceptance criteria:**
- [ ] ci.yml green on GitHub for the pushed branch
- [ ] build.yml dry-run (workflow_dispatch with test version) publishes per-arch tags

**Verification:** GitHub Actions runs.
**Dependencies:** CHECKPOINT C. **Scope:** M.

### Task 4.2: Docs & release checklist

**Description:** Final DOCS.md (options table, voiceprint chain, model
catalog + licenses, conversion guide, manual smoke checklist from SPEC
§5.4), .README.j2, translations ko/en for all options, icon/logo.

**Acceptance criteria:**
- [ ] Every config option documented in DOCS.md + both translations
- [ ] Third-party license table complete
- [ ] markdownlint green

**Verification:** lint + human read-through.
**Dependencies:** 4.1. **Scope:** S.

### CHECKPOINT D — Release candidate
- [ ] Manual smoke checklist executed on amd64 HA instance
- [ ] Tag first release; ha-apps picks it up

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Python binding expects pip-built wheel, not system `.so` | High (Dockerfile shape) | Task 0.1 resolves before Dockerfile work; worst case build wheel in builder stage |
| Streaming API differs per family / not exposed in binding | Med | `supports_streaming` gate; batch fallback is always correct |
| FastEnhancer license unclear | Low (fallback ready) | Task 0.3 gate; GTCRN fallback pre-approved |
| Upstream convert scripts assume GPU/nemo extras | Med | Task 3.3 pins convert deps in the /data venv; test with a real Whisper fine-tune |
| Apple Container CLI vs docker BuildKit differences (e.g. `--target`) | Low | Verify flags in Task 1.2; CI uses HA builder regardless |
| aarch64-only local testing (this Mac) | Med | amd64 verified via CI build + manual smoke on HA box at Checkpoint D |

## Open Questions

- None blocking — all SPEC decisions logged. Curated-list contents finalize in Task 0.2.
