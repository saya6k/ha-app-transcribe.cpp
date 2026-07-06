# Phase 0 Research Notes

Date: 2026-07-06

## 0.1 Upstream pin + binding API

- `TRANSCRIBE_REF = d89ecb75062e8457681c563994675dc60e31db80` (main @ 2026-07-06)
- Upstream license: MIT; ggml + miniz vendored (MIT), THIRD-PARTY-LICENSES.md present.

### Python binding (`bindings/python`)

- **Pure-Python ctypes** package `transcribe_cpp` (hatchling). No compiled
  extension. Normally depends on the `transcribe-cpp-native` wheel — for the
  image we `pip install --no-deps` the binding and point it at our own build
  via **`TRANSCRIBE_LIBRARY=/usr/local/lib/libtranscribe.so`** (documented
  dev-tree path in `_library.py`).
- API surface (all we need):
  - `Model(path)` → `.capabilities()`: `native_sample_rate`, `languages`,
    **`supports_streaming`**, `supports_translate`, `max_audio_ms`, …
    → runtime streaming detection; no hardcoded family table needed.
    Wyoming `describe` languages come from `capabilities().languages`.
  - `model.session(n_threads=…)` → `Session`
  - Batch: `session.run(pcm) -> Result(text, language, segments, words…)`
    - PCM: float32 numpy / buffer at `native_sample_rate` (16k for ASR families)
  - Streaming: `session.stream(...) -> Stream`; `stream.feed(pcm) -> StreamUpdate`;
    `stream.text() -> StreamText(committed, tentative, display)`;
    `stream.finalize()`; raises `NotImplementedByModel` if unsupported.
    Wyoming deltas: emit new tail of `text().display` (or committed-based).
  - Session is single-threaded; one stream at a time → serialize with a lock.
- CMake flags for our builder stage:
  `-DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_BUILD_TOOLS=ON`
  (tools = transcribe-quantize) `-DTRANSCRIBE_BUILD_TESTS=OFF
  -DTRANSCRIBE_BUILD_EXAMPLES=OFF -DTRANSCRIBE_METAL=OFF
  -DGGML_NATIVE=OFF`; `TRANSCRIBE_USE_SYSTEM_BLAS=ON` (default) + libopenblas
  → README claims 10–15× decoder speedup; runtime needs `libopenblas0`.
- Conversion scripts: `scripts/convert-<family>.py` (whisper, parakeet,
  qwen3_asr, moonshine, canary, sensevoice, voxtral, …), env specs under
  `scripts/envs/`, shared helpers `scripts/lib/`. torch-based, CPU-capable.
- Quantizer: `tools/transcribe-quantize` (C++, built with TOOLS=ON).

## 0.2 Model catalog

- Source of truth: `scripts/hf_cards/*.yaml` in the pinned checkout — one
  card per released GGUF repo with `target_repo`, `license`, `languages`,
  `capabilities.streaming`, `quants[].{name,filename}`.
  → `models.py` registry is **generated** from these cards
  (`scripts/gen_registry.py` in this repo), parsed snapshot: `tasks/catalog.json`.
- 65 models. Licenses: apache-2.0 ×23, mit ×21, cc-by-4.0 ×14, other ×6
  (nemotron ×2 = NVIDIA Open Model License; Fun-ASR ×2, SenseVoice, MedASR),
  cc-by-nc-4.0 ×1 (`canary-1b`).
- **Policy:** dropdown includes everything except `cc-by-nc-4.0`
  (SPEC "never non-commercial") → 64 entries. 'other'-licensed models stay,
  license shown in DOCS table (weights are user-downloaded at runtime, never
  redistributed in the image).
- Quant vocab across cards: F32/BF16/F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M — not every
  model has every quant → fallback rule: requested → nearest larger present
  (Q4_K_M→Q5_K_M→Q6_K→Q8_0→F16/BF16→F32), warn on fallback.
- Default confirmed: `handy-computer/whisper-large-v3-turbo-gguf`, Q4_K_M
  artifact exists.
- Streaming-capable (7): moonshine-streaming tiny/small/medium,
  nemotron-3.5-asr-streaming-0.6b, nemotron-speech-streaming-en-0.6b,
  parakeet-unified-en-0.6b, voxtral-mini-4b-realtime-2602.

## 0.3 FastEnhancer license gate — **FAILED → GTCRN fallback**

- Upstream `aask1357/fastenhancer`: MIT ✓ (GitHub license API).
- C port `kdrkdrkdr/fastenhancer.c.wasm`: **no LICENSE file** (license API
  404) → port code is all-rights-reserved by default. Gate fails.
- **Decision:** `speech_enhancement` uses **GTCRN via sherpa-onnx**
  (Apache-2.0 code, MIT model `gtcrn_simple`), applied to the buffered
  utterance at AudioStop. When enhancement is enabled, decode is
  batch-at-utterance-end (streaming partials disabled for that session) —
  pre-approved by SPEC. Revisit FastEnhancer if the port gains a license.
