# Home Assistant App: Transcribe.cpp

## How it works

```text
HA Assist ──► Transcribe.cpp (this app, :10380)
```

Speech-to-text over the [Wyoming protocol](https://github.com/rhasspy/wyoming),
running the [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
GGUF model catalog on the ggml runtime (CPU). The selected model downloads to
`/data/models` on first start and stays resident. Streaming-capable models
(see the table below) emit live partial transcripts; the others return the
final transcript at end of utterance.

## Setup

1. Pick a `model` and `quantization` in the app configuration.
2. In **Settings → Devices & Services → Wyoming Protocol**, add this app
   (port 10380) and select it as the STT engine of your Assist pipeline.
   The language always comes from the pipeline — there is no language option.

## Options

| Option | Default | Description |
|---|---|---|
| `model` | `whisper-large-v3-turbo` | ASR model from the transcribe.cpp GGUF catalog (table below) |
| `quantization` | `q4_k_m` | GGUF weight precision, smallest → largest: `q4_k_m` → `f16`. If the model doesn't publish the chosen one, the nearest larger is used |
| `custom_model` | `''` | HF repo ID of your own fine-tune to convert on-device (see below) — overrides `model` |
| `speech_enhancement` | `false` | GTCRN denoise before decoding (noisy rooms; disables live partials) |
| `diarization` | `false` | Tag the final transcript with `[Speaker N]` (disables live partials) |
| `max_speakers` | unset | Diarization hint: expected number of speakers (unset = auto) |
| `hf_token` | `''` | HuggingFace token for gated repos / rate limits |
| `log_level` | `info` | Log verbosity: trace–fatal |

## Custom fine-tuned models (`custom_model`)

Set `custom_model` to a HuggingFace repo ID (e.g. `you/whisper-small-ko-ft`).
On next start the app:

1. detects the model family from hub metadata (`model_type`, file layout),
2. installs that family's converter dependencies (torch-cpu, pinned by
   upstream) into a persistent venv under `/data/convert-venv/<family>`
   — one-time per family, slow on purpose; conversion is CPU-only,
3. converts the checkpoint with upstream's matching `convert-*.py`,
4. quantizes it with `transcribe-quantize` to your `quantization` choice,
5. caches the result under `/data/models/custom/` and serves it.

A sidecar records the sha256 of every original weight file: restarting
with the same repo reuses the cache instantly, and a re-uploaded
checkpoint (changed weights) reconverts automatically. README-only repo
edits do **not** retrigger conversion.

Conversion runs in a separate unprivileged worker process; the Wyoming
server talks to it only over an internal unix socket.

### Supported families

Every converter shipped by the pinned transcribe.cpp commit works
on-device: `whisper` (incl. Breeze), `moonshine`, `moonshine_streaming`,
`qwen3_asr`, `voxtral`, `voxtral_realtime`, `granite`, `granite_nar`,
`medasr`, `cohere`, `sensevoice`, `funasr_nano`, and the NeMo families
`parakeet`, `canary`, `canary_qwen`. Exceptions and caveats:

- **gigaam** — the upstream converter only fetches official GigaAM
  weights, so fine-tune import is rejected; use the curated catalog.
- **NeMo families** (parakeet/canary/canary_qwen): the first conversion
  downloads **several GB** into `/data/convert-venv/` and can take tens
  of minutes on aarch64. These venvs run on a managed CPython 3.12
  (downloaded once, checksum-pinned) because parts of the NeMo
  dependency tree ship no Python 3.13 wheels and the image carries no
  compiler on purpose.
- Converters for whisper/moonshine/voxtral/parakeet/canary need to know
  the *base* variant of your fine-tune. It is read from the repo's
  `base_model:` tag (set it in your HF model card) or, failing that,
  from a catalog slug embedded in the repo name (e.g.
  `kb-whisper-tiny` → `whisper-tiny`).

Unsupported or undetectable checkpoints stop the add-on with a clear
log message — fix `custom_model` (or clear it to use the catalog
`model`) and start the add-on again.

## Speaker attribution: this app vs. Voiceprint

For **Assist voice commands** (one speaker per utterance), built-in
diarization only yields anonymous `[Speaker 1]` labels. Chaining the
[Voiceprint](https://github.com/saya6k/ha-app-voiceprint) app in front gives
you *named*, enrolled speakers instead — that is the recommended setup:

```text
HA Assist ──► Voiceprint (:10350) ──► Transcribe.cpp (:10380)
```

Enable this app's `diarization` for **multi-speaker recordings** (meetings,
long-form audio) where per-utterance verification can't help.

## Model catalog

Generated from the pinned upstream release cards — every entry is
downloadable at runtime; nothing ships in the image. The one
non-commercially-licensed upstream model (`canary-1b`, CC-BY-NC-4.0) is
excluded from this list.

<!-- registry-table:begin -->
| Model | License | Streaming | Languages |
|---|---|---|---|
| [`breeze-asr-25`](https://huggingface.co/handy-computer/Breeze-ASR-25-gguf) | apache-2.0 | — | 2 |
| [`canary-180m-flash`](https://huggingface.co/handy-computer/canary-180m-flash-gguf) | cc-by-4.0 | — | 4 |
| [`canary-1b-flash`](https://huggingface.co/handy-computer/canary-1b-flash-gguf) | cc-by-4.0 | — | 4 |
| [`canary-1b-v2`](https://huggingface.co/handy-computer/canary-1b-v2-gguf) | cc-by-4.0 | — | 25 |
| [`canary-qwen-2.5b`](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf) | cc-by-4.0 | — | 1 |
| [`cohere-transcribe-03-2026`](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf) | apache-2.0 | — | 14 |
| [`fun-asr-mlt-nano-2512`](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf) | other | — | 31 |
| [`fun-asr-nano-2512`](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf) | other | — | 3 |
| [`gigaam-v3-ctc`](https://huggingface.co/handy-computer/gigaam-v3-ctc-gguf) | mit | — | 1 |
| [`gigaam-v3-e2e-ctc`](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf) | mit | — | 1 |
| [`gigaam-v3-e2e-rnnt`](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf) | mit | — | 1 |
| [`gigaam-v3-rnnt`](https://huggingface.co/handy-computer/gigaam-v3-rnnt-gguf) | mit | — | 1 |
| [`granite-4.0-1b-speech`](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf) | apache-2.0 | — | 6 |
| [`granite-speech-4.1-2b`](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf) | apache-2.0 | — | 6 |
| [`granite-speech-4.1-2b-nar`](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf) | apache-2.0 | — | 5 |
| [`granite-speech-4.1-2b-plus`](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf) | apache-2.0 | — | 5 |
| [`medasr`](https://huggingface.co/handy-computer/medasr-gguf) | other | — | 1 |
| [`moonshine-base`](https://huggingface.co/handy-computer/moonshine-base-gguf) | mit | — | 1 |
| [`moonshine-base-ar`](https://huggingface.co/handy-computer/moonshine-base-ar-gguf) | mit | — | 1 |
| [`moonshine-base-ja`](https://huggingface.co/handy-computer/moonshine-base-ja-gguf) | mit | — | 1 |
| [`moonshine-base-ko`](https://huggingface.co/handy-computer/moonshine-base-ko-gguf) | mit | — | 1 |
| [`moonshine-base-uk`](https://huggingface.co/handy-computer/moonshine-base-uk-gguf) | mit | — | 1 |
| [`moonshine-base-vi`](https://huggingface.co/handy-computer/moonshine-base-vi-gguf) | mit | — | 1 |
| [`moonshine-base-zh`](https://huggingface.co/handy-computer/moonshine-base-zh-gguf) | mit | — | 1 |
| [`moonshine-streaming-medium`](https://huggingface.co/handy-computer/moonshine-streaming-medium-gguf) | mit | yes | 1 |
| [`moonshine-streaming-small`](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf) | mit | yes | 1 |
| [`moonshine-streaming-tiny`](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf) | mit | yes | 1 |
| [`moonshine-tiny`](https://huggingface.co/handy-computer/moonshine-tiny-gguf) | mit | — | 1 |
| [`moonshine-tiny-ar`](https://huggingface.co/handy-computer/moonshine-tiny-ar-gguf) | mit | — | 1 |
| [`moonshine-tiny-ja`](https://huggingface.co/handy-computer/moonshine-tiny-ja-gguf) | mit | — | 1 |
| [`moonshine-tiny-ko`](https://huggingface.co/handy-computer/moonshine-tiny-ko-gguf) | mit | — | 1 |
| [`moonshine-tiny-uk`](https://huggingface.co/handy-computer/moonshine-tiny-uk-gguf) | mit | — | 1 |
| [`moonshine-tiny-vi`](https://huggingface.co/handy-computer/moonshine-tiny-vi-gguf) | mit | — | 1 |
| [`moonshine-tiny-zh`](https://huggingface.co/handy-computer/moonshine-tiny-zh-gguf) | mit | — | 1 |
| [`nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) | other | yes | 28 |
| [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf) | other | yes | 1 |
| [`parakeet-ctc-0.6b`](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-ctc-1.1b`](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-rnnt-0.6b`](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-rnnt-1.1b`](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-tdt-0.6b-v2`](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-tdt-0.6b-v3`](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf) | cc-by-4.0 | — | 25 |
| [`parakeet-tdt-1.1b`](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-tdt_ctc-1.1b`](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-tdt_ctc-110m`](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf) | cc-by-4.0 | — | 1 |
| [`parakeet-unified-en-0.6b`](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf) | cc-by-4.0 | yes | 1 |
| [`qwen3-asr-0.6b`](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf) | apache-2.0 | — | 30 |
| [`qwen3-asr-1.7b`](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf) | apache-2.0 | — | 30 |
| [`sensevoice-small`](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf) | other | — | 5 |
| [`voxtral-mini-3b-2507`](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf) | apache-2.0 | — | 8 |
| [`voxtral-mini-4b-realtime-2602`](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf) | apache-2.0 | yes | 13 |
| [`voxtral-small-24b-2507`](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf) | apache-2.0 | — | 8 |
| [`whisper-base`](https://huggingface.co/handy-computer/whisper-base-gguf) | apache-2.0 | — | 99 |
| [`whisper-base.en`](https://huggingface.co/handy-computer/whisper-base.en-gguf) | apache-2.0 | — | 1 |
| [`whisper-large`](https://huggingface.co/handy-computer/whisper-large-gguf) | apache-2.0 | — | 99 |
| [`whisper-large-v2`](https://huggingface.co/handy-computer/whisper-large-v2-gguf) | apache-2.0 | — | 99 |
| [`whisper-large-v3`](https://huggingface.co/handy-computer/whisper-large-v3-gguf) | apache-2.0 | — | 100 |
| [`whisper-large-v3-turbo`](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf) | apache-2.0 | — | 100 |
| [`whisper-medium`](https://huggingface.co/handy-computer/whisper-medium-gguf) | apache-2.0 | — | 99 |
| [`whisper-medium.en`](https://huggingface.co/handy-computer/whisper-medium.en-gguf) | apache-2.0 | — | 1 |
| [`whisper-small`](https://huggingface.co/handy-computer/whisper-small-gguf) | apache-2.0 | — | 99 |
| [`whisper-small.en`](https://huggingface.co/handy-computer/whisper-small.en-gguf) | apache-2.0 | — | 1 |
| [`whisper-tiny`](https://huggingface.co/handy-computer/whisper-tiny-gguf) | apache-2.0 | — | 99 |
| [`whisper-tiny.en`](https://huggingface.co/handy-computer/whisper-tiny.en-gguf) | apache-2.0 | — | 1 |
<!-- registry-table:end -->

## Third-party components

| Component | License | Used for |
|---|---|---|
| [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (+ vendored ggml, miniz) | MIT | ASR engine, GGUF conversion/quantization tools |
| [wyoming](https://github.com/rhasspy/wyoming) | MIT | Protocol server |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0 | Speech enhancement + diarization runtime |
| [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (`gtcrn_simple.onnx`) | MIT | Speech enhancement model |
| [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (ONNX export) | MIT | Diarization segmentation model |
| [3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker) | Apache-2.0 | Diarization speaker embedding model |

ASR model weights are downloaded by the user at runtime from the
`handy-computer` HuggingFace org; each model's license is listed in the
catalog table above.

## Release smoke checklist

- [ ] Wyoming `describe` healthcheck answers on :10380
- [ ] Assist end-to-end phrase with the default model (batch)
- [ ] Live partials visible with a streaming model (e.g. `moonshine-streaming-tiny`)
- [ ] `diarization: true` on a 2-speaker WAV yields `[Speaker N]` tags
- [ ] `speech_enhancement: true` transcribes a noisy WAV sensibly
- [ ] One `custom_model` Whisper fine-tune converts and serves (amd64)
- [ ] One NeMo-family conversion (e.g. a parakeet checkpoint) completes
      on amd64 (slow; needs several GB free in `/data`)
- [ ] AppArmor: app starts and serves STT with the profile enforced on
      HA OS; `ps` shows the server as `transcribe` and the worker as
      `converter`
