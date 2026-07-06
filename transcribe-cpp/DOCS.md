# Home Assistant App: Transcribe (cpp)

## How it works

```text
HA Assist ──► Transcribe (cpp) (this app, :10380)
```

Speech-to-text over the [Wyoming protocol](https://github.com/rhasspy/wyoming),
running the [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
GGUF model catalog on the ggml runtime (CPU). The selected model is downloaded
to `/data/models` on first start and stays resident.

## Setup

1. Pick a `model` and `quantization` in the app configuration.
2. In **Settings → Devices & Services → Wyoming Protocol**, add this app
   (port 10380) and select it as the STT engine of your Assist pipeline.

## Options

| Option | Default | Description |
|---|---|---|
| `model` | `whisper-large-v3-turbo` | ASR model from the transcribe.cpp GGUF catalog |
| `quantization` | `q4_k_m` | GGUF weight precision (smallest → largest: q4_k_m → f16) |
| `hf_token` | `''` | Optional HuggingFace token for gated repos / rate limits |
| `debug_logging` | `false` | Verbose logging |

More options (speech enhancement, diarization, custom fine-tuned model
conversion) are documented here as they land — see the repository README for
status.
