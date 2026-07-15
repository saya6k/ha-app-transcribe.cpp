---
name: feedback-silent-ignore-visible-log
description: "When a design calls for \"silently ignore\" invalid/unsupported input, the ignore must still be observable in logs at default verbosity — not truly silent."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5f731575-e4ff-4c7b-868f-4742ef1859cb
---

When a feature is designed to silently ignore bad/unsupported input rather than error (e.g. an unsupported per-model config key), log the ignore path at a level visible under the addon's *default* log level — not `.debug()`, which is invisible unless the user raises verbosity first.

**Why:** Caught during planning for [[model-options-feature]] in ha-app-transcribe.cpp — the addon's default `log_level` is `info`, and `.debug()` calls are filtered out at that threshold (`__main__.py`'s `_LOG_LEVEL_MAP`). A user whose config typo or unsupported option silently no-ops would see zero trace of why. The user's correction: "적용이 안되는건 s6-overlay log로 notice" — the s6-overlay service log must show it by default.

**How to apply:** For this project (and likely others with similar HA-addon log-level plumbing), any "ignore invalid/unsupported X" behavior should log at `.warning()` (or whatever maps to the addon's default-visible threshold) rather than `.debug()`. Applies broadly: "fail soft" should still be discoverable without the user having to guess and raise log verbosity to debug it.
