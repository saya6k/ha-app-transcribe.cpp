---
name: feedback-docs-link-upstream-not-reexplain
description: "Document new addon behavior by linking to upstream's own docs/source (pinned to the vendored commit) rather than re-explaining every field by hand."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5f731575-e4ff-4c7b-868f-4742ef1859cb
---

When documenting a feature in this addon (ha-app-transcribe.cpp) that exposes upstream `transcribe.cpp` behavior (e.g. per-family runtime options), prefer linking to upstream's docs/source — pinned to the exact vendored commit (`TRANSCRIBE_REF` in the Dockerfile) so the link matches what's actually in the image — over writing a full field-by-field re-description in `DOCS.md`.

**Why:** User correction during planning for [[model-options-feature]]: "DOCS.md에 문서화, upstream 문서 링크." Re-documenting upstream's own option semantics duplicates effort and rots on the next `TRANSCRIBE_REF` bump; upstream already documents it (docstrings, `--help` text) better than a second copy would.

**How to apply:** Only write local content for things that are genuinely this addon's own glue (e.g. the config-key-name → upstream-class mapping table). Link out — with a commit-pinned permalink — for anything that's upstream's own semantics/types/valid ranges.
