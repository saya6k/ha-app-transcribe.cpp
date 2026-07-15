---
name: feedback-log-severity-by-mistake-likelihood
description: "When a dropped/ignored config value can fail for different reasons, split the log level by how likely each reason is to be a genuine user mistake, not one uniform level for all."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5f731575-e4ff-4c7b-868f-4742ef1859cb
---

Don't log every "this input was ignored" case at the same level. Split by how likely the specific reason is to be a real mistake:

- Wrong **name** (unrecognized key/typo) or right name with a wrong **value** (fails to parse/coerce) → `WARNING`. The user almost certainly meant something else.
- Right name, right value, but it just doesn't apply in this context (e.g. a per-family option set while a different family is loaded) → `INFO` (or whatever the addon's default-visible-but-quiet level is). This is often intentional/routine — e.g. one shared options list reused across multiple model choices — not a mistake.

**Why:** Refined during [[model-options-feature]] in ha-app-transcribe.cpp (builds on [[feedback_silent_ignore_visible_log]] — "ignored things must still be visibly logged"). The user's follow-up: uniform WARNING for both cases is wrong — "name/value가 아예 적용 안되면 notice를 해도, name이 맞는데 적용이 안되는 등 오류가 있는 value인 경우 경고 단계를 조정할 필요"). A uniform level either over-alarms on routine cases or under-alarms on real typos.

**How to apply:** Whenever adding "silently ignore but log" behavior with more than one distinct drop reason, ask which reasons are "user probably made a typo" (warn loud) vs "this is normal/expected given how the feature is used" (log quiet-but-visible). Don't collapse them into one severity by default.
