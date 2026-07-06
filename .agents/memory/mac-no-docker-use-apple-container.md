---
name: mac-no-docker-use-apple-container
description: "This MacBook has no Docker; use Apple Container CLI (`container`) for local image build tests"
metadata: 
  node_type: memory
  type: user
  originSessionId: 4bd1a92e-49fc-4b5a-a928-4d6aa6b077fb
---

The user's MacBook (darwin) does not have Docker installed. For local container image builds/tests of the ha-app-* projects, use the Apple Container CLI (`container build` / `container run`) instead of `docker`.

**Why:** User explicitly corrected a spec that used `docker build` commands (2026-07-06).
**How to apply:** In specs, docs, and local verification steps, write `container build ...` / `container run ...` commands; keep `docker` only for CI (GitHub Actions / HA builder) contexts.
