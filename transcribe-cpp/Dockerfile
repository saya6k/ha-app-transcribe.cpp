# Debian (glibc) base — we compile libtranscribe (+ vendored ggml) from source
# and link the stock libstdc++. The generic tag is a multi-arch manifest;
# BuildKit resolves amd64 / aarch64 and the build runs natively per arch, so
# the compiled libs match the target CPU.
ARG BUILD_FROM=ghcr.io/home-assistant/base-debian:trixie

# Pin upstream transcribe.cpp — no release tags yet, so we pin a commit SHA.
# GGML_NATIVE is OFF for a portable build (ggml still picks AVX2/NEON kernels
# at runtime). ggml is vendored in-tree (no submodules).
ARG TRANSCRIBE_REF=d89ecb75062e8457681c563994675dc60e31db80

# ---- builder: compile libtranscribe.so + transcribe-quantize, install deps ----
FROM ${BUILD_FROM} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TRANSCRIBE_REF
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        ca-certificates \
        libopenblas-dev \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Partial clone (blob:none) + sparse checkout: upstream ships ~33 MB of
# sample WAVs plus reports/docs/tests we never build — skip their blobs
# entirely instead of downloading and discarding them.
WORKDIR /src
RUN git init -q . \
    && git remote add origin https://github.com/handy-computer/transcribe.cpp.git \
    && git config remote.origin.promisor true \
    && git config remote.origin.partialclonefilter blob:none \
    && git sparse-checkout set --no-cone \
        '/*' '!/samples' '!/reports' '!/docs' '!/tests' \
    && git fetch --depth 1 --filter=blob:none origin "${TRANSCRIBE_REF}" \
    && git checkout -q FETCH_HEAD

# Shared lib + quantize tool only (no CLI/examples/tests). System BLAS stays
# ON (OpenBLAS) — upstream README: ~10-15x faster decoder. cmake --install is
# skipped; find gathers every built .so into /usr/local/lib so one
# COPY --from=builder /usr/local covers libs, tools, scripts, and Python deps.
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRANSCRIBE_BUILD_SHARED=ON \
        -DTRANSCRIBE_BUILD_TOOLS=ON \
        -DTRANSCRIBE_BUILD_TESTS=OFF \
        -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
        -DGGML_NATIVE=OFF \
    && cmake --build build -j --target transcribe transcribe-quantize \
    && find build -name '*.so*' \( -type f -o -type l \) -exec cp -a {} /usr/local/lib/ \; \
    && find /usr/local/lib -name '*.so*' -type f -exec strip --strip-unneeded {} \; \
    && test -e /usr/local/lib/libtranscribe.so \
    && install -m 0755 "$(find build -name transcribe-quantize -type f | head -n1)" \
        /usr/local/bin/transcribe-quantize \
    && strip --strip-unneeded /usr/local/bin/transcribe-quantize

# Upstream conversion scripts (torch runs from lazy /data venvs at runtime,
# never from the image) — needed by the custom_model pipeline. Converters +
# their shared lib/ (~0.7 MB) + the per-family envs/*/pyproject.toml dep
# manifests convert.py parses; the rest of scripts/ (uv lockfiles, CI, WER
# tooling — ~6 MB) is dev-only upstream.
RUN mkdir -p /usr/local/share/transcribe-cpp/scripts \
    && cp scripts/convert-*.py /usr/local/share/transcribe-cpp/scripts/ \
    && cp -r scripts/lib /usr/local/share/transcribe-cpp/scripts/lib \
    && for env in scripts/envs/*/; do \
        install -D -m 0644 "${env}pyproject.toml" \
            "/usr/local/share/transcribe-cpp/envs/$(basename "${env}")/pyproject.toml"; \
    done

RUN pip3 install --no-cache-dir --no-compile "wyoming>=1.9,<2" "huggingface_hub" "numpy" "onnxruntime" \
    && pip3 install --no-cache-dir --no-compile --no-deps /src/bindings/python
COPY pyproject.toml /usr/src/app/
COPY wyoming_transcribe_cpp/ /usr/src/app/wyoming_transcribe_cpp/
RUN pip3 install --no-cache-dir --no-compile --no-deps /usr/src/app

# Slim the /usr/local payload before the runtime COPY: strip native
# wheels (onnxruntime alone carries tens of MB of symbols) and drop
# bundled test suites/bytecode.
RUN find /usr/local/lib/python3*/dist-packages -name '*.so*' -type f \
        -exec strip --strip-unneeded {} + \
    && find /usr/local/lib/python3*/dist-packages -type d \( -name tests -o -name __pycache__ \) \
        -prune -exec rm -rf {} +

# ---- runtime ----
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# HF_HUB_DISABLE_XET: the xet download backend buffers aggressively and
# has OOM-killed NeMo weight fetches on small hosts; plain HTTP ranges
# are slightly slower but memory-flat.
ARG TRANSCRIBE_REF
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TRANSCRIBE_LIBRARY=/usr/local/lib/libtranscribe.so \
    TRANSCRIBE_REF=${TRANSCRIBE_REF}

# libopenblas0: host decoder BLAS. libstdc++6: C++ runtime for libtranscribe/
# ggml. No git — github git+ converter deps install from commit tarballs.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libopenblas0 \
        libstdc++6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# libtranscribe + ggml shared libs, transcribe-quantize, conversion scripts,
# Python deps and our bridge — all under /usr/local from the builder stage.
COPY --from=builder /usr/local /usr/local

COPY rootfs /

# Privilege split: transcribe = Wyoming server, converter = conversion
# worker. Shared primary group so the worker's unix socket (0660) accepts
# the server; write separation stays owner-only on the /data subtrees.
RUN ldconfig \
    && groupadd -r transcribe \
    && useradd -r -g transcribe -d /data -s /usr/sbin/nologin transcribe \
    && useradd -r -g transcribe -d /data/convert-venv -s /usr/sbin/nologin converter \
    && chmod +x \
        /etc/s6-overlay/s6-rc.d/transcribe-cpp/run \
        /etc/s6-overlay/s6-rc.d/transcribe-cpp/finish \
        /etc/s6-overlay/s6-rc.d/discovery/run \
        /etc/s6-overlay/s6-rc.d/init-perms/run \
        /etc/s6-overlay/s6-rc.d/convert-worker/run

# stdlib probe (netcat dropped from the image): send a Wyoming describe,
# read the exact data_length payload, expect the service name in it.
HEALTHCHECK --start-period=10m \
    CMD python3 -c 'import json,socket,sys; s=socket.create_connection(("localhost",10380),2); s.settimeout(2); s.sendall(b"""{ "type": "describe" }\n"""); f=s.makefile("rb"); h=json.loads(f.readline()); sys.exit(0 if b"transcribe" in f.read(h.get("data_length",0)).lower() else 1)'
