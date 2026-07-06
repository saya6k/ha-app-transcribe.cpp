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

# Upstream conversion scripts (torch runs from a lazy /data venv at runtime,
# never from the image) — needed by the custom_model pipeline. Only the
# converters + their shared lib/ (~0.7 MB); the rest of scripts/ (uv lockfiles,
# CI, WER tooling — ~6 MB) is dev-only upstream.
RUN mkdir -p /usr/local/share/transcribe-cpp/scripts \
    && cp scripts/convert-*.py /usr/local/share/transcribe-cpp/scripts/ \
    && cp -r scripts/lib /usr/local/share/transcribe-cpp/scripts/lib

RUN pip3 install --no-cache-dir --no-compile "wyoming>=1.9,<2" "huggingface_hub" "numpy" "sherpa-onnx" \
    && pip3 install --no-cache-dir --no-compile --no-deps /src/bindings/python
COPY pyproject.toml /usr/src/app/
COPY wyoming_transcribe_cpp/ /usr/src/app/wyoming_transcribe_cpp/
RUN pip3 install --no-cache-dir --no-compile --no-deps /usr/src/app

# ---- runtime ----
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSCRIBE_LIBRARY=/usr/local/lib/libtranscribe.so

# libopenblas0: host decoder BLAS. libstdc++6: C++ runtime for libtranscribe/ggml.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libopenblas0 \
        libstdc++6 \
        ca-certificates \
        netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# libtranscribe + ggml shared libs, transcribe-quantize, conversion scripts,
# Python deps and our bridge — all under /usr/local from the builder stage.
COPY --from=builder /usr/local /usr/local

COPY rootfs /

RUN ldconfig \
    && chmod +x \
        /etc/s6-overlay/s6-rc.d/transcribe-cpp/run \
        /etc/s6-overlay/s6-rc.d/transcribe-cpp/finish \
        /etc/s6-overlay/s6-rc.d/discovery/run

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10380 \
        | grep -qi "transcribe" \
        || exit 1
