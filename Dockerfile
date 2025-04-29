# syntax=docker/dockerfile:1

######################## 1️⃣ Builder ########################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang cmake git curl pkg-config \
        libssl-dev libsndfile1-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | bash -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# -------- build-time switches --------
ARG FLASH_ATTN=0
ARG CUDA_COMPUTE_CAP=86
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

WORKDIR /workspace
COPY . .

RUN if [ "$FLASH_ATTN" = "1" ]; then \
      cargo build --release --features cuda,flash-attn --bin server ; \
    else \
      cargo build --release --features cuda --bin server ; \
    fi

######################## 2️⃣ Runtime ########################
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV FISH_PORT=3000

# ---- Python + RunPod SDK ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libsndfile1 libssl3 curl && \
    pip3 install --no-cache-dir runpod==1.* aiohttp httpx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Rust binary ----
COPY --from=builder /workspace/target/release/server /usr/local/bin/fish-speech

# ---- Voices + handler ----
COPY voices-template/ ./voices
COPY rp_handler.py ./

EXPOSE 3000 

CMD ["python3", "-u", "/app/rp_handler.py"]