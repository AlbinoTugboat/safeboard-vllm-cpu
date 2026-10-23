# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b AS uv
FROM python:3.12.15-slim-trixie@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47 AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libnuma1 libgomp1 libstdc++6 gcc g++ \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

FROM base AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.txt /build/requirements.txt
RUN uv venv --python /usr/local/bin/python /opt/venv \
    && uv pip install --python /opt/venv/bin/python --require-hashes --no-build \
         --torch-backend cpu -r /build/requirements.txt \
    && uv pip check --python /opt/venv/bin/python
RUN /opt/venv/bin/python -c "import torch, vllm; assert torch.version.cuda is None; print(vllm.__version__, torch.__version__)"

FROM base AS runtime
COPY --from=dependencies /opt/venv /opt/venv
RUN rm -rf /usr/local/lib/python3.12/site-packages/pip \
      /usr/local/lib/python3.12/site-packages/pip-*.dist-info /usr/local/lib/python3.12/ensurepip \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid 10001 --no-create-home app \
    && mkdir -p /app /tmp/home && chown 10001:10001 /tmp/home
WORKDIR /app
COPY app/ /app/
COPY requirements.txt /app/requirements.txt
ENV PYTHONPATH=/app HOME=/tmp/home XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    VLLM_LOGGING_CONFIG_PATH=/app/logging.json VLLM_LOGGING_LEVEL=INFO \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE=0 \
    LD_PRELOAD=/opt/venv/lib/libiomp5.so
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "/app/launch.py"]
