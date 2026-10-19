# Build context is docker/ only. No paper, source repository, credentials,
# previous experiment, or generated implementation can enter an image layer.
FROM python:3.11.10-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/pipeline/requirements.txt
# CPU wheels only; execution stages do not install or download dependencies.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /opt/pipeline/requirements.txt
COPY worker.py evaluate.py replay.py /opt/pipeline/
ENV PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HOME=/tmp/home
WORKDIR /workspace

FROM node:22.14.0-bookworm-slim AS controller
ARG CODEX_VERSION=0.160.1
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@${CODEX_VERSION}
COPY bridge.py driver.py config.toml /opt/pipeline/
ENV HOME=/tmp/home CODEX_HOME=/tmp/codex PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/pipeline
ENTRYPOINT ["python3", "/opt/pipeline/driver.py"]

FROM python:3.11.10-slim-bookworm AS proxy
COPY proxy.py /opt/pipeline/proxy.py
ENTRYPOINT ["python3", "/opt/pipeline/proxy.py"]
