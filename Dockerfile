# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Installs Python packages into /install so nothing from the build toolchain
# leaks into the final image.
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Bring in installed packages from the builder stage
COPY --from=builder /install /usr/local

# Application source only — model/artifacts/ is excluded via .dockerignore
# and mounted as a read-only volume at runtime.
COPY api/           ./api/
COPY data/          ./data/
COPY model/         ./model/
COPY config.py      .
COPY logging_config.py .

# Run as non-root for container security
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# TRANSFORMERS_OFFLINE=1 prevents any HuggingFace Hub network calls at runtime.
# All required files (model weights + tokenizer) are in the mounted artifact volume.
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    ARTIFACTS_DIR=/app/model/artifacts/final \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# start-period gives the model time to load before the first health probe fires
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
