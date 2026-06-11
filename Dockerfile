# Multi-stage build: builder installs deps, runtime copies only what's needed.

# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY part1_prepare.py   .
COPY part2_clustering.py .
COPY part3_cache.py     .
COPY part4_api.py       .
COPY rag/               rag/
COPY static/            static/
COPY embeddings/        embeddings/

EXPOSE 7860

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["/opt/venv/bin/uvicorn", "part4_api:app", "--host", "0.0.0.0", "--port", "7860"]
