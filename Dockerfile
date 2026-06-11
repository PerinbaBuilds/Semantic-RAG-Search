FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY part1_prepare.py   .
COPY part2_clustering.py .
COPY part3_cache.py     .
COPY part4_api.py       .
COPY rag/               rag/
COPY static/            static/
COPY embeddings/        embeddings/

EXPOSE 7860

RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "-m", "uvicorn", "part4_api:app", "--host", "0.0.0.0", "--port", "7860"]
