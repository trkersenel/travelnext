# TravelNext container image.
#
# Builds a single image that can run the API, the Streamlit UI, the ingestion
# pipeline or the experiments -- the command decides which. Everything runs
# locally; no cloud service and no API key is involved.

FROM python:3.12-slim AS base

# libgomp1 is required by LightGBM; the rest keeps the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first so code edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY tests/ ./tests/

# data/ and models/ are mounted as volumes in docker-compose so an ingested
# dataset survives image rebuilds; create them for the standalone case.
RUN mkdir -p data/raw data/interim data/processed models reports/figures

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 travelnext \
    && chown -R travelnext:travelnext /app
USER travelnext

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
