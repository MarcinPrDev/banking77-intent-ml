# syntax=docker/dockerfile:1
# Obraz serwujący BANKING77 (W3, ADR-0003): DistilBERT na CPU za FastAPI + uvicorn.
# Budowa z korzenia repo; MODEL_VERSION = deploy.model_version z configs/config.yaml:
#   docker build --build-arg MODEL_VERSION=distilbert-v1 -t banking77-intent-api:distilbert-v1 .
# Artefakt models/final/distilbert jest wpiekany w obraz (W4: z GitHub Release, sha256 w configu).
# Torch wyłącznie z indeksu CPU (extra `cpu`), bez CUDA; zależności główne zostają w obrazie
# (decyzja autora 15.10: akceptujemy i mierzymy rozmiar).

FROM ghcr.io/astral-sh/uv:0.10.4 AS uv

FROM python:3.11-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Najpierw same zależności — warstwa cache niezależna od kodu projektu.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra serving --extra cpu
# Potem pakiet projektu, instalacja nieedytowalna (kod w site-packages, bez src/ w runtime).
COPY README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable --extra serving --extra cpu

FROM python:3.11-slim AS runtime
ARG MODEL_VERSION=""
ARG API_PORT=8000
ENV PATH="/app/.venv/bin:$PATH" \
    BANKING77_CONFIG=/app/configs/config.yaml \
    MODEL_VERSION=${MODEL_VERSION} \
    API_PORT=${API_PORT} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs/ configs/
COPY models/final/distilbert/ models/final/distilbert/
USER app
EXPOSE ${API_PORT}
CMD ["sh", "-c", "exec uvicorn banking77_intent_ml.api.app:create_app --factory --host 0.0.0.0 --port ${API_PORT}"]
