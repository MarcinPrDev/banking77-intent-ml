"""FastAPI: `POST /predict` i `GET /health` (W2, N3) — predyktor `api.model` (Strategy).

Decyzje autora (13.10.2026): żądanie = jeden tekst `{"text": ...}`; odpowiedź top-1 — `intent`,
`intent_id`, `score`, `model_version`; tekst pusty, same spacje albo dłuższy niż
`api.max_text_chars` -> 422 (spacje na brzegach obcinane przed sprawdzeniem długości).
`score` = wynik modelu dla zwróconej klasy z `predict_proba` — jedno wywołanie modelu, bo kontrakt
gwarantuje `predict` == argmax `predict_proba`; to nie skalibrowane prawdopodobieństwo.
Predyktor i mapa etykiet ładowane raz, na starcie (lifespan). Artefakt z `deploy.artifact_dir`
(ADR-0003: DistilBERT po reficie, oceniony na teście); wersja modelu ze zmiennej środowiskowej
`api.model_version_env` (w obrazie), a bez niej z `deploy.model_version` — jedno źródło wersji.

`GET /metrics` (dodatek W5, 29.10.2026): format Prometheusa — liczba i czas żądań według metody,
trasy i statusu (trasy spoza aplikacji jako `other`), predykcje według intencji i rozkład `score`.
Rozkład predykcji i score to sygnały zmiany ruchu, nie miara jakości. Rejestr domyślny jednego
procesu (kontener uruchamia jeden proces uvicorn); serwera Prometheus w klastrze nie ma.

Uruchomienie lokalne (port 8000 na tej maszynie zarezerwowany przez Windows):
`uv run uvicorn banking77_intent_ml.api.app:create_app --factory --port 18080`

Bez `from __future__ import annotations`: FastAPI czyta adnotacje endpointów w czasie działania,
a model żądania powstaje w `create_app` z limitem długości z configu.
"""

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, StringConstraints, create_model

from banking77_intent_ml.config import Settings, load_settings
from banking77_intent_ml.data.labels import LabelMap
from banking77_intent_ml.models.factory import load_predictor
from banking77_intent_ml.models.predictor import IntentClassifier

# Na poziomie modułu: kolejne `create_app` (np. w testach) używają tych samych metryk,
# zamiast rejestrować je drugi raz w rejestrze domyślnym.
REQUESTS = Counter(
    "banking77_http_requests_total", "Żądania HTTP", ["method", "path", "status"]
)
LATENCY = Histogram(
    "banking77_http_request_duration_seconds", "Czas obsługi żądania HTTP", ["method", "path"]
)
PREDICTIONS = Counter("banking77_predictions_total", "Predykcje według intencji", ["intent"])
SCORES = Histogram(
    "banking77_prediction_score",
    "Score zwróconej klasy (nieskalibrowany)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


class PredictResponse(BaseModel):
    intent: str
    intent_id: int
    score: float  # wynik modelu dla zwróconej klasy, nie skalibrowane prawdopodobieństwo
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str


def create_app(
    settings: Settings | None = None,
    loader: Callable[[str, Settings, Path], IntentClassifier] = load_predictor,
) -> FastAPI:
    """Aplikacja serwująca `api.model` z `deploy.artifact_dir`; `loader` podmieniany w testach."""
    settings = settings or load_settings()
    cfg = settings.api
    text_type = Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=cfg.max_text_chars)
    ]
    PredictRequest = create_model("PredictRequest", text=(text_type, ...))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = loader(cfg.model, settings, settings.deploy.artifact_dir)
        app.state.labels = LabelMap.from_settings(settings)
        app.state.model_version = (
            os.environ.get(cfg.model_version_env) or settings.deploy.model_version
        )
        yield

    app = FastAPI(title="BANKING77 intent API", lifespan=lifespan)

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model=cfg.model, model_version=app.state.model_version)

    @app.post("/predict")
    def predict(request: PredictRequest) -> PredictResponse:
        scores = app.state.predictor.predict_proba([request.text])[0]
        intent_id = int(scores.argmax())
        intent = app.state.labels.name(intent_id)
        score = float(scores[intent_id])
        PREDICTIONS.labels(intent=intent).inc()
        SCORES.observe(score)
        return PredictResponse(
            intent=intent,
            intent_id=intent_id,
            score=score,
            model_version=app.state.model_version,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    known_paths = frozenset(route.path for route in app.routes)  # bez etykiet z dowolnych URL

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.url.path if request.url.path in known_paths else "other"
        LATENCY.labels(method=request.method, path=path).observe(time.perf_counter() - start)
        REQUESTS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        return response

    return app
