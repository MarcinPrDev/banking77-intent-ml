"""FastAPI: `POST /predict` i `GET /health` (W2, N3) — predyktor `api.model` (Strategy).

Decyzje autora (13.10.2026): żądanie = jeden tekst `{"text": ...}`; odpowiedź top-1 — `intent`,
`intent_id`, `score`, `model_version`; tekst pusty, same spacje albo dłuższy niż
`api.max_text_chars` -> 422 (spacje na brzegach obcinane przed sprawdzeniem długości).
`score` = wynik modelu dla zwróconej klasy z `predict_proba` — jedno wywołanie modelu, bo kontrakt
gwarantuje `predict` == argmax `predict_proba`; to nie skalibrowane prawdopodobieństwo.
Predyktor i mapa etykiet ładowane raz, na starcie (lifespan). Artefakt z `deploy.artifact_dir`
(ADR-0003: DistilBERT po reficie, oceniony na teście); wersja modelu ze zmiennej środowiskowej
`api.model_version_env` (w obrazie), a bez niej z `deploy.model_version` — jedno źródło wersji.

Uruchomienie lokalne (port 8000 na tej maszynie zarezerwowany przez Windows):
`uv run uvicorn banking77_intent_ml.api.app:create_app --factory --port 8080`

Bez `from __future__ import annotations`: FastAPI czyta adnotacje endpointów w czasie działania,
a model żądania powstaje w `create_app` z limitem długości z configu.
"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints, create_model

from banking77_intent_ml.config import Settings, load_settings
from banking77_intent_ml.data.labels import LabelMap
from banking77_intent_ml.models.factory import load_predictor
from banking77_intent_ml.models.predictor import IntentClassifier


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
        return PredictResponse(
            intent=app.state.labels.name(intent_id),
            intent_id=intent_id,
            score=float(scores[intent_id]),
            model_version=app.state.model_version,
        )

    return app
