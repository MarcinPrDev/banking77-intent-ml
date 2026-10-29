"""Testy API (api/app.py, W2 N3) — TestClient z atrapą predyktora, bez artefaktu i bez torch.

Decyzje autora 13.10: jeden tekst, odpowiedź top-1 (`intent`, `intent_id`, `score`,
`model_version`), limit `api.max_text_chars`; pusty tekst, same spacje albo za długi -> 422.
Predyktor ładowany raz (lifespan). CI instaluje tylko extra `dev` — bez FastAPI moduł jest pomijany.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # wymagany przez TestClient
pytest.importorskip("prometheus_client")  # GET /metrics (extra `serving`)

from fastapi.testclient import TestClient  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

from banking77_intent_ml.api.app import create_app  # noqa: E402
from banking77_intent_ml.config import load_settings  # noqa: E402
from banking77_intent_ml.data.labels import LabelMap  # noqa: E402
from banking77_intent_ml.models.factory import STRATEGIES  # noqa: E402
from banking77_intent_ml.models.predictor import IntentClassifier  # noqa: E402

_WINNER = 11  # id klasy, którą atrapa zawsze wskazuje
_LIMIT = load_settings().api.max_text_chars


class _FakePredictor(IntentClassifier):
    """Klasa `_WINNER` z wynikiem 0,6, reszta po równo; zapamiętuje teksty wejściowe."""

    name = "fake"

    def __init__(self, settings):
        super().__init__(settings)
        self.seen: list[str] = []

    def fit(self, texts, labels):
        raise NotImplementedError

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)

    def predict_proba(self, texts):
        self.seen.extend(texts)
        n_classes = len(LabelMap.from_settings(self.settings))
        scores = np.full((len(texts), n_classes), 0.4 / (n_classes - 1))
        scores[:, _WINNER] = 0.6
        return scores

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory, settings):
        raise NotImplementedError


@pytest.fixture
def api(monkeypatch):
    """(klient, wywołania loadera, ustawienia) — aplikacja z atrapą zamiast artefaktu."""
    settings = load_settings()
    loads = []

    def loader(name, settings, directory):
        loads.append((name, directory))
        return _FakePredictor(settings)

    monkeypatch.setenv(settings.api.model_version_env, "test-1")
    with TestClient(create_app(settings, loader=loader)) as client:
        yield client, loads, settings


def test_health_reports_model_and_version(api):
    client, _, settings = api

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model": settings.api.model, "model_version": "test-1",
    }


def test_predict_returns_top1_intent_from_label_map(api):
    client, _, settings = api

    response = client.post("/predict", json={"text": "  my card has not arrived yet  "})

    assert response.status_code == 200
    body = response.json()
    assert body.pop("score") == pytest.approx(0.6)
    assert body == {
        "intent": LabelMap.from_settings(settings).name(_WINNER),
        "intent_id": _WINNER,
        "model_version": "test-1",
    }
    assert client.app.state.predictor.seen == ["my card has not arrived yet"]  # bez spacji


def test_predictor_is_loaded_once(api):
    client, loads, settings = api

    for _ in range(3):
        assert client.post("/predict", json={"text": "top up failed"}).status_code == 200

    assert loads == [(settings.api.model, settings.deploy.artifact_dir)]


def test_text_at_limit_is_accepted_after_strip(api):
    client, *_ = api

    response = client.post("/predict", json={"text": f"  {'a' * _LIMIT}  "})

    assert response.status_code == 200


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": "a" * (_LIMIT + 1)},
    {"text": 5},
    {},
])
def test_invalid_text_is_rejected(api, payload):
    client, *_ = api

    assert client.post("/predict", json=payload).status_code == 422
    assert client.app.state.predictor.seen == []  # model nie był wołany


def test_version_falls_back_to_deploy_config_without_env(monkeypatch):
    settings = load_settings()
    monkeypatch.delenv(settings.api.model_version_env, raising=False)
    app = create_app(settings, loader=lambda name, s, directory: _FakePredictor(s))

    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == settings.deploy.model_version


def _sample(name: str, **labels: str) -> float:
    """Wartość metryki z rejestru domyślnego (wspólna dla testów, więc porównujemy przyrosty)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_text(api):
    client, *_ = api
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "banking77_http_requests_total" in response.text


def test_predict_updates_intent_counter_and_score_histogram(api):
    client, _, settings = api
    intent = LabelMap.from_settings(settings).name(_WINNER)
    before = _sample("banking77_predictions_total", intent=intent)
    scores_before = _sample("banking77_prediction_score_count")

    assert client.post("/predict", json={"text": "top up failed"}).status_code == 200

    assert _sample("banking77_predictions_total", intent=intent) == before + 1
    assert _sample("banking77_prediction_score_count") == scores_before + 1


def test_requests_are_counted_by_route_and_status(api):
    client, *_ = api
    rejected = {"method": "POST", "path": "/predict", "status": "422"}
    unknown = {"method": "GET", "path": "other", "status": "404"}  # dowolny URL nie tworzy etykiety
    predictions_before = _sample("banking77_prediction_score_count")
    before = [_sample("banking77_http_requests_total", **labels) for labels in (rejected, unknown)]

    client.post("/predict", json={"text": "   "})
    client.get("/no-such-path")

    after = [_sample("banking77_http_requests_total", **labels) for labels in (rejected, unknown)]
    assert after == [before[0] + 1, before[1] + 1]
    assert _sample("banking77_prediction_score_count") == predictions_before  # 422 bez predykcji


def test_deploy_config_serves_evaluated_artifact_of_api_model():
    """ADR-0003: API serwuje oceniony artefakt z refitu — katalog = strategia `api.model`."""
    settings = load_settings()

    assert settings.api.model in STRATEGIES
    assert settings.deploy.artifact_dir == settings.paths.final_models_dir / settings.api.model
    assert settings.deploy.model_version
