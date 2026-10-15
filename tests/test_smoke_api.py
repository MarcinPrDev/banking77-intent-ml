"""Testy smoke API (api/smoke.py) — TestClient z atrapą predyktora jako klient HTTP, bez serwera.

Smoke przechodzi wobec zgodnej usługi, wykrywa zmienione id, score i wersję, nagrywa przypadki
dla zdań kontrolnych; czekanie na `/health` ponawia próby i kończy się limitem czasu.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # wymagany przez TestClient

from fastapi.testclient import TestClient  # noqa: E402

from banking77_intent_ml.api import smoke  # noqa: E402
from banking77_intent_ml.api.app import create_app  # noqa: E402
from banking77_intent_ml.config import load_settings  # noqa: E402
from banking77_intent_ml.data.labels import LabelMap  # noqa: E402
from banking77_intent_ml.models.predictor import IntentClassifier  # noqa: E402


class _ByTextPredictor(IntentClassifier):
    """Klasa zależna od długości tekstu (deterministycznie), wynik 0,9 dla zwycięskiej klasy."""

    name = "by_text"

    def fit(self, texts, labels):
        raise NotImplementedError

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)

    def predict_proba(self, texts):
        scores = np.full((len(texts), 77), 0.1 / 76)
        for row, text in enumerate(texts):
            scores[row, len(text) % 77] = 0.9
        return scores

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory, settings):
        raise NotImplementedError


@pytest.fixture
def service(monkeypatch):
    """(klient HTTP jak `smoke.Request`, ustawienia, oczekiwane wyniki z tej samej atrapy)."""
    settings = load_settings()
    monkeypatch.delenv(settings.api.model_version_env, raising=False)  # wersja z configu
    monkeypatch.setattr(smoke, "load_predictor", lambda name, s, d: _ByTextPredictor(s))
    expected = smoke.record_cases(settings)
    app = create_app(settings, loader=lambda name, s, directory: _ByTextPredictor(s))
    with TestClient(app) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            return response.status_code, response.json()

        yield request, settings, expected


def test_smoke_passes_against_matching_service(service):
    request, settings, expected = service

    assert smoke.run_smoke(request, expected, settings) == []


def test_smoke_reports_changed_id_and_score(service):
    request, settings, expected = service
    expected["cases"][0]["intent_id"] = (expected["cases"][0]["intent_id"] + 1) % 77
    expected["cases"][1]["score"] += 0.01

    failures = smoke.run_smoke(request, expected, settings)

    assert len(failures) == 2
    assert "oczekiwano" in failures[0] and "score" in failures[1]


def test_smoke_reports_service_version_other_than_config(service, monkeypatch):
    _, settings, expected = service
    monkeypatch.setenv(settings.api.model_version_env, "inna-wersja")
    app = create_app(settings, loader=lambda name, s, directory: _ByTextPredictor(s))
    with TestClient(app) as client:
        def request(method, path, body=None):
            response = client.request(method, path, json=body)
            return response.status_code, response.json()

        failures = smoke.run_smoke(request, expected, settings)

    assert any(failure.startswith("/health") for failure in failures)
    assert any("wersja inna-wersja" in failure for failure in failures)


def test_record_cases_covers_control_texts_with_label_names(service):
    _, settings, expected = service
    labels = LabelMap.from_settings(settings)

    assert expected["model_version"] == settings.deploy.model_version
    assert [case["text"] for case in expected["cases"]] == list(smoke.CONTROL_TEXTS)
    for case in expected["cases"]:
        assert case["intent"] == labels.name(case["intent_id"])
        assert case["margin"] >= 0


def test_wait_until_healthy_retries_then_times_out():
    calls = []

    def flaky(method, path, body):
        calls.append(path)
        if len(calls) < 3:
            raise ConnectionRefusedError("usługa jeszcze nie słucha")
        return 200, {"status": "ok"}

    assert smoke.wait_until_healthy(flaky, timeout_s=5, interval_s=0.01) == {"status": "ok"}
    assert calls == ["/health"] * 3
    with pytest.raises(TimeoutError, match="HTTP 503"):
        smoke.wait_until_healthy(lambda *args: (503, None), timeout_s=0.05, interval_s=0.01)
