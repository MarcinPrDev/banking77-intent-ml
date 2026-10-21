"""Testy UI (stretch 3): klient i strona Streamlit (ui/client.py, ui/app.py) — atrapa API.

Dowodzą: odpowiedź API mapuje się na intencję, id, score i wersję; 422 i brak połączenia dają
czytelny komunikat zamiast wyjątku; strona pokazuje stan `/health` i wynik `/predict` dla
wpisanego tekstu. Nie dowodzą wyglądu strony ani działania z prawdziwym API (to sprawdza podgląd).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")  # extra `ui`; w CI (tylko `dev`) moduł jest pomijany

from streamlit.testing.v1 import AppTest  # noqa: E402

from banking77_intent_ml.ui import client  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "banking77_intent_ml" / "ui" / "app.py"
HEALTH = {"status": "ok", "model": "distilbert", "model_version": "distilbert-v1"}
PREDICTION = {"intent": "card_arrival", "intent_id": 11, "score": 0.98,
              "model_version": "distilbert-v1"}


def _down(method, path, body):
    raise ConnectionRefusedError("usługa nie słucha")


def test_classify_maps_response_and_reports_errors():
    ok = client.classify(lambda m, p, b: (200, PREDICTION), "where is my card?")
    assert ok == client.Prediction("card_arrival", 11, 0.98, "distilbert-v1")
    assert "422" in client.classify(lambda m, p, b: (422, {"detail": []}), "").message
    assert "HTTP 500" in client.classify(lambda m, p, b: (500, None), "x").message
    assert "nie odpowiada" in client.classify(_down, "x").message


def test_health_returns_body_or_error():
    assert client.health(lambda m, p, b: (200, HEALTH)) == HEALTH
    assert "HTTP 503" in client.health(lambda m, p, b: (503, None)).message
    assert "nie odpowiada" in client.health(_down).message


def test_app_shows_health_and_prediction(monkeypatch):
    calls = []

    def fake(method, path, body):
        calls.append((method, path, body))
        return (200, HEALTH) if path == "/health" else (200, PREDICTION)

    monkeypatch.setattr(client, "request_factory", lambda url: fake)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "distilbert-v1" in at.sidebar.success[0].value

    at.text_area[0].input("where is my card?").run()
    at.button[0].click().run()

    assert at.metric[0].value == "card_arrival"
    assert ("POST", "/predict", {"text": "where is my card?"}) in calls
    assert not at.error


def test_app_shows_error_when_api_is_down(monkeypatch):
    monkeypatch.setattr(client, "request_factory", lambda url: _down)
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert "nie odpowiada" in at.sidebar.error[0].value
