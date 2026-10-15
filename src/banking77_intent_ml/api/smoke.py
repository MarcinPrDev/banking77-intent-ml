"""Smoke usługi (W3 N3, W4 CI): zdrowie, wersja i zdania kontrolne wobec działającego API.

Decyzje autora (15.10.2026, ADR-0003 + wybór wariantu (b)): w CI zdania kontrolne napisane na
potrzeby testu (nie ze zbioru danych) z oczekiwanymi wynikami ocenionego artefaktu —
`deploy.smoke_cases`, nagrywane przez `record_cases` z `deploy.artifact_dir`; test przeniesienia
na zdaniach z dawnej walidacji wykonany jednorazowo 15.10 (te same id w 50/50).

Sprawdza: `/health` (status, model `api.model`, `model_version` == `deploy.model_version`),
`/predict` dla każdego zdania (schemat odpowiedzi, id, nazwa intencji z mapy, score z tolerancją)
i 422 dla pustego tekstu. Klient HTTP przekazywany z zewnątrz: w CLI urllib (biblioteka
standardowa), w testach TestClient — ten sam kod wobec kontenera, kind i lokalnego uvicorna.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.labels import LabelMap
from banking77_intent_ml.models.factory import load_predictor

# (metoda, ścieżka, body) -> (kod HTTP, json odpowiedzi)
Request = Callable[[str, str, dict | None], tuple[int, Any]]
RESPONSE_KEYS = {"intent", "intent_id", "score", "model_version"}
SCORE_TOLERANCE = 1e-4  # ta sama wersja torcha na CPU: różnice ~1e-7 (test przeniesienia 15.10)

CONTROL_TEXTS = (  # zdania kontrolne napisane na potrzeby smoke — nie pochodzą z BANKING77
    "My new card still hasn't arrived, when will it come?",
    "What exchange rate will I get if I pay in euros?",
    "My top up didn't go through, what happened?",
    "I want to close my account for good.",
    "Someone stole my card, please block it right away.",
    "How do I change the PIN on my card?",
    "I was charged twice for the same purchase.",
    "Why is my bank transfer still pending?",
    "I'd like a refund for the item I bought.",
    "Is there a fee for withdrawing cash abroad?",
    "Why do you need to verify my identity?",
    "The ATM swallowed my card.",
    "Can I get a virtual card for online shopping?",
    "Which countries are you available in?",
)


def http_request(base_url: str, timeout_s: float = 30.0) -> Request:
    """Klient urllib dla `base_url` (np. http://localhost:8080); błędy HTTP zwraca jako kod."""
    def request(method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            base_url.rstrip("/") + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")
    return request


def wait_until_healthy(request: Request, timeout_s: float, interval_s: float = 1.0) -> dict:
    """Ponawia `GET /health` do odpowiedzi 200 albo przekroczenia `timeout_s` (TimeoutError)."""
    deadline = time.monotonic() + timeout_s
    last_error = "brak odpowiedzi"
    while time.monotonic() < deadline:
        try:
            status, body = request("GET", "/health", None)
            if status == 200:
                return body
            last_error = f"HTTP {status}"
        except OSError as exc:  # usługa jeszcze nie słucha (ConnectionRefused, URLError, timeout)
            last_error = str(exc)
        time.sleep(interval_s)
    raise TimeoutError(f"/health nie odpowiedział 200 w {timeout_s:.0f} s: {last_error}")


def record_cases(settings: Settings, texts: tuple[str, ...] = CONTROL_TEXTS) -> dict:
    """Oczekiwane wyniki ocenionego artefaktu (`deploy.artifact_dir`) dla zdań kontrolnych."""
    predictor = load_predictor(settings.api.model, settings, settings.deploy.artifact_dir)
    labels = LabelMap.from_settings(settings)
    scores = predictor.predict_proba(list(texts))
    cases = []
    for text, row in zip(texts, scores, strict=True):
        top2 = sorted(row, reverse=True)[:2]
        intent_id = int(row.argmax())
        cases.append({
            "text": text,
            "intent_id": intent_id,
            "intent": labels.name(intent_id),
            "score": float(row[intent_id]),
            "margin": float(top2[0] - top2[1]),  # informacyjnie: mały margines = ryzyko remisu
        })
    return {
        "model": settings.api.model,
        "model_version": settings.deploy.model_version,
        "score_tolerance": SCORE_TOLERANCE,
        "cases": cases,
    }


def run_smoke(request: Request, expected: dict, settings: Settings) -> list[str]:
    """Lista niezgodności (pusta = smoke przeszedł) dla `/health`, zdań kontrolnych i 422."""
    failures = []
    if expected["model_version"] != settings.deploy.model_version:
        failures.append(f"plik przypadków dla wersji {expected['model_version']}, config: "
                        f"{settings.deploy.model_version} — nagraj ponownie (--record)")
    status, health = request("GET", "/health", None)
    want = {"status": "ok", "model": settings.api.model,
            "model_version": settings.deploy.model_version}
    if status != 200 or health != want:
        failures.append(f"/health: HTTP {status}, {health} — oczekiwano {want}")
    labels = LabelMap.from_settings(settings)
    tolerance = expected["score_tolerance"]
    for case in expected["cases"]:
        status, body = request("POST", "/predict", {"text": case["text"]})
        where = f"/predict {case['text']!r}"
        if status != 200 or not isinstance(body, dict) or set(body) != RESPONSE_KEYS:
            failures.append(f"{where}: HTTP {status}, odpowiedź {body}")
            continue
        if body["intent_id"] != case["intent_id"]:
            failures.append(f"{where}: id {body['intent_id']}, oczekiwano {case['intent_id']}")
        if body["intent"] != labels.name(body["intent_id"]):
            failures.append(f"{where}: nazwa {body['intent']} niezgodna z mapą etykiet")
        if abs(body["score"] - case["score"]) > tolerance:
            failures.append(f"{where}: score {body['score']:.6f}, oczekiwano {case['score']:.6f}")
        if body["model_version"] != settings.deploy.model_version:
            failures.append(f"{where}: wersja {body['model_version']}")
    status, _ = request("POST", "/predict", {"text": "   "})
    if status != 422:
        failures.append(f"/predict pustego tekstu: HTTP {status}, oczekiwano 422")
    return failures
