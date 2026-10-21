"""Klient UI (stretch 3, W4): zapytania do API przez klienta HTTP ze smoke — model zostaje w API.

`request_factory` zwraca funkcję `(metoda, ścieżka, body) -> (status, json)`; testy ją podmieniają.
Błędy połączenia i odpowiedzi inne niż 200 wracają jako `ApiError` z komunikatem dla użytkownika.
"""

from __future__ import annotations

from dataclasses import dataclass

from banking77_intent_ml.api.smoke import Request, http_request


@dataclass(frozen=True)
class Prediction:
    intent: str
    intent_id: int
    score: float  # wynik modelu dla zwróconej klasy, nie skalibrowane prawdopodobieństwo
    model_version: str


@dataclass(frozen=True)
class ApiError:
    message: str


def request_factory(base_url: str) -> Request:
    return http_request(base_url, timeout_s=10.0)


def _unreachable(exc: OSError) -> ApiError:
    return ApiError(f"API nie odpowiada ({type(exc).__name__}) — uruchom API "
                    "(README → API lokalnie) albo zmień adres w panelu bocznym")


def health(request: Request) -> dict | ApiError:
    try:
        status, body = request("GET", "/health", None)
    except OSError as exc:
        return _unreachable(exc)
    return body if status == 200 else ApiError(f"/health: HTTP {status}")


def classify(request: Request, text: str) -> Prediction | ApiError:
    try:
        status, body = request("POST", "/predict", {"text": text})
    except OSError as exc:
        return _unreachable(exc)
    if status == 422:
        return ApiError("API odrzuciło tekst (pusty albo dłuższy niż limit) — HTTP 422")
    if status != 200:
        return ApiError(f"/predict: HTTP {status}")
    return Prediction(
        intent=body["intent"],
        intent_id=int(body["intent_id"]),
        score=float(body["score"]),
        model_version=body["model_version"],
    )
