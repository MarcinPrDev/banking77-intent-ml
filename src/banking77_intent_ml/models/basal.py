"""Klient silnika Basal (`basal-serve`, POST /v1/systemone) — dodatek W5, zero-shot.

Jedno pytanie typu `choice` na wiadomość: `state` = tekst klienta, `criteria` = opisy 77 intencji
(klucze = nazwy z mapy etykiet). Silnik nie generuje tekstu — odpowiedź jest zawsze jednym
z kluczy, więc bez błędów formatu. Prawdopodobieństwa i `confidence` pochodzą z kalibracji
autora modelu, nie naszej. Basal działa w osobnym środowisku (transformers 5.17.0), a projekt
łączy się z nim przez HTTP (stdlib, bez nowej zależności). Teksty nie trafiają do logów.

Model serwowany sprawdzamy po nazwie: serwer startuje z `--name <repo>@<7 znaków rewizji>`
(`served_name`), a klient porównuje pole `model` każdej odpowiedzi.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from banking77_intent_ml.config.settings import BasalModelConfig

QUESTION = "intent"


class BasalError(RuntimeError):
    """Odpowiedź serwera niezgodna z pytaniem albo z kandydatem."""


@dataclass(frozen=True)
class BasalDecision:
    choice: str  # klucz z `criteria`
    probabilities: dict[str, float]  # klucz -> prawdopodobieństwo (kalibracja autora modelu)
    confidence: float
    model: str  # nazwa serwowanego modelu z odpowiedzi
    server_ms: float  # `usage.latency_ms` (czas po stronie serwera)
    wall_ms: float  # czas żądania HTTP po stronie klienta


class BasalClient(Protocol):
    def decide(self, text: str, criteria: dict[str, str], instructions: str) -> BasalDecision:
        ...


def served_name(model: BasalModelConfig) -> str:
    """Nazwa dla `basal-serve --name`: identyfikuje repo i przypiętą rewizję."""
    return f"{model.repo.split('/')[-1]}@{model.revision[:7]}"


def request_body(text: str, criteria: dict[str, str], instructions: str) -> dict:
    return {
        "state": text,
        "questions": {
            QUESTION: {"type": "choice", "instructions": instructions, "criteria": criteria},
        },
    }


def parse_decision(payload: dict, criteria: dict[str, str], wall_ms: float) -> BasalDecision:
    """Odpowiedź serwera -> decyzja; wybór spoza `criteria` albo inne klucze = `BasalError`."""
    try:
        answer = payload["answers"][QUESTION]
        choice = str(answer["choice"])
        probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
        confidence = float(answer["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BasalError(f"niepełna odpowiedź serwera: {exc!r}") from exc
    if choice not in criteria:
        raise BasalError(f"wybór {choice!r} spoza podanych opcji")
    if set(probabilities) != set(criteria):
        raise BasalError("prawdopodobieństwa dla innych opcji niż podane")
    usage = payload.get("usage") or {}
    return BasalDecision(
        choice=choice,
        probabilities=probabilities,
        confidence=confidence,
        model=str(payload.get("model", "")),
        server_ms=float(usage.get("latency_ms", float("nan"))),
        wall_ms=wall_ms,
    )


class HttpBasalClient:
    """Synchroniczne żądania do `basal-serve`; jedna wiadomość na żądanie (batch = 1)."""

    def __init__(self, url: str, timeout_s: float, expected_model: str, retries: int = 0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.expected_model = expected_model
        self.retries = retries  # tylko zerwane połączenie (np. tunel SSH); błąd HTTP od razu w górę

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def decide(self, text: str, criteria: dict[str, str], instructions: str) -> BasalDecision:
        body = json.dumps(request_body(text, criteria, instructions)).encode("utf-8")
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.url}/v1/systemone",
                data=body,
                headers={"content-type": "application/json"},
                method="POST",
            )
            start = time.perf_counter()  # czas udanej próby, bez przerw na ponowienie
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError:
                raise
            except OSError:
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)
        decision = parse_decision(payload, criteria, (time.perf_counter() - start) * 1000)
        if decision.model != self.expected_model:
            raise BasalError(
                f"serwer serwuje {decision.model!r}, oczekiwano {self.expected_model!r}"
            )
        return decision


@dataclass(frozen=True)
class BasalPredictions:
    ids: list[int]  # id klas z mapy etykiet
    scores: list[float]  # prawdopodobieństwo wybranej opcji
    server_ms: list[float]
    wall_ms: list[float]


def predict_texts(
    texts: Sequence[str],
    criteria: dict[str, str],
    instructions: str,
    client: BasalClient,
    label_map: dict[str, int],
) -> BasalPredictions:
    """Decyzje dla tekstów po kolei; klucze `criteria` to nazwy z `label_map`."""
    ids, scores, server_ms, wall_ms = [], [], [], []
    for text in texts:
        decision = client.decide(text, criteria, instructions)
        ids.append(label_map[decision.choice])
        scores.append(decision.probabilities[decision.choice])
        server_ms.append(decision.server_ms)
        wall_ms.append(decision.wall_ms)
    return BasalPredictions(ids, scores, server_ms, wall_ms)
