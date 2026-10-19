"""LLM zero-shot (stretch 2, W4): klient LiteLLM, szablon promptu, parsowanie odpowiedzi, cache.

Decyzje autora (18.10.2026): model `llm.model` (Groq gpt-oss-20b), temperatura 0, LLM poza
`IntentClassifier` — kontrakt predyktora wymaga predict == argmax(predict_proba), a odpowiedź spoza
77 nazw ma zostać błędem formatu (`INVALID_ID`, w ocenie `evaluate(..., invalid=INVALID_ID)`), nie
klasą 0. Odpowiedź = dokładnie jedna nazwa etykiety po normalizacji. Szablony promptów z polami
`{labels}` i `{text}` pisze autor w `llm.prompts`.

`LLMClient` to wąski interfejs (DIP) — testy podstawiają atrapę, bez sieci. `CachedClient` trzyma
odpowiedzi w JSONL pod hashem modelu, parametrów i promptu (bez tekstów wiadomości), więc
przerwany przebieg można powtórzyć bez ponownego kosztu. Ocena: evaluation/llm_eval.py.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from banking77_intent_ml.config import Settings
from banking77_intent_ml.evaluation.metrics import class_names

INVALID_ID = -1  # odpowiedź spoza mapy etykiet (błąd formatu)
_STRIP = " \t\r\n\"'`"


@dataclass(frozen=True)
class LlmReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient(Protocol):
    def complete(self, prompt: str) -> LlmReply: ...


class LlmRateLimitError(RuntimeError):
    """Limit API, którego nie da się przeczekać w rozsądnym czasie (np. dzienny) — wznów później."""


_WAIT_HINT = re.compile(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+(?:ms|s))?)")
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def retry_after_s(error: Exception) -> float | None:
    """Czas oczekiwania z błędu 429: nagłówek `retry-after` albo „try again in 1m2.5s”."""
    headers = {key.lower(): value for key, value in (getattr(error, "headers", None) or {}).items()}
    if "retry-after" in headers:
        try:
            return float(headers["retry-after"])
        except ValueError:
            pass
    match = _WAIT_HINT.search(str(error))
    if not match or not match.group(1):
        return None
    parts = re.findall(r"([\d.]+)(h|ms|m|s)", match.group(1))
    return sum(float(amount) * _UNIT_S[unit] for amount, unit in parts)


class _TokenWindow:
    """Tokeny wysłane w ostatnich 60 s — dławienie do ~90% limitu tokenów na minutę."""

    def __init__(self, per_minute: int, clock: Callable[[], float], sleep: Callable[[float], None]):
        self._budget = 0.9 * per_minute
        self._clock, self._sleep = clock, sleep
        self._entries: deque[tuple[float, int]] = deque()

    def wait_for(self, tokens: int) -> None:
        if self._budget <= 0:
            return
        while True:
            now = self._clock()
            while self._entries and self._entries[0][0] <= now - 60:
                self._entries.popleft()
            if not self._entries or sum(t for _, t in self._entries) + tokens <= self._budget:
                return
            self._sleep(self._entries[0][0] + 60 - now)

    def add(self, tokens: int) -> None:
        self._entries.append((self._clock(), tokens))


class LiteLLMClient:
    """Jedno zapytanie = jedna wiadomość użytkownika (Groq: instrukcje w wiadomości użytkownika).

    Dławi tokeny do `llm.tokens_per_minute`; po 429 czeka wg `retry_after_s` (bez wskazówki:
    2, 4, 8… s) najwyżej `max_retries` razy; czekanie > `max_wait_s` (limit dzienny) =
    `LlmRateLimitError`. Latencja = czas udanego wywołania, bez czekania.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg = settings.llm
        self._sleep = sleep
        self._window = _TokenWindow(self._cfg.tokens_per_minute, clock, sleep)
        self._last_tokens = 0  # szacunek następnego zapytania = ostatnie zużycie

    def complete(self, prompt: str) -> LlmReply:
        import litellm  # extra `llm`; reszta projektu działa bez niego

        cfg = self._cfg
        params = {
            "temperature": cfg.temperature,
            "max_completion_tokens": cfg.max_completion_tokens,
            "include_reasoning": cfg.include_reasoning,
        }
        if cfg.reasoning_effort:
            params["reasoning_effort"] = cfg.reasoning_effort
        for attempt in range(cfg.max_retries + 1):
            self._window.wait_for(self._last_tokens)
            start = time.perf_counter()
            try:
                response = litellm.completion(
                    model=cfg.model, messages=[{"role": "user", "content": prompt}], **params
                )
            except litellm.RateLimitError as error:
                wait = retry_after_s(error)
                wait = min(2.0 ** (attempt + 1), cfg.max_wait_s) if wait is None else wait
                if attempt == cfg.max_retries or wait > cfg.max_wait_s:
                    raise LlmRateLimitError(
                        f"limit API: czekanie {wait:.0f} s (limit {cfg.max_wait_s:.0f} s), "
                        f"próba {attempt + 1}/{cfg.max_retries + 1} — wznów później; "
                        "odpowiedzi zostają w cache"
                    ) from error
                self._sleep(wait)
                continue
            latency_s = time.perf_counter() - start
            usage = response.usage
            reply = LlmReply(
                text=response.choices[0].message.content or "",
                prompt_tokens=int(usage.prompt_tokens),
                completion_tokens=int(usage.completion_tokens),
                latency_s=latency_s,
            )
            self._last_tokens = reply.prompt_tokens + reply.completion_tokens
            self._window.add(self._last_tokens)
            return reply
        raise AssertionError("nieosiągalne")  # pętla zawsze kończy się return albo raise


class CachedClient:
    """Odpowiedzi w pliku JSONL; klucz = SHA-256 (model, parametry, prompt), bez tekstów."""

    def __init__(self, client: LLMClient, path: Path, settings: Settings) -> None:
        self._client = client
        self._path = path
        cfg = settings.llm
        self._params = json.dumps(
            [cfg.model, cfg.temperature, cfg.reasoning_effort, cfg.include_reasoning,
             cfg.max_completion_tokens]
        )
        self._entries: dict[str, LlmReply] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                key = entry.pop("key")
                self._entries[key] = LlmReply(**entry)
        self.calls = 0  # zapytania wysłane do klienta (nie z cache)

    def complete(self, prompt: str) -> LlmReply:
        key = hashlib.sha256(f"{self._params}\0{prompt}".encode()).hexdigest()
        if key in self._entries:
            return self._entries[key]
        reply = self._client.complete(prompt)
        self.calls += 1
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, **asdict(reply)}, ensure_ascii=False) + "\n")
        self._entries[key] = reply
        return reply


def render_prompt(template: str, text: str, label_names: Sequence[str]) -> str:
    """Szablon autora z `{labels}` (nazwy po jednej w linii) i `{text}`; bez str.format."""
    missing = [name for name in ("{labels}", "{text}") if name not in template]
    if missing:
        raise ValueError(f"szablon promptu bez pól {missing}")
    return template.replace("{labels}", "\n".join(label_names)).replace("{text}", text)


def normalize_answer(answer: str) -> str:
    """Małe litery, bez spacji na brzegach, cudzysłowów, backticków i końcowej kropki."""
    text = answer.strip(_STRIP).removesuffix(".").strip(_STRIP)
    return text.lower()


def parse_label(answer: str, label_map: dict[str, int]) -> int:
    """Id etykiety, gdy odpowiedź (po normalizacji) = jedna z nazw; inaczej `INVALID_ID`."""
    lookup = {name.lower(): index for name, index in label_map.items()}
    return lookup.get(normalize_answer(answer), INVALID_ID)


@dataclass(frozen=True)
class LlmPredictions:
    ids: np.ndarray  # id etykiety albo INVALID_ID, w kolejności tekstów
    replies: list[LlmReply]


def predict_texts(
    texts: Sequence[str], template: str, client: LLMClient, label_map: dict[str, int]
) -> LlmPredictions:
    names = class_names(label_map)
    replies = [client.complete(render_prompt(template, text, names)) for text in texts]
    ids = np.array([parse_label(reply.text, label_map) for reply in replies], dtype=int)
    return LlmPredictions(ids=ids, replies=replies)


def usage_metrics(predictions: LlmPredictions, settings: Settings) -> dict[str, float]:
    """Błędy formatu, tokeny, latencja p50/p95 i koszt wg cennika z configu."""
    cfg = settings.llm
    prompt_tokens = sum(reply.prompt_tokens for reply in predictions.replies)
    completion_tokens = sum(reply.completion_tokens for reply in predictions.replies)
    latencies = np.array([reply.latency_s for reply in predictions.replies])
    return {
        "format_errors": float(np.sum(predictions.ids == INVALID_ID)),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "cost_usd": prompt_tokens / 1e6 * cfg.price_input_usd_per_mtok
        + completion_tokens / 1e6 * cfg.price_output_usd_per_mtok,
    }
