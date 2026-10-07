"""Modele bazowe BANKING77 na zamrożonej walidacji: Dummy (punkt odniesienia), w N1 TF-IDF + LR.

Trening leży w `models/` — katalogu, który sprawdza strażnik protokołu; dane wyłącznie przez
`load_train_val()` (train → fit, val → ocena). Dummy `stratified` losuje, więc seed z configu
(`models.seed`); `most_frequent` jest deterministyczny. Strategie podajemy jawnie — domyślna
w sklearn to `prior`.

TF-IDF + LR (N1): konfiguracja wyłącznie z configu (`models.tfidf_lr.tfidf` / `.logreg`;
decyzja autora 7.10.2026 — pierwsza konfiguracja = wartości domyślne sklearn 1.9, wpisane jawnie).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from banking77_intent_ml.config import Settings
from banking77_intent_ml.config.settings import TfidfLrConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import EvaluationResult, evaluate
from banking77_intent_ml.evaluation.tracking import log_validation_run

DUMMY_STRATEGIES = ("most_frequent", "stratified")


@dataclass(frozen=True)
class BaselineRun:
    name: str
    run_id: str
    result: EvaluationResult
    fit_seconds: float


def build_dummy(strategy: str, *, seed: int) -> DummyClassifier:
    """Dummy z jawną strategią i seedem z configu."""
    if strategy not in DUMMY_STRATEGIES:
        raise ValueError(f"nieznana strategia Dummy {strategy!r}; dozwolone: {DUMMY_STRATEGIES}")
    return DummyClassifier(strategy=strategy, random_state=seed)


def run_dummy_baselines(data: TrainVal, settings: Settings) -> list[BaselineRun]:
    """Dummy x 2: fit na train, predykcje na val, evaluate(), run MLflow dla każdej strategii."""
    runs = []
    for strategy in DUMMY_STRATEGIES:
        model = build_dummy(strategy, seed=settings.models.seed)
        start = time.perf_counter()
        model.fit(data.x_train, data.y_train)
        fit_seconds = time.perf_counter() - start
        result = evaluate(data.y_val, model.predict(data.x_val), data.label_map)
        name = f"dummy-{strategy}"
        run_id = log_validation_run(
            run_name=name,
            params={"model": "dummy", "strategy": strategy, "seed": settings.models.seed},
            result=result,
            settings=settings,
            tags={"model_family": "dummy"},
            fit_seconds=fit_seconds,
        )
        runs.append(BaselineRun(name=name, run_id=run_id, result=result, fit_seconds=fit_seconds))
    return runs


def build_tfidf_lr(config: TfidfLrConfig, *, seed: int) -> Pipeline:
    """Pipeline TF-IDF -> LogisticRegression z parametrami z configu (`models.tfidf_lr`).

    TODO(Marcin, N1). Kontrakt:
    - parametry kroków wyłącznie z `config.tfidf` i `config.logreg` — konfiguracja żyje
      w configu, nie w kodzie (słowników nie modyfikuj w miejscu);
    - `seed` = `models.seed` z configu (lbfgs go nie używa — gdzie i czy przekazać: Twoja decyzja);
    - zwraca nieuczony `Pipeline` — fit robi wywołujący, tylko na części treningowej.
    """
    raise NotImplementedError("N1: build_tfidf_lr — pisze autor")


def run_tfidf_lr(data: TrainVal, settings: Settings) -> BaselineRun:
    """TF-IDF + LR na zamrożonej walidacji: fit na train, evaluate() na val, run MLflow.

    TODO(Marcin, N1). Kontrakt (jak `run_dummy_baselines`):
    - fit wyłącznie na `data.x_train` / `data.y_train` — słownik i IDF tylko z train;
    - ocena przez `evaluate()` na `data.x_val` / `data.y_val`;
    - run przez `log_validation_run`: parametry obu kroków + seed, czas fitu, tag `model_family`;
    - zwraca `BaselineRun`.
    """
    raise NotImplementedError("N1: run_tfidf_lr — pisze autor")
