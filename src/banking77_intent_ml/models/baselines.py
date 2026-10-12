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

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    model: BaseEstimator | None = None  # wytrenowany model — potrzebny do zapisu (`--save`, N2)


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

    Kontrakt:
    - parametry kroków wyłącznie z `config.tfidf` i `config.logreg` — konfiguracja żyje
      w configu, nie w kodzie (słowników nie modyfikuj w miejscu);
    - `seed` trafia do `random_state` regresji (lbfgs go nie używa);
    - zwraca nieuczony `Pipeline` — fit robi wywołujący, tylko na części treningowej.
    """
    logreg_params = {**config.logreg, "random_state": seed}
    return Pipeline([
        ("tfidf", TfidfVectorizer(**config.tfidf)),
        ("logreg", LogisticRegression(**logreg_params)),
    ])


def run_tfidf_lr(
    data: TrainVal,
    settings: Settings,
    *,
    config: TfidfLrConfig | None = None,
    run_name: str = "tfidf-lr",
    tags: dict[str, str] | None = None,
) -> BaselineRun:
    """TF-IDF + LR na zamrożonej walidacji: fit na train, evaluate() na val, run MLflow.

    Kontrakt (jak `run_dummy_baselines`):
    - fit wyłącznie na `data.x_train` / `data.y_train` — słownik i IDF tylko z train;
    - ocena przez `evaluate()` na `data.x_val` / `data.y_val`;
    - run przez `log_validation_run`: parametry obu kroków + seed, czas fitu, tag `model_family`;
    - zwraca `BaselineRun` z wytrenowanym pipeline'em w polu `model`.

    `config`, `run_name`, `tags` (W2, HPO): próba z innymi parametrami niż w configu; bez nich
    — pierwsza konfiguracja z `models.tfidf_lr` (N1), jak dotąd.
    """
    config = config or settings.models.tfidf_lr
    model = build_tfidf_lr(config, seed=settings.models.seed)
    start = time.perf_counter()
    model.fit(data.x_train, data.y_train)
    fit_seconds = time.perf_counter() - start
    result = evaluate(data.y_val, model.predict(data.x_val), data.label_map)
    name = run_name
    run_id = log_validation_run(
        run_name=name,
        params={
            "model": "tfidf_lr",
            "seed": settings.models.seed,
            **{f"tfidf__{key}": value for key, value in config.tfidf.items()},
            **{f"logreg__{key}": value for key, value in config.logreg.items()},
        },
        result=result,
        settings=settings,
        tags={"model_family": "tfidf_lr", **(tags or {})},
        fit_seconds=fit_seconds,
        extra_metrics={
            "n_iter": int(model.named_steps["logreg"].n_iter_[0]),
            "n_features": len(model.named_steps["tfidf"].vocabulary_),
        },
    )
    return BaselineRun(
        name=name, run_id=run_id, result=result, fit_seconds=fit_seconds, model=model
    )
