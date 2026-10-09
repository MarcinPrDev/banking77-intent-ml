"""HPO TF-IDF + LR na zamrożonej walidacji (W2, blok S1) — każda próba = osobny run MLflow.

Decyzje autora (9.10.2026): wariant B — Optuna TPE z seedem `models.seed`, budżet
`models.tfidf_lr.hpo_trials`; przestrzeń: reprezentacja {słowa (1,1), słowa (1,2), znaki
char_wb (2,4), znaki char_wb (3,5)}, `min_df` {1, 2}, `sublinear_tf`, `C` log-jednostajnie
0,1–100, `class_weight` {None, balanced}; cel = `evaluation.selection_metric` na walidacji;
remis (różnica < `hpo.tie_tolerance`) -> mniej cech TF-IDF.

Pętla (`run_tfidf_lr_hpo`) to infra: studium Optuny, run MLflow na próbę (`tfidf-lr-hpo-NN`,
tagi `hpo_study`, `hpo_trial`), zbiór prób do wyboru zwycięzcy. Przestrzeń, cel i wybór —
autor. Budżet pilnuje CLI: drugie badanie o tej samej nazwie = dodatkowe próby na tej samej
walidacji, więc `scripts/hpo_tfidf_lr.py` odmawia, gdy w MLflow są już próby tego badania.
"""

from __future__ import annotations

from dataclasses import dataclass

import optuna

from banking77_intent_ml.config import Settings
from banking77_intent_ml.config.settings import TfidfLrConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import EvaluationResult
from banking77_intent_ml.models.baselines import run_tfidf_lr

STUDY_NAME = "tfidf_lr_hpo_w2"  # wartość tagu `hpo_study` — jedno badanie = jeden budżet
STUDY_TAG = "hpo_study"


@dataclass(frozen=True)
class HpoTrial:
    number: int
    run_id: str
    config: TfidfLrConfig  # parametry próby (tfidf + logreg)
    result: EvaluationResult
    objective: float  # wartość celu, którą maksymalizowała Optuna
    n_features: int  # rozmiar słownika TF-IDF — kryterium remisu


def suggest_tfidf_lr(trial: optuna.Trial, settings: Settings) -> TfidfLrConfig:
    """Parametry jednej próby z przestrzeni `models.tfidf_lr.hpo.space`.

    TODO(Marcin, W2 S1). Kontrakt:
    - wszystkie wartości przestrzeni z configu (nic na sztywno w kodzie) — kształt `space`
      w YAML-u ustalasz razem z tą funkcją;
    - Optuna przyjmuje w `suggest_categorical` tylko str/int/float/bool/None, więc
      reprezentację podaj jako nazwę i zamień ją na `analyzer` + `ngram_range` (krotka!);
    - `C`: `trial.suggest_float(..., log=True)`;
    - zwraca kopię konfiguracji z configu (`dataclasses.replace`) z nowymi `tfidf`/`logreg`
      — słowników z configu nie modyfikuj w miejscu.
    """
    raise NotImplementedError("W2 S1: suggest_tfidf_lr — pisze autor")


def objective_value(result: EvaluationResult, settings: Settings) -> float:
    """Wartość, którą maksymalizuje Optuna dla jednej próby.

    TODO(Marcin, W2 S1). Kontrakt: metryka `evaluation.selection_metric` z configu, liczona
    przez wspólne `evaluate()` na walidacji (nigdy test).
    """
    raise NotImplementedError("W2 S1: objective_value — pisze autor")


def select_best(trials: list[HpoTrial], *, tolerance: float) -> HpoTrial:
    """Zwycięzca badania wg reguły autora.

    TODO(Marcin, W2 S1). Kontrakt: najwyższy `objective`; próby, które ustępują najlepszej
    o mniej niż `tolerance`, są remisem — wtedy wygrywa ta z najmniejszym `n_features`.
    """
    raise NotImplementedError("W2 S1: select_best — pisze autor")


def run_tfidf_lr_hpo(data: TrainVal, settings: Settings) -> list[HpoTrial]:
    """Badanie Optuny: `hpo_trials` prób, każda trenowana i logowana przez `run_tfidf_lr`."""
    base = settings.models.tfidf_lr
    if base.hpo is None or base.hpo.sampler != "tpe":
        raise ValueError("models.tfidf_lr.hpo: oczekiwano samplera 'tpe' (decyzja autora 9.10)")
    trials: list[HpoTrial] = []

    def objective(trial: optuna.Trial) -> float:
        config = suggest_tfidf_lr(trial, settings)
        run = run_tfidf_lr(
            data,
            settings,
            config=config,
            run_name=f"tfidf-lr-hpo-{trial.number:02d}",
            tags={STUDY_TAG: STUDY_NAME, "hpo_trial": str(trial.number)},
        )
        value = objective_value(run.result, settings)
        n_features = len(run.model.named_steps["tfidf"].vocabulary_)
        trial.set_user_attr("run_id", run.run_id)
        trials.append(HpoTrial(trial.number, run.run_id, config, run.result, value, n_features))
        return value

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.models.seed),
    )
    study.optimize(objective, n_trials=base.hpo_trials)
    return trials
