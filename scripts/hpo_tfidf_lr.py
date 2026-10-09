"""HPO TF-IDF + LR na zamrożonej walidacji (W2, blok S1) — cienkie CLI.

Uruchomienie:  uv run python scripts/hpo_tfidf_lr.py
Budżet (decyzja autora 9.10.2026): jedno badanie = `models.tfidf_lr.hpo_trials` prób. Gdy w MLflow
są już próby tego badania (tag `hpo_study`), skrypt odmawia przed treningiem — kolejne próby na
tej samej walidacji zawyżałyby wynik zwycięzcy. Zamrożenie zwycięzcy to ręczny wpis autora do
`models.tfidf_lr.frozen` w configu; skrypt drukuje gotowy fragment YAML.
Logika: src/banking77_intent_ml/models/hpo.py.
"""

from __future__ import annotations

import sys

import optuna
import yaml

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models.hpo import STUDY_NAME, STUDY_TAG, run_tfidf_lr_hpo, select_best


def _yaml_params(params: dict) -> dict:
    """Krotki (np. `ngram_range`) jako listy — YAML nie ma krotek."""
    return {key: list(val) if isinstance(val, tuple) else val for key, val in params.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    tfidf_lr = settings.models.tfidf_lr
    done = count_runs_with_tag(settings, STUDY_TAG, STUDY_NAME)
    if done:
        print(f"[FAIL] HPO: w MLflow jest już {done} prób badania {STUDY_NAME} — budżet "
              f"({tfidf_lr.hpo_trials}) wykorzystany; kolejne próby na tej samej walidacji "
              "zawyżyłyby wynik zwycięzcy")
        sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"HPO {STUDY_NAME}: {tfidf_lr.hpo_trials} prób | train {len(data.y_train)} | "
          f"val {len(data.y_val)} | eksperyment MLflow: {settings.mlflow.experiment}")
    trials = run_tfidf_lr_hpo(data, settings)
    for trial in sorted(trials, key=lambda t: t.objective, reverse=True):
        print(f"  #{trial.number:02d} cel {trial.objective:.4f} | accuracy "
              f"{trial.result.accuracy:.4f} | cechy {trial.n_features:6d} | run {trial.run_id[:8]}")

    best = select_best(trials, tolerance=tfidf_lr.hpo.tie_tolerance)
    print(f"\nZwycięzca: próba #{best.number:02d} (run {best.run_id[:8]}), "
          f"cel {best.objective:.4f}, cechy {best.n_features}.")
    print("Do zamrożenia (decyzja autora) — wpis w configs/config.yaml, models.tfidf_lr:")
    frozen = {"tfidf": _yaml_params(best.config.tfidf), "logreg": _yaml_params(best.config.logreg)}
    print(yaml.safe_dump({"frozen": frozen}, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
