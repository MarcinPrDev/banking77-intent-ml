"""Modele bazowe na zamrożonej walidacji → runy w MLflow (cienkie CLI).

Uruchomienie — model wybierany jawnie (decyzja autora 7.10.2026: bez flagi skrypt odmawia, więc
ponowne uruchomienie nie dopisze przypadkiem duplikatów runów):
  uv run python scripts/train_baselines.py --model dummy      # Dummy x2 (most_frequent, stratified)
  uv run python scripts/train_baselines.py --model tfidf_lr   # TF-IDF + LR z configu
Podgląd runów (z korzenia repo): uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
Logika: src/banking77_intent_ml/models/baselines.py.
Decyzje — autor (6.10.2026, 7.10.2026).
"""

from __future__ import annotations

import argparse
import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.models.baselines import run_dummy_baselines, run_tfidf_lr

MODELS = ("dummy", "tfidf_lr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele bazowe: fit na train, ocena na zamrożonej walidacji, run w MLflow."
    )
    parser.add_argument("--model", required=True, choices=MODELS, help="model do treningu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    args = parse_args(argv)
    settings = load_settings()
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    print(f"train: {len(data.y_train)} | val: {len(data.y_val)} | "
          f"eksperyment MLflow: {settings.mlflow.experiment}")
    if args.model == "dummy":
        runs = run_dummy_baselines(data, settings)
    else:
        runs = [run_tfidf_lr(data, settings)]
    for run in runs:
        result = run.result
        print(f"{run.name:22s} accuracy {result.accuracy:.4f} | macro-F1 {result.macro_f1:.4f} "
              f"| run {run.run_id[:8]}")


if __name__ == "__main__":
    main()
