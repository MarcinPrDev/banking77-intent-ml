"""Modele bazowe na zamrożonej walidacji → runy w MLflow (cienkie CLI).

Uruchomienie: uv run python scripts/train_baselines.py
Podgląd runów (z korzenia repo): uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
Logika: src/banking77_intent_ml/models/baselines.py (w N1 dojdzie TF-IDF + LR).
Decyzje — autor (6.10.2026).
"""

from __future__ import annotations

import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.models.baselines import run_dummy_baselines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    print(f"train: {len(data.y_train)} | val: {len(data.y_val)} | "
          f"eksperyment MLflow: {settings.mlflow.experiment}")
    for run in run_dummy_baselines(data, settings):
        result = run.result
        print(f"{run.name:22s} accuracy {result.accuracy:.4f} | macro-F1 {result.macro_f1:.4f} "
              f"| run {run.run_id[:8]}")


if __name__ == "__main__":
    main()
