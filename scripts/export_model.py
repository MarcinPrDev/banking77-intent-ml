"""Eksport zamrożonej konfiguracji przez wspólny interfejs predyktora (W2 pkt 5) — cienkie CLI.

Uruchomienie:  uv run python scripts/export_model.py tfidf_lr
Trening na części train zamrożonego splitu — tej samej, na której konfigurację oceniono na
walidacji; refit na 10 003 to osobna decyzja przed testem (W3). Artefakt trafia do
`paths.models_dir/<nazwa>` (stamtąd czyta go API); istniejący katalog = odmowa przed wczytaniem
danych i treningiem — artefaktów nie nadpisujemy. Kontrola: accuracy i macro-F1 na walidacji
mają się zgadzać z runem zamrożonej konfiguracji w MLflow.
Logika: src/banking77_intent_ml/models/{factory,classifiers}.py.
"""

from __future__ import annotations

import argparse
import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.metrics import evaluate
from banking77_intent_ml.models.factory import STRATEGIES, build_predictor


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Eksport zamrożonej konfiguracji do models/")
    parser.add_argument("model", choices=sorted(STRATEGIES))
    args = parser.parse_args()
    settings = load_settings()
    target = settings.paths.models_dir / args.model
    if target.exists():
        print(f"[FAIL] artefakt już istnieje: {target} — nie nadpisuję")
        sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    predictor = build_predictor(args.model, settings).fit(data.x_train, data.y_train)
    result = evaluate(data.y_val, predictor.predict(data.x_val), data.label_map)
    predictor.save(target)
    print(f"Zapisano {target} (trening: {len(data.x_train)} zdań train) | walidacja "
          f"{len(data.x_val)}: accuracy {result.accuracy:.10f}, macro-F1 {result.macro_f1:.10f}")


if __name__ == "__main__":
    main()
