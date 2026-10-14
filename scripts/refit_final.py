"""Refit zamrożonych konfiguracji na 10 003 zdaniach przed oceną na teście (W3) — cienkie CLI.

Uruchomienie:  uv run python scripts/refit_final.py                     (trzy ścieżki)
               uv run python scripts/refit_final.py --model tfidf_lr    (wybrane, można powtórzyć)
Kolejność z decyzji autora 14.10: zapis decyzji (ADR-0002, README) -> tag `experiments-frozen`
-> refit -> jednorazowa ocena na teście. Skrypt odmawia przed wczytaniem danych, gdy brak tagu
zamrożenia, gdy finalny artefakt już istnieje albo gdy w MLflow jest już refit tej ścieżki.
Logika: src/banking77_intent_ml/models/refit.py, data/datasets.py (`full_train`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import full_train, load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models.factory import STRATEGIES
from banking77_intent_ml.models.refit import REFIT_TAG, refit

FROZEN_TAG = "experiments-frozen"
_REPO = Path(__file__).resolve().parents[1]


def _tag_exists(tag: str) -> bool:
    listed = subprocess.run(
        ["git", "tag", "--list", tag], cwd=_REPO, capture_output=True, text=True, check=False
    )
    return listed.stdout.strip() == tag


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Refit zamrożonych konfiguracji na 10 003")
    parser.add_argument("--model", action="append", choices=sorted(STRATEGIES))
    args = parser.parse_args()
    names = args.model or sorted(STRATEGIES)
    settings = load_settings()

    if not _tag_exists(FROZEN_TAG):
        print(f"[FAIL] brak tagu {FROZEN_TAG} — najpierw zamrożenie (ADR-0002, README, tag)")
        sys.exit(1)
    for name in names:
        target = settings.paths.final_models_dir / name
        if target.exists():
            print(f"[FAIL] {name}: finalny artefakt już istnieje ({target}) — nie nadpisuję")
            sys.exit(1)
        if count_runs_with_tag(settings, REFIT_TAG, name):
            print(f"[FAIL] {name}: w MLflow jest już refit tej ścieżki — oficjalny refit był")
            sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    x, y = full_train(data)
    print(f"Refit na {len(x)} zdaniach (train {len(data.x_train)} + dawna walidacja "
          f"{len(data.x_val)}), kolejność wierszy train.csv")
    for name in names:
        result = refit(name, x, y, settings)
        print(f"  {name:11s} fit {result.fit_seconds:.1f} s -> {result.directory} "
              f"| run {result.run_id[:8]}")


if __name__ == "__main__":
    main()
