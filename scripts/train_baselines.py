"""Modele bazowe na zamrożonej walidacji → runy w MLflow (cienkie CLI).

Uruchomienie — model wybierany jawnie (decyzja autora 7.10.2026: bez flagi skrypt odmawia, więc
ponowne uruchomienie nie dopisze przypadkiem duplikatów runów):
  uv run python scripts/train_baselines.py --model dummy      # Dummy x2 (most_frequent, stratified)
  uv run python scripts/train_baselines.py --model tfidf_lr   # TF-IDF + LR z configu
  uv run python scripts/train_baselines.py --model tfidf_lr --save   # + artefakt w models/
`--save` (decyzja autora 9.10.2026): po runie zapisuje model do `paths.models_dir` razem
z metadanymi wskazującymi ten run — oficjalny zapis dopiero po commicie kodu (tag gita w runie).
Artefaktów nie nadpisujemy: gdy plik już istnieje, skrypt kończy się PRZED treningiem i runem.
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
from banking77_intent_ml.models.persistence import (
    METADATA_SUFFIX,
    MODEL_SUFFIX,
    build_metadata,
    save_model,
)

MODELS = ("dummy", "tfidf_lr")
SAVABLE = ("tfidf_lr",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele bazowe: fit na train, ocena na zamrożonej walidacji, run w MLflow."
    )
    parser.add_argument("--model", required=True, choices=MODELS, help="model do treningu")
    parser.add_argument(
        "--save", action="store_true", help="zapisz wytrenowany model do paths.models_dir"
    )
    args = parser.parse_args(argv)
    if args.save and args.model not in SAVABLE:
        parser.error(f"--save działa tylko z --model {' / '.join(SAVABLE)}")
    return args


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    args = parse_args(argv)
    settings = load_settings()
    save_path = settings.paths.models_dir / f"{args.model}{MODEL_SUFFIX}" if args.save else None
    if save_path is not None:
        existing = [p for p in (save_path, save_path.with_suffix(METADATA_SUFFIX)) if p.exists()]
        if existing:
            print(f"[FAIL] ZAPIS: artefakt już istnieje ({', '.join(map(str, existing))}) — "
                  "nie nadpisuję; usuń go świadomie albo zmień nazwę przed treningiem")
            sys.exit(1)
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

    if save_path is not None:
        (run,) = runs
        if run.model is None:
            print("[FAIL] ZAPIS: run nie zwrócił wytrenowanego modelu (BaselineRun.model)")
            sys.exit(1)
        metadata = build_metadata(model_name=args.model, run_id=run.run_id, settings=settings)
        print(f"model zapisany: {save_model(run.model, save_path, metadata=metadata)}")


if __name__ == "__main__":
    main()
