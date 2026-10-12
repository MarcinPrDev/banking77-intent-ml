"""DistilBERT fine-tuning na zamrożonej walidacji (W2, blok S3) — cienkie CLI.

Uruchomienie:  uv run python scripts/train_distilbert.py
Budżet (decyzja autora 12.10.2026): jedno badanie = wszystkie warianty z
`models.distilbert.variants`; gdy w MLflow są już runy tego badania (tag `hpo_study`), skrypt
odmawia przed treningiem. Wagi każdej epoki zostają w `paths.cache_dir/distilbert` (ok. 268 MB
na epokę). Zamrożenie zwycięzcy to wpis autora do `models.distilbert.frozen`; skrypt drukuje
fragment YAML i ścieżkę checkpointu do skopiowania do `paths.models_dir`.
Logika: src/banking77_intent_ml/models/distilbert.py.
"""

from __future__ import annotations

import os
import sys

import yaml

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models.distilbert import (
    STUDY_NAME,
    run_distilbert,
    select_best_distilbert,
)
from banking77_intent_ml.models.hpo import STUDY_TAG


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # jawnie (Windows)
    settings = load_settings()
    cfg = settings.models.distilbert
    done = count_runs_with_tag(settings, STUDY_TAG, STUDY_NAME)
    if done:
        print(f"[FAIL] DistilBERT: w MLflow jest już {done} runów badania {STUDY_NAME} — budżet "
              f"({len(cfg.variants)} wariantów) wykorzystany")
        sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    print(f"DistilBERT {STUDY_NAME}: {len(cfg.variants)} wariantów × {cfg.epochs} epok | "
          f"train {len(data.y_train)} | val {len(data.y_val)}")
    runs = run_distilbert(data, settings)
    for run in runs:
        curve = " ".join(f"{score.objective:.4f}" for score in run.epochs)
        print(f"  wariant {run.variant} (lr {run.lr:g}): cel po epokach {curve} | trening "
              f"{run.train_seconds:.0f} s | VRAM {run.peak_vram_gb:.2f} GB | run {run.run_id[:8]}")

    best = select_best_distilbert(
        [score for run in runs for score in run.epochs], tolerance=cfg.tie_tolerance
    )
    print(f"\nZwycięzca: wariant {best.variant} (lr {best.lr:g}), epoka {best.epoch}, "
          f"cel {best.objective:.4f} (run {best.run_id[:8]}).")
    print(f"Checkpoint: {best.checkpoint}")
    print("Do zamrożenia (decyzja autora) — wpis w configs/config.yaml, models.distilbert:")
    print(yaml.safe_dump({"frozen": {"lr": best.lr, "epoch": best.epoch}}, sort_keys=False))


if __name__ == "__main__":
    main()
