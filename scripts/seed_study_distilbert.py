"""Stretch 1: rozrzut zamrożonej konfiguracji DistilBERT po seedach na walidacji — cienkie CLI.

Uruchomienie:  uv run python scripts/seed_study_distilbert.py
Decyzje autora (18.10.2026, przed wynikiem): seedy `models.distilbert.seed_study` (1–5)
+ kontrolnie `models.seed` (musi odtworzyć zamrożony run W2), wynik każdego seeda po epoce
`frozen.epoch`, bez wyboru epoki; test nietknięty. Jedno badanie: gdy w MLflow są już runy
z tagiem `seed_study`, skrypt odmawia przed treningiem. Wniosek z rozrzutu pisze autor.
Logika: src/banking77_intent_ml/models/distilbert.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models.distilbert import (
    SEED_STUDY_NAME,
    SEED_STUDY_TAG,
    frozen_reference,
    run_seed_study,
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # jawnie (Windows)
    settings = load_settings()
    cfg = settings.models.distilbert
    done = count_runs_with_tag(settings, SEED_STUDY_TAG, SEED_STUDY_NAME)
    if done:
        print(f"[FAIL] Seedy DistilBERT: w MLflow jest już {done} runów badania "
              f"{SEED_STUDY_NAME} — budżet wykorzystany")
        sys.exit(1)
    reference = frozen_reference(settings)  # przed treningiem: bez runu W2 nie ma kontroli
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    seeds = (settings.models.seed, *cfg.seed_study)
    print(f"Seedy DistilBERT {SEED_STUDY_NAME}: lr {cfg.frozen['lr']:g}, {cfg.epochs} epok, "
          f"wynik po epoce {cfg.frozen['epoch']} | seedy {seeds} | train {len(data.y_train)} | "
          f"val {len(data.y_val)}")
    runs = run_seed_study(data, settings)
    selection = settings.evaluation.selection_metric
    primary = settings.evaluation.primary_metric
    for run in runs:
        print(f"  seed {run.seed:>3}: {selection} {run.metrics[selection]:.4f} | {primary} "
              f"{run.metrics[primary]:.4f} | trening {run.train_seconds:.0f} s | "
              f"run {run.run_id[:8]}")

    values = np.array([run.metrics[selection] for run in runs])
    print(f"\n{selection} po epoce {cfg.frozen['epoch']} ({len(values)} seedów): średnia "
          f"{values.mean():.4f}, odch. std {values.std(ddof=1):.4f}, min {values.min():.4f}, "
          f"max {values.max():.4f}")
    control = runs[0].metrics[selection]
    if control != reference:
        print(f"[UWAGA] kontrola: seed {runs[0].seed} = {control:.6f}, zamrożony run W2 = "
              f"{reference:.6f} — pętla badania nie odtwarza treningu W2")
        sys.exit(1)
    print(f"[OK] kontrola: seed {runs[0].seed} = {control:.6f} = zamrożony run W2")
    print("Wniosek z rozrzutu (np. względem MiniLM+kNN na walidacji) — autor, do README.")


if __name__ == "__main__":
    main()
