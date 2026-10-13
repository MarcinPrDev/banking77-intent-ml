"""Pomiary kosztu CPU trzech zamrożonych ścieżek (W2 N2) — cienkie CLI.

Uruchomienie:  uv run python scripts/measure_cost.py   (zamknij inne obciążenia, w tym serwer API)
Dla każdej ścieżki i liczby wątków z `evaluation.cost`: latencja rozgrzana (p50/p95), zimny start
w nowych procesach i rozmiar artefaktu; jeden run MLflow na ścieżkę × wątki (tag `cost_study`).
Artefakty muszą być w `paths.models_dir` (eksport: scripts/export_model.py). Pomiar oficjalny raz:
gdy w MLflow są już runy tego badania, skrypt odmawia — powtórka po usunięciu runów w MLflow UI.
Logika: src/banking77_intent_ml/evaluation/cost.py i cost_worker.py.
"""

from __future__ import annotations

import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError, ids_sha256
from banking77_intent_ml.evaluation.cost import (
    STUDY_NAME,
    STUDY_TAG,
    artifact_bytes,
    log_cost_run,
    measure,
    sample_texts,
)
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models.factory import STRATEGIES


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    cfg = settings.evaluation.cost
    done = count_runs_with_tag(settings, STUDY_TAG, STUDY_NAME)
    if done:
        print(f"[FAIL] koszt: w MLflow jest już {done} runów badania {STUDY_NAME} — pomiar "
              "oficjalny był; powtórka po usunięciu runów w MLflow UI")
        sys.exit(1)
    missing = [name for name in STRATEGIES if not (settings.paths.models_dir / name).is_dir()]
    if missing:
        print(f"[FAIL] brak artefaktów w {settings.paths.models_dir}: {missing} — "
              "najpierw scripts/export_model.py")
        sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    sample = sample_texts(data, settings)
    sample_sha = ids_sha256(sample.index)
    print(f"{len(sample)} zdań z walidacji (seed {settings.models.seed}, id {sample_sha[:12]}) | "
          f"wątki {list(cfg.threads)} | rozgrzewka {cfg.warmup}, przebiegi {cfg.passes}, "
          f"zimne starty {cfg.cold_runs}")
    for name in sorted(STRATEGIES):
        size = artifact_bytes(name, settings)
        for threads in cfg.threads:
            result = measure(name, threads, sample.tolist(), settings)
            run_id = log_cost_run(
                result, settings, sample_ids_sha256=sample_sha, artifact_size=size
            )
            m = result.metrics
            print(f"  {name:11s} wątki {threads}: p50 {m['warm_p50_ms']:.2f} ms | "
                  f"p95 {m['warm_p95_ms']:.2f} ms | zimny start {m['cold_median_s']:.2f} s "
                  f"(max {m['cold_max_s']:.2f}) | {size / 1e6:.1f} MB | "
                  f"torch {result.torch_threads} | run {run_id[:8]}")


if __name__ == "__main__":
    main()
