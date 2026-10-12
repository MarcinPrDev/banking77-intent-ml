"""MiniLM (zamrożone embeddingi) + kNN na zamrożonej walidacji (W2, blok S2) — cienkie CLI.

Uruchomienie:  uv run python scripts/knn_minilm.py
Embeddingi liczone raz (GPU, jeśli dostępne) i trzymane w `paths.cache_dir/minilm` — kolejne
uruchomienia biorą je z cache. Budżet: jedna siatka `k_values` × `weights` z configu; gdy w MLflow
są już runy tego badania (tag `hpo_study`), skrypt odmawia przed liczeniem. Zamrożenie zwycięzcy
to ręczny wpis autora do `models.minilm_knn.frozen`; skrypt drukuje gotowy fragment YAML.
Logika: src/banking77_intent_ml/models/knn.py, src/banking77_intent_ml/features/embeddings.py.
"""

from __future__ import annotations

import sys

import yaml

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.features.embeddings import load_or_compute
from banking77_intent_ml.models.hpo import STUDY_TAG
from banking77_intent_ml.models.knn import STUDY_NAME, run_knn_grid, select_best_knn


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    cfg = settings.models.minilm_knn
    done = count_runs_with_tag(settings, STUDY_TAG, STUDY_NAME)
    if done:
        print(f"[FAIL] kNN: w MLflow jest już {done} runów badania {STUDY_NAME} — budżet "
              f"(siatka {len(cfg.k_values)} × {len(cfg.weights)}) wykorzystany")
        sys.exit(1)
    try:
        data = load_train_val(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    embeddings = load_or_compute(data, settings)
    meta = embeddings.meta
    source = "z cache" if embeddings.from_cache else f"policzone na {meta['device']}"
    print(f"Embeddingi {cfg.model_id} @ {cfg.revision[:8]}: {source} | wymiar {meta['dim']} | "
          f"kodowanie train {meta['encode_train_seconds']:.1f} s | normy "
          f"{meta['norm_min']:.4f}–{meta['norm_max']:.4f}")

    trials = run_knn_grid(data, embeddings, settings)
    for trial in sorted(trials, key=lambda t: t.objective, reverse=True):
        print(f"  k={trial.n_neighbors:2d} {trial.weights:8s} cel {trial.objective:.4f} | "
              f"accuracy {trial.result.accuracy:.4f} | run {trial.run_id[:8]}")

    best = select_best_knn(trials)
    print(f"\nZwycięzca: k={best.n_neighbors}, {best.weights} (run {best.run_id[:8]}), "
          f"cel {best.objective:.4f}.")
    print("Do zamrożenia (decyzja autora) — wpis w configs/config.yaml, models.minilm_knn:")
    frozen = {"n_neighbors": best.n_neighbors, "weights": best.weights}
    print(yaml.safe_dump({"frozen": frozen}, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
