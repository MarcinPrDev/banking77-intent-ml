"""Ocena końcowa na teście — raz na zamrożoną ścieżkę (W3 §1.2, ADR-0002) — cienkie CLI.

Uruchomienie:  uv run python scripts/evaluate_test.py                    (trzy ścieżki)
               uv run python scripts/evaluate_test.py --model tfidf_lr   (wybrane, można powtórzyć)
Odmawia przed wczytaniem testu: bez tagu `experiments-frozen`, bez finalnego artefaktu
(`scripts/refit_final.py`) albo gdy ścieżka ma już run `split=test`. Wypisuje tylko metryki
3 080 i 3 055 — wynik na 385 jest w MLflow, w raporcie dopiero razem z LLM. Teksty testowe
nie są wypisywane ani zapisywane.
Logika: src/banking77_intent_ml/evaluation/final.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError
from banking77_intent_ml.evaluation.final import (
    count_test_runs,
    load_test_set,
    log_test_run,
    predict_test,
    subset_results,
)
from banking77_intent_ml.models.factory import STRATEGIES, load_predictor

FROZEN_TAG = "experiments-frozen"
_REPO = Path(__file__).resolve().parents[1]


def _tag_exists(tag: str) -> bool:
    listed = subprocess.run(
        ["git", "tag", "--list", tag], cwd=_REPO, capture_output=True, text=True, check=False
    )
    return listed.stdout.strip() == tag


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Ocena końcowa na teście (raz na ścieżkę)")
    parser.add_argument("--model", action="append", choices=sorted(STRATEGIES))
    args = parser.parse_args()
    names = args.model or sorted(STRATEGIES)
    settings = load_settings()

    if not _tag_exists(FROZEN_TAG):
        print(f"[FAIL] brak tagu {FROZEN_TAG} — test dopiero po zamrożeniu i reficie")
        sys.exit(1)
    for name in names:
        if not (settings.paths.final_models_dir / name).is_dir():
            print(f"[FAIL] {name}: brak finalnego artefaktu — najpierw scripts/refit_final.py")
            sys.exit(1)
        if count_test_runs(settings, name):
            print(f"[FAIL] {name}: w MLflow jest już run split=test — ocena końcowa była")
            sys.exit(1)
    try:
        test = load_test_set(settings)
    except (DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] DANE: {exc}")
        sys.exit(1)

    print(f"Test: {len(test.x)} zdań | podzbiory {len(test.clean_ids)} i {len(test.llm_ids)} "
          "(hashe zgodne z manifestem)")
    for name in names:
        directory = settings.paths.final_models_dir / name
        predictions = predict_test(load_predictor(name, settings, directory), test)
        results = subset_results(predictions, test)
        run_id = log_test_run(name, predictions, results, settings, artifact_dir=directory)
        full, clean = results["test"], results["clean"]
        print(f"  {name:11s} 3 080: accuracy {full.accuracy:.4f}, macro-F1 {full.macro_f1:.4f} | "
              f"3 055: accuracy {clean.accuracy:.4f}, macro-F1 {clean.macro_f1:.4f} "
              f"| run {run_id[:8]}")


if __name__ == "__main__":
    main()
