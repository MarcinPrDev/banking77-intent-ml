"""Refit zamrożonych konfiguracji na całym train (10 003) przed oceną na teście — W3.

Decyzja autora (14.10.2026, ADR-0002): każda zamrożona konfiguracja uczona od nowa przez wspólny
interfejs predyktora na train + dawnej walidacji (`data.datasets.full_train`); hiperparametry
wyłącznie z `models.<nazwa>.frozen` — TF-IDF+LR: nowy słownik, IDF i klasyfikator; MiniLM+kNN:
zamrożony enkoder, indeks na 10 003; DistilBERT: od bazowych wag, harmonogram na `epochs`
z zatrzymaniem po `frozen.epoch`. Po reficie nie ma walidacji — świadoma cena porównywalności
z protokołem Full. Artefakt: `paths.final_models_dir/<nazwa>` (artefakty W2 zostają).

Każdy refit = run MLflow (tag `final_refit`) z czasem `fit` adaptera (z wczytaniem wag
lub enkodera) i hashem id 10 003 wierszy; oficjalny refit raz na ścieżkę.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.splits import ids_sha256
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.factory import build_predictor

REFIT_TAG = "final_refit"  # wartość = nazwa strategii; blokada drugiego refitu w CLI
_TORCH_STRATEGIES = frozenset({"minilm_knn", "distilbert"})  # GPU w `fit`, gdy dostępne


@dataclass(frozen=True)
class RefitResult:
    model: str
    directory: Path
    fit_seconds: float
    run_id: str


def _device(name: str) -> str:
    """Gdzie liczył `fit`: enkoder MiniLM i DistilBERT na GPU, gdy dostępne; TF-IDF+LR na CPU.

    Nie z `sys.modules`: w jednym procesie torch bywa już zaimportowany przez inną ścieżkę.
    """
    if name not in _TORCH_STRATEGIES:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def refit(name: str, x: pd.Series, y: pd.Series, settings: Settings) -> RefitResult:
    """Fit zamrożonej konfiguracji `name` na (x, y), zapis do `final_models_dir`, run MLflow."""
    directory = settings.paths.final_models_dir / name
    if directory.exists():
        raise FileExistsError(f"{directory} istnieje — finalnego artefaktu nie nadpisujemy")
    predictor = build_predictor(name, settings)
    start = time.perf_counter()
    predictor.fit(x, y)
    fit_seconds = time.perf_counter() - start
    predictor.save(directory)

    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"final-{name}") as run:
        mlflow.set_tags({
            "model_family": name,
            REFIT_TAG: name,
            "stage": "final",
            "split": "train+val",
            "data_commit": settings.dataset.commit,
            "sha256_train_ids": manifest["sha256_train_ids"],
            "sha256_val_ids": manifest["sha256_val_ids"],
        })
        mlflow.log_params({
            "n_train": len(x),
            "sha256_refit_ids": ids_sha256(x.index),
            "frozen": json.dumps(getattr(settings.models, name).frozen, sort_keys=True),
            "device": _device(name),
            "artifact_dir": str(directory),
        })
        mlflow.log_metrics({"fit_seconds": fit_seconds})
        run_id = run.info.run_id
    return RefitResult(name, directory, fit_seconds, run_id)
