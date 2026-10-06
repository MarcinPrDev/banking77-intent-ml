"""Logowanie runu ewaluacji na walidacji do MLflow — wspólne dla wszystkich modeli BANKING77.

Decyzje projektowe autora (6.10.2026):
agregaty (accuracy, macro-F1) jako metryki MLflow — do porównań między runami w UI; raport per
klasa i macierz pomyłek jako tabele-artefakty (`log_table`); przy każdym runie tagi protokołu:
`split=val`, `sha256_val_ids` z manifestu, commit danych, N-train i N-val.

Magazyn (decyzja autora 6.10.2026): runy w bazie SQLite (`paths.mlflow_db`), artefakty w `mlruns/`
— oba jako bezwzględne URI z settings. MLflow 3.14 odrzuca magazyn plikowy, a bez jawnego URI
zapisuje do `mlflow.db` w bieżącym katalogu (np. w notebooks/), czyli do drugiego magazynu.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from banking77_intent_ml.config import Settings
from banking77_intent_ml.evaluation.metrics import EvaluationResult

PER_CLASS_ARTIFACT = "per_class_val.json"  # log_table przyjmuje tylko .json albo .parquet
CONFUSION_ARTIFACT = "confusion_matrix_val.json"


def configure_mlflow(settings: Settings) -> None:
    """Jeden magazyn dla wszystkich skryptów i notebooków + eksperyment z configu.

    Eksperyment powstaje przy pierwszym użyciu z jawnym, bezwzględnym `artifact_location`,
    więc artefakty nie zależą od katalogu, z którego uruchomiono proces.
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    client = MlflowClient(settings.mlflow.tracking_uri)
    if client.get_experiment_by_name(settings.mlflow.experiment) is None:
        client.create_experiment(
            settings.mlflow.experiment, artifact_location=settings.mlflow.artifact_location
        )
    mlflow.set_experiment(settings.mlflow.experiment)


def protocol_tags(settings: Settings) -> dict[str, str]:
    """Tagi wiążące run z zamrożonym protokołem: manifest splitu i przypięte dane."""
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    return {
        "split": "val",
        "sha256_val_ids": manifest["sha256_val_ids"],
        "data_commit": settings.dataset.commit,
        "n_train": str(manifest["n_train"]),
        "n_val": str(manifest["n_val"]),
    }


def log_validation_run(
    *,
    run_name: str,
    params: dict[str, Any],
    result: EvaluationResult,
    settings: Settings,
    tags: dict[str, str] | None = None,
    fit_seconds: float | None = None,
) -> str:
    """Run MLflow: parametry, metryki z configu (+ czas fitu), tagi protokołu, tabele; zwraca id."""
    configured = {settings.evaluation.primary_metric, settings.evaluation.selection_metric}
    missing = configured - set(result.metrics())
    if missing:
        raise ValueError(f"evaluate() nie liczy metryk wskazanych w configu: {sorted(missing)}")

    configure_mlflow(settings)
    run_tags = {**protocol_tags(settings), **(tags or {})}
    with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
        mlflow.log_params(params)
        metrics = result.metrics()
        if fit_seconds is not None:
            metrics["fit_seconds"] = fit_seconds
        mlflow.log_metrics(metrics)
        mlflow.log_table(result.per_class.reset_index(), artifact_file=PER_CLASS_ARTIFACT)
        mlflow.log_table(result.confusion.reset_index(), artifact_file=CONFUSION_ARTIFACT)
        return run.info.run_id
