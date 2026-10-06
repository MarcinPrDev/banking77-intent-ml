"""Testy logowania runu walidacyjnego do MLflow (evaluation/tracking.py) — tymczasowy magazyn.

Sprawdzają decyzje autora z 6.10: agregaty jako metryki, tabele per klasa i macierz pomyłek jako
artefakty w jawnie wskazanym katalogu, tagi protokołu (split=val, hash walidacji, commit danych).
"""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.evaluation.metrics import evaluate
from banking77_intent_ml.evaluation.tracking import (
    CONFUSION_ARTIFACT,
    PER_CLASS_ARTIFACT,
    log_validation_run,
)


def test_run_has_metrics_tags_and_tables(tmp_mlflow_settings):
    settings = tmp_mlflow_settings
    result = evaluate([0, 1, 2], [0, 1, 1], {"a": 0, "b": 1, "c": 2})
    run_id = log_validation_run(
        run_name="pytest-run",
        params={"model": "test"},
        result=result,
        settings=settings,
        tags={"model_family": "test"},
        fit_seconds=0.25,
    )

    client = MlflowClient(settings.mlflow.tracking_uri)
    run = client.get_run(run_id)
    assert set(run.data.metrics) == {"accuracy", "macro_f1", "fit_seconds"}
    assert run.data.metrics["accuracy"] == pytest.approx(2 / 3)
    assert run.data.params["model"] == "test"
    tags = run.data.tags
    assert tags["split"] == "val" and tags["model_family"] == "test"
    assert len(tags["sha256_val_ids"]) == 64 and tags["data_commit"] == settings.dataset.commit
    assert run.info.artifact_uri.startswith(settings.mlflow.artifact_location)
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert {PER_CLASS_ARTIFACT, CONFUSION_ARTIFACT} <= artifacts
