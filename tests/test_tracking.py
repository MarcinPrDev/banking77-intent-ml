"""Testy logowania runu walidacyjnego do MLflow (evaluation/tracking.py) — tymczasowy magazyn.

Sprawdzają decyzje autora z 6.10: agregaty jako metryki, tabele per klasa i macierz pomyłek jako
artefakty w jawnie wskazanym katalogu, tagi protokołu (split=val, hash walidacji, commit danych);
oraz z 7.10: tag z wersją sklearn i metryki dodatkowe bez prawa nadpisania metryk runu.
"""

from __future__ import annotations

import pytest
import sklearn
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
    assert tags["sklearn_version"] == sklearn.__version__
    assert run.info.artifact_uri.startswith(settings.mlflow.artifact_location)
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert {PER_CLASS_ARTIFACT, CONFUSION_ARTIFACT} <= artifacts


def test_extra_metrics_are_logged(tmp_mlflow_settings):
    settings = tmp_mlflow_settings
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    run_id = log_validation_run(
        run_name="pytest-extra",
        params={"model": "test"},
        result=result,
        settings=settings,
        extra_metrics={"n_iter": 12},
    )

    metrics = MlflowClient(settings.mlflow.tracking_uri).get_run(run_id).data.metrics
    assert set(metrics) == {"accuracy", "macro_f1", "n_iter"}
    assert metrics["n_iter"] == 12


@pytest.mark.parametrize("name", ["accuracy", "fit_seconds"])
def test_extra_metrics_cannot_overwrite_run_metrics(tmp_mlflow_settings, name):
    settings = tmp_mlflow_settings
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="nadpisałyby"):
        log_validation_run(
            run_name="pytest-clash",
            params={"model": "test"},
            result=result,
            settings=settings,
            fit_seconds=0.1,
            extra_metrics={name: 0.0},
        )

    # błąd pada przed startem runu — w magazynie nie zostaje pół-zalogowany run
    client = MlflowClient(settings.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow.experiment)
    assert experiment is None or not client.search_runs([experiment.experiment_id])


def test_curves_are_logged_per_epoch(tmp_mlflow_settings):
    settings = tmp_mlflow_settings
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    run_id = log_validation_run(
        run_name="pytest-curves",
        params={"model": "test"},
        result=result,
        settings=settings,
        curves={"val_macro_f1": [0.5, 0.75, 0.875]},
    )

    history = MlflowClient(settings.mlflow.tracking_uri).get_metric_history(run_id, "val_macro_f1")
    assert [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)] == [
        (1, 0.5), (2, 0.75), (3, 0.875),
    ]


def test_curves_cannot_overwrite_run_metrics(tmp_mlflow_settings):
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="nadpisałyby"):
        log_validation_run(
            run_name="pytest-curves-clash",
            params={"model": "test"},
            result=result,
            settings=tmp_mlflow_settings,
            curves={"macro_f1": [0.5]},
        )
