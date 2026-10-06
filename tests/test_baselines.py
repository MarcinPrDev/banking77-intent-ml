"""Testy modeli bazowych (models/baselines.py): sztuczne dane i tymczasowy magazyn MLflow."""

from __future__ import annotations

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.models.baselines import DUMMY_STRATEGIES, build_dummy, run_dummy_baselines


def test_unknown_dummy_strategy_is_rejected():
    with pytest.raises(ValueError, match="nieznana strategia"):
        build_dummy("prior", seed=42)


def test_dummy_runs_are_logged_for_each_strategy(tmp_mlflow_settings):
    data = TrainVal(
        x_train=pd.Series(["x"] * 5),
        y_train=pd.Series([0, 0, 0, 1, 2]),
        x_val=pd.Series(["y"] * 3),
        y_val=pd.Series([0, 1, 2]),
        label_map={"a": 0, "b": 1, "c": 2},
    )

    runs = run_dummy_baselines(data, tmp_mlflow_settings)

    assert [run.name for run in runs] == [f"dummy-{s}" for s in DUMMY_STRATEGIES]
    assert runs[0].result.accuracy == pytest.approx(1 / 3)  # most_frequent: zawsze klasa 0
    client = MlflowClient(tmp_mlflow_settings.mlflow.tracking_uri)
    for run in runs:
        assert client.get_run(run.run_id).data.tags["model_family"] == "dummy"
