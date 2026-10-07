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


# --- TF-IDF + LR (N1): szkielet — treść testów pisze autor, potem usuwa `skip` -----------------


@pytest.fixture
def toy_data() -> TrainVal:
    """Sztuczne dane do testów TF-IDF + LR: 3 klasy o rozróżnialnym słownictwie."""
    return TrainVal(
        x_train=pd.Series([
            "top up failed", "top up pending", "card not arrived", "card arrived late",
            "exchange rate fee", "bad exchange rate",
        ]),
        y_train=pd.Series([0, 0, 1, 1, 2, 2]),
        x_val=pd.Series(["top up declined", "card delivery", "exchange rate today"]),
        y_val=pd.Series([0, 1, 2]),
        label_map={"a": 0, "b": 1, "c": 2},
    )


@pytest.mark.skip(reason="N1: test pisze autor")
def test_tfidf_vocabulary_comes_only_from_train(toy_data):
    """Słowo obecne wyłącznie w x_val nie trafia do słownika TF-IDF (fit tylko na train)."""


@pytest.mark.skip(reason="N1: test pisze autor")
def test_tfidf_lr_steps_use_config_params():
    """Parametry kroków Pipeline'u = wartości z `models.tfidf_lr` (konfiguracja z configu)."""


@pytest.mark.skip(reason="N1: test pisze autor")
def test_tfidf_lr_run_is_logged(toy_data, tmp_mlflow_settings):
    """Run TF-IDF + LR ląduje w MLflow: tag `model_family`, parametry obu kroków, metryki."""
