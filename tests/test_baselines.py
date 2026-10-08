"""Testy modeli bazowych (models/baselines.py): sztuczne dane i tymczasowy magazyn MLflow."""

from __future__ import annotations

from copy import deepcopy
from math import log

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from banking77_intent_ml.config.settings import TfidfLrConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.models import baselines
from banking77_intent_ml.models.baselines import (
    DUMMY_STRATEGIES,
    build_dummy,
    build_tfidf_lr,
    run_dummy_baselines,
    run_tfidf_lr,
)


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


# --- TF-IDF + LR (N1) --------------------------------------------------------------------


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


def test_tfidf_vocabulary_comes_only_from_train(toy_data, tmp_mlflow_settings, monkeypatch):
    """Słowo obecne wyłącznie w x_val nie trafia do słownika TF-IDF (fit tylko na train)."""
    settings = tmp_mlflow_settings
    model = build_tfidf_lr(settings.models.tfidf_lr, seed=settings.models.seed)
    monkeypatch.setattr(baselines, "build_tfidf_lr", lambda config, *, seed: model)

    run_tfidf_lr(toy_data, settings)

    tfidf = model.named_steps["tfidf"]
    assert {"top", "card", "exchange"} <= set(tfidf.vocabulary_)
    assert {"declined", "delivery", "today"}.isdisjoint(tfidf.vocabulary_)
    # "top" jest w 2 z 6 tekstów treningowych; domyślnie IDF ma wygładzanie.
    assert tfidf.idf_[tfidf.vocabulary_["top"]] == pytest.approx(log(7 / 3) + 1)


def test_tfidf_lr_steps_use_config_params():
    """Parametry kroków Pipeline'u = wartości z `models.tfidf_lr` (konfiguracja z configu)."""
    config = TfidfLrConfig(
        hpo_trials=1,
        tfidf={"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True},
        logreg={"C": 0.25, "max_iter": 250, "class_weight": "balanced"},
    )
    original = deepcopy(config)

    model = build_tfidf_lr(config, seed=17)

    assert isinstance(model, Pipeline)
    tfidf, logreg = model.named_steps["tfidf"], model.named_steps["logreg"]
    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(logreg, LogisticRegression)
    for key, value in config.tfidf.items():
        assert tfidf.get_params()[key] == value
    for key, value in config.logreg.items():
        assert logreg.get_params()[key] == value
    assert logreg.random_state == 17
    assert config == original
    assert not hasattr(tfidf, "vocabulary_")
    assert not hasattr(logreg, "coef_")


def test_tfidf_lr_run_is_logged(toy_data, tmp_mlflow_settings, monkeypatch):
    """Run TF-IDF + LR ląduje w MLflow: tag `model_family`, parametry obu kroków, metryki."""
    settings = tmp_mlflow_settings
    model = build_tfidf_lr(settings.models.tfidf_lr, seed=settings.models.seed)
    monkeypatch.setattr(baselines, "build_tfidf_lr", lambda config, *, seed: model)

    run = run_tfidf_lr(toy_data, settings)

    assert run.name == "tfidf-lr"
    assert run.model is model
    assert run.result.n_samples == len(toy_data.y_val)
    assert run.result.accuracy == pytest.approx(1.0)
    assert run.result.macro_f1 == pytest.approx(1.0)
    assert run.fit_seconds >= 0

    client = MlflowClient(settings.mlflow.tracking_uri)
    logged = client.get_run(run.run_id)
    assert logged.info.status == "FINISHED"
    assert logged.data.tags["model_family"] == "tfidf_lr"
    assert logged.data.tags["split"] == "val"
    assert logged.data.params["model"] == "tfidf_lr"
    assert logged.data.params["seed"] == str(settings.models.seed)
    for step in ("tfidf", "logreg"):
        for key, value in getattr(settings.models.tfidf_lr, step).items():
            assert logged.data.params[f"{step}__{key}"] == str(value)
    assert logged.data.metrics == pytest.approx({
        "accuracy": run.result.accuracy,
        "macro_f1": run.result.macro_f1,
        "fit_seconds": run.fit_seconds,
        "n_iter": int(model.named_steps["logreg"].n_iter_[0]),
    })
