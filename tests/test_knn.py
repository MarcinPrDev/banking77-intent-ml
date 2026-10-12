"""Testy MiniLM + kNN (models/knn.py, W2 S2).

Infra: każdy punkt siatki = osobny run MLflow z tagami badania; blokada budżetu w CLI (druga
siatka odmawia przed liczeniem). Klasyfikator i reguła wyboru sprawdzane na sztucznych danych.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.neighbors import KNeighborsClassifier

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import evaluate
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.features.embeddings import SplitEmbeddings
from banking77_intent_ml.models import knn
from banking77_intent_ml.models.hpo import STUDY_TAG

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "knn_minilm.py"


@pytest.fixture
def toy() -> tuple[TrainVal, SplitEmbeddings]:
    """3 klasy, embeddingi skupione wokół osi (znormalizowane)."""
    data = TrainVal(
        x_train=pd.Series([f"t{i}" for i in range(6)]),
        y_train=pd.Series([0, 0, 1, 1, 2, 2]),
        x_val=pd.Series(["v0", "v1", "v2"]),
        y_val=pd.Series([0, 1, 2]),
        label_map={"a": 0, "b": 1, "c": 2},
    )
    rng = np.random.default_rng(0)
    centers = np.eye(3)
    train = centers[[0, 0, 1, 1, 2, 2]] + 0.05 * rng.normal(size=(6, 3))
    val = centers[[0, 1, 2]] + 0.05 * rng.normal(size=(3, 3))
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    val /= np.linalg.norm(val, axis=1, keepdims=True)
    meta = {"encode_train_seconds": 0.1, "sentence_transformers": "test", "device": "cpu"}
    return data, SplitEmbeddings(train, val, meta, from_cache=True)


def test_every_grid_point_is_a_tagged_mlflow_run(toy, tmp_mlflow_settings):
    data, emb = toy
    minilm = replace(
        tmp_mlflow_settings.models.minilm_knn, k_values=(1, 3), weights=("uniform", "distance")
    )
    settings = replace(
        tmp_mlflow_settings, models=replace(tmp_mlflow_settings.models, minilm_knn=minilm)
    )
    trials = knn.run_knn_grid(data, emb, settings)

    assert [(t.n_neighbors, t.weights) for t in trials] == [
        (1, "uniform"), (1, "distance"), (3, "uniform"), (3, "distance"),
    ]
    assert count_runs_with_tag(settings, STUDY_TAG, knn.STUDY_NAME) == 4
    client = MlflowClient(settings.mlflow.tracking_uri)
    for trial in trials:
        run = client.get_run(trial.run_id)
        assert run.data.tags["mlflow.runName"] == f"minilm-knn-k{trial.n_neighbors}-{trial.weights}"
        assert run.data.tags["model_family"] == "minilm_knn"
        assert run.data.params["knn__n_neighbors"] == str(trial.n_neighbors)
        assert run.data.params["knn__weights"] == trial.weights
        assert run.data.params["knn__metric"] == "cosine"
        assert run.data.params["knn__algorithm"] == "brute"
        assert run.data.params["revision"] == settings.models.minilm_knn.revision
        assert run.data.params["sentence_transformers"] == "test"
        assert run.data.params["embedding_device"] == "cpu"
        assert run.data.metrics["encode_train_seconds"] == pytest.approx(0.1)
        assert trial.objective == trial.result.macro_f1


def test_cli_refuses_second_grid_before_computing(tmp_mlflow_settings, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("knn_minilm_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: tmp_mlflow_settings)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda *args: 8)
    monkeypatch.setattr(
        cli, "load_train_val", lambda *args: pytest.fail("dane wczytane mimo zużytego budżetu")
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "budżet" in capsys.readouterr().out


# --- Klasyfikator i reguła wyboru (W2 S2) -----------------------------------------------------


@pytest.mark.parametrize(("n_neighbors", "weights", "metric"), [
    (1, "uniform", "cosine"),
    (3, "distance", "cosine"),
    (2, "distance", "euclidean"),  # metryka z configu, nie wpisana na sztywno
])
def test_build_knn_uses_grid_point_and_config_metric(n_neighbors, weights, metric):
    """`n_neighbors` i `weights` z argumentów, metryka z configu; klasyfikator nieuczony."""
    settings = load_settings()
    minilm = replace(settings.models.minilm_knn, metric=metric)
    settings = replace(settings, models=replace(settings.models, minilm_knn=minilm))

    model = knn.build_knn(n_neighbors, weights, settings)

    assert isinstance(model, KNeighborsClassifier)
    params = model.get_params()
    assert params["n_neighbors"] == n_neighbors
    assert params["weights"] == weights
    assert params["metric"] == metric
    assert params["algorithm"] == "brute"
    assert not hasattr(model, "classes_")
    assert not hasattr(model, "n_features_in_")


@pytest.mark.parametrize(("points", "winner"), [
    ([(1, "uniform", 0.8, "a"), (3, "distance", 0.9, "b")], "b"),
    ([(1, "uniform", 0.9, "a"), (3, "distance", 0.9001, "b")], "b"),  # brak tolerancji
    ([(3, "uniform", 0.9, "a"), (1, "distance", 0.9, "b")], "b"),  # najpierw mniejsze k
    ([(3, "distance", 0.9, "a"), (3, "uniform", 0.9, "b")], "b"),  # potem uniform
    ([(3, "uniform", 0.9, "b"), (3, "uniform", 0.9, "a")], "a"),  # powtórzony punkt
    ([(1, "uniform", 0.9, "a")], "a"),
    ([], None),
])
def test_select_best_knn_rule(points, winner):
    """Najwyższy cel, potem mniejsze k, uniform i run_id; wszystkie permutacje i pusty zbiór."""
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    trials = [
        knn.KnnTrial(k, weights, run_id, replace(result, macro_f1=score), score)
        for k, weights, score, run_id in points
    ]

    if winner is None:
        with pytest.raises(ValueError):
            knn.select_best_knn(trials)
        return

    expected = next(trial for trial in trials if trial.run_id == winner)
    for ordered in permutations(trials):
        assert knn.select_best_knn(list(ordered)) is expected
