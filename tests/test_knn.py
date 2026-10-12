"""Testy MiniLM + kNN (models/knn.py, W2 S2).

Infra: każdy punkt siatki = osobny run MLflow z tagami badania; blokada budżetu w CLI (druga
siatka odmawia przed liczeniem). Szkielet testów autora (klasyfikator, wybór zwycięzcy) — `skip`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.neighbors import KNeighborsClassifier

from banking77_intent_ml.data.datasets import TrainVal
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
    meta = {"encode_train_seconds": 0.1}
    return data, SplitEmbeddings(train, val, meta, from_cache=True)


def test_every_grid_point_is_a_tagged_mlflow_run(toy, tmp_mlflow_settings, monkeypatch):
    data, emb = toy
    minilm = replace(
        tmp_mlflow_settings.models.minilm_knn, k_values=(1, 3), weights=("uniform", "distance")
    )
    settings = replace(
        tmp_mlflow_settings, models=replace(tmp_mlflow_settings.models, minilm_knn=minilm)
    )
    monkeypatch.setattr(
        knn, "build_knn",
        lambda k, weights, settings: KNeighborsClassifier(k, weights=weights, metric="cosine"),
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
        assert run.data.params["knn__metric"] == "cosine"
        assert run.data.params["revision"] == settings.models.minilm_knn.revision
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


# --- Testy autora (W2 S2): treść pisze autor, potem usuwa `skip` -------------------------------


@pytest.mark.skip(reason="W2 S2: test pisze autor")
def test_build_knn_uses_grid_point_and_config_metric():
    """`n_neighbors` i `weights` z argumentów, metryka z configu; klasyfikator nieuczony."""


@pytest.mark.skip(reason="W2 S2: test pisze autor")
def test_select_best_knn_rule():
    """Najwyższy cel; remis rozstrzygany regułą autora, niezależnie od kolejności prób."""
