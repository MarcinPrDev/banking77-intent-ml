"""Testy wczytywania danych do analizy błędów (evaluation/error_analysis.py, W3 §1.3).

Artefakty testu z MLflow (tabele i predykcje bez tekstów), jeden run `split=test` na ścieżkę,
predykcje modeli W2 na walidacji wyrównane z tekstami i id, nazwy klas, wykres macierzy.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest

from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.data.labels import load_label_map
from banking77_intent_ml.evaluation import error_analysis, final
from banking77_intent_ml.evaluation.metrics import class_names, evaluate
from banking77_intent_ml.models.factory import STRATEGIES


def _log_test_run(settings, name: str, tmp_path) -> pd.DataFrame:
    label_map = load_label_map(settings.dataset.label_map_path)
    predictions = pd.DataFrame(
        {"row_id": [0, 1, 2, 3], "y_true": [0, 1, 2, 3], "y_pred": [0, 1, 2, 0],
         "score": [0.9, 0.8, 0.7, 0.4]},
        columns=list(final.PREDICTION_COLUMNS),
    )
    results = {
        "test": evaluate([0, 1, 2, 3], [0, 1, 2, 0], label_map),
        "clean": evaluate([0, 1, 2], [0, 1, 2], label_map),
        "llm385": evaluate([0, 2], [0, 2], label_map),
    }
    final.log_test_run(name, predictions, results, settings, artifact_dir=tmp_path)
    return predictions


def test_test_artifacts_reads_tables_and_predictions(tmp_mlflow_settings, tmp_path):
    names = class_names(load_label_map(tmp_mlflow_settings.dataset.label_map_path))
    logged = {name: _log_test_run(tmp_mlflow_settings, name, tmp_path)
              for name in STRATEGIES}
    # Dodatkowe rodziny nie mają artefaktów pełnego testu, np. LLM oceniony na 385.
    for family in ("llm", "future_model"):
        with mlflow.start_run(tags={"split": "test", "model_family": family}):
            mlflow.log_metric("accuracy", 0.5)
    with mlflow.start_run(tags={"split": "test"}):
        mlflow.log_metric("accuracy", 0.5)

    artifacts = error_analysis.test_artifacts(tmp_mlflow_settings)

    assert set(artifacts) == set(logged)
    for name, predictions in logged.items():
        loaded = artifacts[name]
        assert loaded.confusion.shape == (len(names), len(names))
        assert loaded.confusion.index.tolist() == names
        assert loaded.confusion.columns.tolist() == names
        assert loaded.per_class.index.tolist() == names
        assert {"precision", "recall", "f1", "support"} <= set(loaded.per_class.columns)
        pd.testing.assert_frame_equal(loaded.predictions, predictions)
        assert "text" not in loaded.predictions.columns  # zdań testowych nie wczytujemy


def test_test_artifacts_rejects_second_test_run_of_a_path(tmp_mlflow_settings, tmp_path):
    for name in STRATEGIES:
        _log_test_run(tmp_mlflow_settings, name, tmp_path)
    _log_test_run(tmp_mlflow_settings, "tfidf_lr", tmp_path)

    with pytest.raises(ValueError, match="tfidf_lr"):
        error_analysis.test_artifacts(tmp_mlflow_settings)


@pytest.mark.parametrize("missing", [*STRATEGIES, None])
def test_test_artifacts_rejects_missing_run(tmp_mlflow_settings, tmp_path, missing):
    """Brak dowolnej ścieżki lub wszystkich runów nie może dać częściowego wyniku."""
    if missing is not None:
        for name in STRATEGIES:
            if name != missing:
                _log_test_run(tmp_mlflow_settings, name, tmp_path)
    with pytest.raises(ValueError, match=rf"{missing or 'distilbert'}: 0 runów"):
        error_analysis.test_artifacts(tmp_mlflow_settings)


class _FixedScores:
    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def predict_proba(self, texts):
        assert len(texts) == len(self.scores)
        return self.scores


def test_val_predictions_w2_align_ids_texts_and_argmax(tmp_mlflow_settings, monkeypatch):
    index = [7, 3, 5]  # id wierszy train.csv, nie 0..n-1
    data = TrainVal(
        x_train=pd.Series(["t"]), y_train=pd.Series([0]),
        x_val=pd.Series(["a", "b", "c"], index=index), y_val=pd.Series([1, 2, 3], index=index),
        label_map={},
    )
    scores = np.zeros((3, 77))
    scores[[0, 1, 2], [1, 9, 3]] = [0.7, 0.6, 0.5]
    loaded = []
    monkeypatch.setattr(
        error_analysis, "load_predictor",
        lambda name, settings: loaded.append(name) or _FixedScores(scores),
    )

    frames = error_analysis.val_predictions_w2(tmp_mlflow_settings, data)

    assert sorted(frames) == sorted(loaded) == ["distilbert", "minilm_knn", "tfidf_lr"]
    frame = frames["tfidf_lr"]
    assert frame["row_id"].tolist() == index
    assert frame["text"].tolist() == ["a", "b", "c"]
    assert frame["y_true"].tolist() == [1, 2, 3]
    assert frame["y_pred"].tolist() == [1, 9, 3]
    np.testing.assert_allclose(frame["score"], [0.7, 0.6, 0.5])


def test_add_label_names_maps_ids_with_project_label_map(tmp_mlflow_settings):
    names = class_names(load_label_map(tmp_mlflow_settings.dataset.label_map_path))
    frame = pd.DataFrame({"y_true": [0, 76], "y_pred": [76, 0]})

    named = error_analysis.add_label_names(frame, tmp_mlflow_settings)

    assert named["true_name"].tolist() == [names[0], names[76]]
    assert named["pred_name"].tolist() == [names[76], names[0]]
    assert "true_name" not in frame.columns  # kopia, bez mutacji wejścia


def test_plot_confusion_draws_normalized_matrix():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    confusion = pd.DataFrame([[3, 1], [0, 0]], index=["a", "b"], columns=["a", "b"])

    ax = error_analysis.plot_confusion(confusion, title="próba")

    assert ax.get_title() == "próba"
    np.testing.assert_allclose(ax.images[0].get_array(), [[0.75, 0.25], [0.0, 0.0]])
