"""Testy opisu kalibracji (evaluation/calibration.py) — prerejestracja w configu, wczytywanie
predykcji z MLflow (tymczasowy magazyn), obliczenia na danych syntetycznych i wykres.

Testy obliczeń (`reliability_table`, `expected_calibration_error`, `calibration_summary`) pisze
autor — w szkielecie oznaczone `TODO-Marcin`.
"""

from __future__ import annotations

from dataclasses import replace

import mlflow
import numpy as np
import pandas as pd
import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.evaluation import calibration
from banking77_intent_ml.evaluation.basal_eval import TEST_PREDICTIONS_ARTIFACT
from banking77_intent_ml.evaluation.error_analysis import TestArtifacts
from banking77_intent_ml.evaluation.final import PREDICTION_COLUMNS
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.factory import STRATEGIES


def test_preregistration_names_known_models_and_bins():
    """Modele z configu to ścieżki `STRATEGIES` albo kandydaci badań Basala; kNN tylko jako nota."""
    settings = load_settings()
    cfg = settings.evaluation.calibration
    basal = set(settings.basal.models) | set(settings.basal_ultra.models)
    assert cfg.models == ("distilbert", "tfidf_lr", "basal_4_5b", "basal_omni_ultra")
    assert set(cfg.models) <= set(STRATEGIES) | basal
    assert "minilm_knn" not in cfg.models  # n_neighbors 1: score zawsze 1,0
    assert cfg.n_bins == 15
    assert cfg.figure_path.suffix == ".png"


# --- wczytywanie predykcji (asystent) ---

def _predictions(n: int = 4) -> pd.DataFrame:
    """Ramka jak `predictions_test.csv`: `row_id, y_true, y_pred, score`."""
    return pd.DataFrame({
        "row_id": np.arange(n) + 100,
        "y_true": np.arange(n) % 3,
        "y_pred": [0, 2, 2, 0][:n],
        "score": np.linspace(0.3, 0.9, n),
    })


def _fake_test_artifacts(settings):
    empty = pd.DataFrame()
    return {
        name: TestArtifacts(f"run-{name}", empty, empty, _predictions())
        for name in sorted(STRATEGIES)
    }


def _log_basal_test_run(settings, candidate: str, tmp_path, *, split: str = "test") -> str:
    """Run jak `basal_eval.run_test`: tagi i `predictions_basal_test.csv` (kolumny Basala)."""
    frame = _predictions().rename(columns={"y_true": "true_id", "y_pred": "predicted_id"})
    folder = tmp_path / f"{candidate}-{split}"
    folder.mkdir(exist_ok=True)
    path = folder / TEST_PREDICTIONS_ARTIFACT
    frame.to_csv(path, index=False)
    configure_mlflow(settings)
    tags = {"split": split, "model_family": "basal", "candidate": candidate}
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_artifact(str(path))
    return run.info.run_id


def test_calibration_inputs_reads_strategies_and_basal_runs(
    tmp_mlflow_settings, tmp_path, monkeypatch
):
    monkeypatch.setattr(calibration, "test_artifacts", _fake_test_artifacts)
    ids = {name: _log_basal_test_run(tmp_mlflow_settings, name, tmp_path)
           for name in ("basal_4_5b", "basal_omni_ultra")}
    _log_basal_test_run(tmp_mlflow_settings, "basal_4_5b", tmp_path, split="val")  # pomijany

    inputs = calibration.calibration_inputs(tmp_mlflow_settings)

    assert list(inputs) == list(tmp_mlflow_settings.evaluation.calibration.models)
    assert inputs["distilbert"].run_id == "run-distilbert"
    for name, run_id in ids.items():
        assert inputs[name].run_id == run_id
    for item in inputs.values():
        assert list(item.predictions.columns) == list(PREDICTION_COLUMNS)
        pd.testing.assert_frame_equal(item.predictions, _predictions(), check_dtype=False)


@pytest.mark.parametrize(
    "logged", [("basal_4_5b",), ("basal_4_5b", "basal_4_5b", "basal_omni_ultra")]
)
def test_calibration_inputs_needs_exactly_one_test_run_per_basal_model(
    tmp_mlflow_settings, tmp_path, monkeypatch, logged
):
    monkeypatch.setattr(calibration, "test_artifacts", _fake_test_artifacts)
    for name in logged:
        _log_basal_test_run(tmp_mlflow_settings, name, tmp_path)
    with pytest.raises(ValueError, match="oczekiwano jednego"):
        calibration.calibration_inputs(tmp_mlflow_settings)


def test_calibration_inputs_rejects_unknown_model(tmp_mlflow_settings):
    settings = tmp_mlflow_settings
    cfg = replace(settings.evaluation.calibration, models=("distilbert", "nieznany"))
    settings = replace(settings, evaluation=replace(settings.evaluation, calibration=cfg))
    with pytest.raises(ValueError, match="nieznany"):
        calibration.calibration_inputs(settings)


def test_plot_reliability_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    summary = pd.DataFrame({
        "model": ["distilbert", "basal_omni_ultra"], "n": [4, 4],
        "accuracy": [0.75, 0.5], "mean_confidence": [0.8, 0.6], "ece": [0.05, 0.1],
    })
    edges = np.linspace(0, 1, 3)
    bins = pd.DataFrame({
        "model": ["distilbert"] * 2 + ["basal_omni_ultra"] * 2,
        "lower": [*edges[:-1]] * 2, "upper": [*edges[1:]] * 2,
        "n": [0, 4, 2, 2],
        "confidence": [np.nan, 0.8, 0.4, 0.8], "accuracy": [np.nan, 0.75, 0.5, 0.5],
        "gap": [np.nan, -0.05, 0.1, -0.3],
    })
    path = calibration.plot_reliability(summary, bins, tmp_path / "fig" / "reliability.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- obliczenia (autor) ---

def test_reliability_table_bins_are_right_closed():
    """Pewność 0 → pierwszy przedział; dokładnie na granicy k/n_bins → przedział niższy;
    tuż nad granicą → wyższy; 1,0 → ostatni. Zawsze `n_bins` wierszy, `lower`/`upper` rosnąco."""
    pytest.fail("TODO-Marcin")


def test_reliability_table_means_gap_and_empty_bins():
    """W przedziale: `n`, średnia pewność, odsetek trafień, `gap` = accuracy − confidence
    (wartości policzone ręcznie); pusty przedział: `n` = 0 i NaN w średnich oraz `gap`."""
    pytest.fail("TODO-Marcin")


def test_reliability_table_rejects_invalid_input():
    """`ValueError`: różne długości, puste dane, NaN, pewność < 0 lub > 1, `n_bins` < 1."""
    pytest.fail("TODO-Marcin")


def test_ece_is_zero_when_accuracy_equals_confidence_in_every_bin():
    pytest.fail("TODO-Marcin")


def test_ece_matches_hand_computed_example_and_skips_empty_bins():
    """Mały przykład z kilkoma przedziałami, w tym pustym; ECE policzone ręcznie w docstringu."""
    pytest.fail("TODO-Marcin")


def test_calibration_summary_checks_rows_and_alignment():
    """Kolumny i kolejność modeli; `ValueError` przy liczbie wierszy ≠ `dataset.test.n_rows`
    oraz przy różnych `row_id` albo `y_true` między modelami."""
    pytest.fail("TODO-Marcin")
