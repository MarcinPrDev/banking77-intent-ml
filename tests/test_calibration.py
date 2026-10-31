"""Testy opisu kalibracji (evaluation/calibration.py) — prerejestracja w configu, wczytywanie
predykcji z MLflow (tymczasowy magazyn), obliczenia na danych syntetycznych i wykres.

Obliczenia sprawdzane na danych syntetycznych, bez uruchamiania raportu na predykcjach testu.
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
from banking77_intent_ml.evaluation.error_analysis import TestArtifacts as Artifacts  # bez Test*
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
        name: Artifacts(f"run-{name}", empty, empty, _predictions())
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


def test_run_tables_reads_saved_csv_only_from_calibration_run(tmp_mlflow_settings, tmp_path):
    summary = pd.DataFrame({"model": ["distilbert"], "n": [4], "ece": [0.05]})
    bins = pd.DataFrame({"model": ["distilbert"], "lower": [0.0], "upper": [1.0], "n": [4]})
    summary.to_csv(tmp_path / calibration.SUMMARY_ARTIFACT, index=False)
    bins.to_csv(tmp_path / calibration.BINS_ARTIFACT, index=False)
    configure_mlflow(tmp_mlflow_settings)
    run_ids = {}
    for key, tags in (("analysis", dict([calibration.ANALYSIS_TAG])), ("other", {})):
        with mlflow.start_run(tags=tags) as run:
            mlflow.log_artifact(str(tmp_path / calibration.SUMMARY_ARTIFACT))
            mlflow.log_artifact(str(tmp_path / calibration.BINS_ARTIFACT))
        run_ids[key] = run.info.run_id

    loaded_summary, loaded_bins = calibration.run_tables(tmp_mlflow_settings, run_ids["analysis"])

    pd.testing.assert_frame_equal(loaded_summary, summary)
    pd.testing.assert_frame_equal(loaded_bins, bins)
    with pytest.raises(ValueError, match="analysis=calibration"):
        calibration.run_tables(tmp_mlflow_settings, run_ids["other"])


# --- obliczenia (autor) ---

def test_reliability_table_bins_are_right_closed():
    """Pewność 0 → pierwszy przedział; dokładnie na granicy k/n_bins → przedział niższy;
    tuż nad granicą → wyższy; 1,0 → ostatni. Zawsze `n_bins` wierszy, `lower`/`upper` rosnąco."""
    for n_bins in (1, 3, 15):
        edges = np.linspace(0, 1, n_bins + 1)
        for value, expected_bin in [(0.0, 0), (1.0, n_bins - 1)]:
            table = calibration.reliability_table(np.array([value]), np.array([True]), n_bins)
            assert list(table.columns) == ["lower", "upper", "n", "confidence", "accuracy", "gap"]
            np.testing.assert_array_equal(table.lower, edges[:-1])
            np.testing.assert_array_equal(table.upper, edges[1:])
            assert table.n.sum() == 1
            assert table.n.iloc[expected_bin] == 1
        for k, edge in enumerate(edges[1:-1], start=1):
            for value, expected_bin in (
                (np.nextafter(edge, 0.0), k - 1),
                (edge, k - 1),
                (np.nextafter(edge, 1.0), k),
            ):
                table = calibration.reliability_table(
                    np.array([value]), np.array([True]), n_bins
                )
                assert table.n.sum() == 1
                assert table.n.iloc[expected_bin] == 1


def test_reliability_table_means_gap_and_empty_bins():
    """W przedziale: `n`, średnia pewność, odsetek trafień, `gap` = accuracy − confidence
    (wartości policzone ręcznie); pusty przedział: `n` = 0 i NaN w średnich oraz `gap`."""
    table = calibration.reliability_table(
        np.array([0.1, 0.2, 0.8, 0.9, 1.0]), np.array([1, 0, 1, 1, 1]), 4
    )
    assert table.n.tolist() == [2, 0, 0, 3]
    np.testing.assert_allclose(table.loc[[0, 3], "confidence"], [0.15, 0.9])
    np.testing.assert_allclose(table.loc[[0, 3], "accuracy"], [0.5, 1.0])
    np.testing.assert_allclose(table.loc[[0, 3], "gap"], [0.35, 0.1])
    assert table.loc[[1, 2], ["confidence", "accuracy", "gap"]].isna().all().all()


def test_reliability_table_rejects_invalid_input():
    """`ValueError`: różne długości, puste dane, NaN, pewność < 0 lub > 1, `n_bins` < 1."""
    cases = [
        ([0.5], [True, False], 2), ([], [], 2),
        ([np.nan], [True], 2), ([0.5], [np.nan], 2),
        ([-0.01], [True], 2), ([1.01], [True], 2),
        ([np.inf], [True], 2), ([0.5], [np.inf], 2),
        ([0.5], [True], 0), ([0.5], [True], -1),
        ([0.5], [True], 1.5), ([0.5], [True], True),
        ([[0.5]], [[True]], 2), ([0.5], [2], 2),
    ]
    for confidence, correct, n_bins in cases:
        with pytest.raises(ValueError):
            calibration.reliability_table(np.array(confidence), np.array(correct), n_bins)


def test_ece_is_zero_when_accuracy_equals_confidence_in_every_bin():
    table = calibration.reliability_table(
        np.array([0.0, 0.5, 0.5, 1.0]), np.array([0, 0, 1, 1]), 4
    )
    assert calibration.expected_calibration_error(table) == pytest.approx(0.0)


def test_ece_matches_hand_computed_example_and_skips_empty_bins():
    """ECE = (2/5)*|0.5−0.15| + (3/5)*|1−0.9| = 0.20; dwa puste przedziały pomijane."""
    table = calibration.reliability_table(
        np.array([0.1, 0.2, 0.8, 0.9, 1.0]), np.array([1, 0, 1, 1, 1]), 4
    )
    assert calibration.expected_calibration_error(table) == pytest.approx(0.2)
    # Przeciwne znaki luk nie mogą się znosić; wagi zależą od liczności.
    table.loc[3, "gap"] = -0.1
    assert calibration.expected_calibration_error(table) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        calibration.expected_calibration_error(table.loc[table.n == 0])
    with pytest.raises(ValueError):
        calibration.expected_calibration_error(table.iloc[:0])


def test_calibration_summary_checks_rows_and_alignment():
    """Kolumny i kolejność modeli; `ValueError` przy liczbie wierszy ≠ `dataset.test.n_rows`
    oraz przy różnych `row_id` albo `y_true` między modelami."""
    settings = load_settings()
    settings = replace(
        settings,
        dataset=replace(settings.dataset, test=replace(settings.dataset.test, n_rows=4)),
        evaluation=replace(
            settings.evaluation,
            calibration=replace(settings.evaluation.calibration, n_bins=2),
        ),
    )
    first = _predictions()
    second = first.assign(score=[1.0, 0.0, 1.0, 1.0]).iloc[::-1]
    inputs = {
        "tfidf_lr": calibration.CalibrationInput("a", first),
        "distilbert": calibration.CalibrationInput("b", second),
    }
    before = {name: item.predictions.copy(deep=True) for name, item in inputs.items()}
    summary, bins = calibration.calibration_summary(inputs, settings)
    expected = pd.DataFrame({
        "model": ["tfidf_lr", "distilbert"], "n": [4, 4],
        "accuracy": [0.75, 0.75], "mean_confidence": [0.6, 0.75], "ece": [0.15, 0.0],
    })
    pd.testing.assert_frame_equal(summary, expected)
    assert list(bins.columns) == [
        "model", "lower", "upper", "n", "confidence", "accuracy", "gap"
    ]
    assert bins.model.tolist() == ["tfidf_lr", "tfidf_lr", "distilbert", "distilbert"]
    np.testing.assert_allclose(bins.confidence, [0.4, 0.8, 0.0, 1.0])
    assert bins.n.tolist() == [2, 2, 1, 3]
    for name, item in inputs.items():
        pd.testing.assert_frame_equal(item.predictions, before[name])
    with pytest.raises(ValueError, match="3080"):
        calibration.calibration_summary(inputs, load_settings())
    with pytest.raises(ValueError):
        calibration.calibration_summary({}, settings)
    for column, replacement in (
        ("row_id", [999, 101, 102, 103]),
        ("row_id", [100, 100, 102, 103]),
        ("y_true", [2, 1, 2, 0]),
    ):
        changed = first.assign(**{column: replacement})
        bad = {**inputs, "distilbert": calibration.CalibrationInput("b", changed)}
        with pytest.raises(ValueError, match=column):
            calibration.calibration_summary(bad, settings)
