"""Testy oceny końcowej (evaluation/final.py, W3 §1.2) — syntetyczny „test”, bez prawdziwych danych.

Infra (asystent): wczytanie testu z kontrolą hashy i zakresu podzbiorów, run MLflow z metrykami
podzbiorów i artefaktami, licznik runów testowych, blokady CLI przed wczytaniem testu.
Kontrakty `predict_test` i `subset_results` — jeden przebieg i podzbiory po `row_id`.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.config.settings import DatasetFileConfig
from banking77_intent_ml.data.labels import load_label_map
from banking77_intent_ml.data.splits import SplitError, ids_sha256
from banking77_intent_ml.evaluation import final
from banking77_intent_ml.evaluation.metrics import evaluate
from banking77_intent_ml.models.predictor import IntentClassifier

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_test.py"


def _write_ids(path: Path, ids: list[int]) -> None:
    path.write_text(json.dumps(ids), encoding="utf-8")


@pytest.fixture
def synthetic_env(tmp_path, tmp_mlflow_settings):
    """Syntetyczny plik testu (77 wierszy, po jednym na klasę), podzbiory i manifest w tmp_path."""
    settings = tmp_mlflow_settings
    names = sorted(load_label_map(settings.dataset.label_map_path))
    pd.DataFrame({"text": [f"zdanie {i}" for i in range(len(names))], "category": names}).to_csv(
        tmp_path / "test.csv", index=False
    )
    clean, llm = list(range(70)), [0, 10, 20, 30, 40, 50, 60]
    _write_ids(tmp_path / "clean.json", clean)
    _write_ids(tmp_path / "llm.json", llm)
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        sha256_test_clean_ids=ids_sha256(clean), n_test_clean=len(clean),
        sha256_llm_eval_ids=ids_sha256(llm), n_llm_eval=len(llm),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return replace(
        settings,
        dataset=replace(
            settings.dataset,
            test=DatasetFileConfig(path="banking_data/test.csv", sha256="-", n_rows=len(names)),
        ),
        paths=replace(
            settings.paths, raw_dir=tmp_path, final_models_dir=tmp_path / "final"
        ),
        split=replace(settings.split, manifest_path=tmp_path / "manifest.json"),
        evaluation=replace(settings.evaluation, clean_test_subset_path=tmp_path / "clean.json"),
        llm=replace(settings.llm, eval_ids_path=tmp_path / "llm.json"),
    )


def test_load_test_set_reads_test_and_verified_subsets(synthetic_env):
    test = final.load_test_set(synthetic_env)

    assert len(test.x) == len(test.y) == 77
    assert list(test.y) == list(range(77))  # jedna klasa na wiersz, id wg mapy
    assert test.clean_ids == list(range(70))
    assert test.llm_ids == [0, 10, 20, 30, 40, 50, 60]


def test_load_test_set_rejects_subset_changed_after_w1(synthetic_env):
    _write_ids(synthetic_env.evaluation.clean_test_subset_path, list(range(69)))

    with pytest.raises(SplitError, match="hash"):
        final.load_test_set(synthetic_env)


@pytest.mark.parametrize("subset", ["clean", "llm"])
def test_load_test_set_rejects_ids_outside_test(synthetic_env, subset):
    path = (
        synthetic_env.evaluation.clean_test_subset_path if subset == "clean"
        else synthetic_env.llm.eval_ids_path
    )
    ids = [*json.loads(path.read_text(encoding="utf-8"))[:-1], 999]
    _write_ids(path, ids)
    manifest = json.loads(synthetic_env.split.manifest_path.read_text(encoding="utf-8"))
    key = "sha256_test_clean_ids" if subset == "clean" else "sha256_llm_eval_ids"
    manifest[key] = ids_sha256(ids)  # hash zgodny, a pozycja spoza testu
    synthetic_env.split.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SplitError, match="spoza testu"):
        final.load_test_set(synthetic_env)


def _results(label_map: dict[str, int]) -> dict:
    return {
        "test": evaluate([0, 1, 2, 3], [0, 1, 2, 0], label_map),
        "clean": evaluate([0, 1, 2], [0, 1, 2], label_map),
        "llm385": evaluate([0, 2], [0, 2], label_map),
    }


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {"row_id": [0, 1, 2, 3], "y_true": [0, 1, 2, 3], "y_pred": [0, 1, 2, 0],
         "score": [0.9, 0.8, 0.7, 0.4]},
        columns=list(final.PREDICTION_COLUMNS),
    )


def test_log_test_run_records_subset_metrics_tables_and_predictions(tmp_mlflow_settings, tmp_path):
    label_map = load_label_map(tmp_mlflow_settings.dataset.label_map_path)

    run_id = final.log_test_run(
        "tfidf_lr", _predictions(), _results(label_map), tmp_mlflow_settings,
        artifact_dir=tmp_path / "final" / "tfidf_lr",
    )

    client = MlflowClient(tmp_mlflow_settings.mlflow.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["split"] == "test"
    assert run.data.tags["model_family"] == "tfidf_lr"
    assert run.data.params["n_test"] == "4"
    assert run.data.metrics["test_accuracy"] == pytest.approx(0.75)
    assert run.data.metrics["clean_accuracy"] == pytest.approx(1.0)
    assert {"llm385_accuracy", "llm385_macro_f1", "test_macro_f1"} <= set(run.data.metrics)
    artifacts = {artifact.path for artifact in client.list_artifacts(run_id)}
    expected = {final.PREDICTIONS_ARTIFACT, final.PER_CLASS_ARTIFACT, final.CONFUSION_ARTIFACT}
    assert expected <= artifacts
    assert final.count_test_runs(tmp_mlflow_settings, "tfidf_lr") == 1
    assert final.count_test_runs(tmp_mlflow_settings, "distilbert") == 0


def test_log_test_run_rejects_missing_subset_before_creating_run(tmp_mlflow_settings, tmp_path):
    label_map = load_label_map(tmp_mlflow_settings.dataset.label_map_path)
    results = _results(label_map)
    del results["llm385"]

    with pytest.raises(ValueError, match="llm385"):
        final.log_test_run(
            "tfidf_lr", _predictions(), results, tmp_mlflow_settings, artifact_dir=tmp_path
        )

    assert final.count_test_runs(tmp_mlflow_settings, "tfidf_lr") == 0


@pytest.mark.parametrize(
    ("tag", "artifact", "runs", "message"),
    [
        (False, True, 0, "experiments-frozen"),
        (True, False, 0, "refit_final"),
        (True, True, 1, "split=test"),
    ],
)
def test_cli_refuses_before_loading_test(
    synthetic_env, monkeypatch, capsys, tag, artifact, runs, message
):
    if artifact:
        (synthetic_env.paths.final_models_dir / "tfidf_lr").mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("evaluate_test_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: synthetic_env)
    monkeypatch.setattr(cli, "_tag_exists", lambda name: tag)
    monkeypatch.setattr(cli, "count_test_runs", lambda *args: runs)
    monkeypatch.setattr(cli, "load_test_set", lambda *args: pytest.fail("test wczytany"))
    monkeypatch.setattr("sys.argv", ["evaluate_test.py", "--model", "tfidf_lr"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert message in capsys.readouterr().out


# --- Kontrakty funkcji autora (W3 §1.2) --------------------------------------------------


class _CountingPredictor(IntentClassifier):
    """Stałe wyniki (n, 77); liczy wywołania `predict_proba` (jeden przebieg = 1)."""

    name = "counting"

    def __init__(self, settings, scores: np.ndarray):
        super().__init__(settings)
        self.scores = scores
        self.calls = 0
        self.received_texts = None

    def fit(self, texts, labels):
        raise NotImplementedError

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)

    def predict_proba(self, texts):
        self.calls += 1
        self.received_texts = list(texts)
        assert len(texts) == len(self.scores)
        return self.scores

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory, settings):
        raise NotImplementedError


@pytest.fixture
def small_test(synthetic_env):
    """Syntetyczny TestSet (77 przykładów, podzbiory 70 i 7) + predyktor o znanych wynikach."""
    test = final.load_test_set(synthetic_env)
    scores = np.full((len(test.x), 77), 0.001)
    scores[np.arange(len(test.x)), (np.arange(len(test.x)) + 1) % 77] = 0.9  # zawsze „następna”
    return test, _CountingPredictor(synthetic_env, scores)


def test_predict_test_single_pass_over_full_test(small_test):
    """Kontrakt (decyzje autora 14.10): jeden `predict_proba` na wszystkich tekstach; kolumny
    `PREDICTION_COLUMNS`; `row_id` = indeks testu; `y_true` dopasowane po id; `y_pred` = argmax;
    `score` = wynik zwróconej klasy."""
    test, predictor = small_test
    test = replace(test, x=test.x.iloc[::-1])  # y w innej kolejności niż teksty
    expected_pred = np.r_[np.arange(1, 77), 0]
    expected_score = np.linspace(0.6, 0.9, 77)
    predictor.scores[np.arange(77), expected_pred] = expected_score

    predictions = final.predict_test(predictor, test)

    assert predictor.calls == 1
    assert predictor.received_texts == test.x.tolist()
    assert tuple(predictions.columns) == ("row_id", "y_true", "y_pred", "score")
    assert len(predictions) == 77
    np.testing.assert_array_equal(predictions["row_id"], test.x.index)
    np.testing.assert_array_equal(predictions["y_true"], np.arange(76, -1, -1))
    np.testing.assert_array_equal(predictions["y_pred"], expected_pred)
    np.testing.assert_allclose(predictions["score"], expected_score)
    for column in ("row_id", "y_true", "y_pred"):
        assert pd.api.types.is_integer_dtype(predictions[column])


def test_subset_results_come_from_the_same_predictions(small_test):
    """Kontrakt: klucze `SUBSETS`; `n_samples` = 77 / 70 / 7 (tu syntetycznie); metryki =
    `evaluate()` na wierszach podzbioru wybranych po `row_id` z tej samej ramki predykcji."""
    test, _ = small_test
    # Inne etykiety niż test.y: evaluate ma korzystać wyłącznie z zapisanej ramki.
    true = (test.y.to_numpy() + 1) % 77
    pred = true.copy()
    mistakes = [0, 10, 69, 76]
    pred[mistakes] = (true[mistakes] + 1) % 77
    predictions = pd.DataFrame({
        "row_id": test.x.index, "y_true": true, "y_pred": pred, "score": 0.9,
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    before = predictions.copy(deep=True)

    results = final.subset_results(predictions, test)

    assert tuple(results) == final.SUBSETS
    expected_subsets = {
        "test": (list(range(77)), 77, 73 / 77),
        "clean": (test.clean_ids, 70, 67 / 70),
        "llm385": (test.llm_ids, 7, 5 / 7),
    }
    for name, (ids, count, accuracy) in expected_subsets.items():
        expected = evaluate(true[ids], pred[ids], test.label_map)
        actual = results[name]
        assert actual.n_samples == count
        assert actual.accuracy == pytest.approx(accuracy)
        assert actual.metrics() == pytest.approx(expected.metrics())
        pd.testing.assert_frame_equal(actual.per_class, expected.per_class)
        pd.testing.assert_frame_equal(actual.confusion, expected.confusion)
    pd.testing.assert_frame_equal(predictions, before)
