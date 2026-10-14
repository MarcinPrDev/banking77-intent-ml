"""Testy refitu przed oceną na teście (models/refit.py, data/datasets.full_train, W3).

`full_train`: train + dawna walidacja w kolejności wierszy train.csv. `refit`: fit przez wspólny
interfejs, zapis do `final_models_dir`, run MLflow z czasem treningu; nie nadpisuje artefaktu.
CLI odmawia przed wczytaniem danych bez tagu zamrożenia, z istniejącym artefaktem albo runem.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.data.datasets import TrainVal, full_train
from banking77_intent_ml.models import refit as refit_module
from banking77_intent_ml.models.refit import REFIT_TAG, refit

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "refit_final.py"
_TEXTS = [
    "where is my card", "my card has not arrived", "card still not here",
    "top up failed", "top up declined", "my top up did not work",
]


@pytest.fixture
def final_settings(tmp_path, tmp_mlflow_settings):
    base = tmp_mlflow_settings
    return replace(base, paths=replace(base.paths, final_models_dir=tmp_path / "final"))


def test_full_train_merges_train_and_val_in_file_order():
    data = TrainVal(
        x_train=pd.Series(["a", "c", "e"], index=[0, 2, 4]),
        y_train=pd.Series([10, 12, 14], index=[0, 2, 4]),
        x_val=pd.Series(["d", "b"], index=[3, 1]),
        y_val=pd.Series([13, 11], index=[3, 1]),
        label_map={},
    )

    x, y = full_train(data)

    assert list(x.index) == [0, 1, 2, 3, 4]
    assert list(x) == ["a", "b", "c", "d", "e"]
    assert list(y) == [10, 11, 12, 13, 14]


def test_refit_saves_final_artifact_and_logs_training_run(final_settings):
    index = range(100, 100 + 2 * len(_TEXTS), 2)  # id wierszy jak w train.csv
    x = pd.Series(_TEXTS, index=index)
    y = pd.Series([0, 0, 0, 1, 1, 1], index=index)

    result = refit("tfidf_lr", x, y, final_settings)

    assert result.directory == final_settings.paths.final_models_dir / "tfidf_lr"
    assert (result.directory / "model.joblib").is_file()
    run = MlflowClient(final_settings.mlflow.tracking_uri).get_run(result.run_id)
    assert run.data.tags[REFIT_TAG] == "tfidf_lr"
    assert run.data.tags["stage"] == "final"
    assert run.data.tags["split"] == "train+val"
    assert run.data.params["n_train"] == str(len(_TEXTS))
    assert run.data.params["device"] == "cpu"
    assert "char_wb" in run.data.params["frozen"]
    assert run.data.metrics["fit_seconds"] == pytest.approx(result.fit_seconds)


def test_refit_refuses_existing_artifact_before_fitting(final_settings, monkeypatch):
    (final_settings.paths.final_models_dir / "tfidf_lr").mkdir(parents=True)
    monkeypatch.setattr(
        refit_module, "build_predictor", lambda *args: pytest.fail("fit mimo artefaktu")
    )

    with pytest.raises(FileExistsError):
        refit("tfidf_lr", pd.Series(_TEXTS), pd.Series([0] * 6), final_settings)


def _cli(monkeypatch, settings, *, tag: bool, runs: int = 0):
    spec = importlib.util.spec_from_file_location("refit_final_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "_tag_exists", lambda name: tag)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda *args: runs)
    monkeypatch.setattr(cli, "load_train_val", lambda *args: pytest.fail("dane wczytane"))
    monkeypatch.setattr("sys.argv", ["refit_final.py", "--model", "tfidf_lr"])
    return cli


@pytest.mark.parametrize(
    ("tag", "runs", "existing", "message"),
    [
        (False, 0, False, "experiments-frozen"),
        (True, 0, True, "już istnieje"),
        (True, 1, False, "refit"),
    ],
)
def test_cli_refuses_before_loading_data(
    final_settings, monkeypatch, capsys, tag, runs, existing, message
):
    if existing:
        (final_settings.paths.final_models_dir / "tfidf_lr").mkdir(parents=True)
    cli = _cli(monkeypatch, final_settings, tag=tag, runs=runs)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert message in capsys.readouterr().out
