"""Testy HPO TF-IDF + LR (models/hpo.py, W2 S1).

Infra: każda próba = osobny run MLflow z tagami badania; blokada budżetu w CLI (drugie badanie
odmawia przed treningiem). Szkielet testów autora (przestrzeń, cel, wybór zwycięzcy) — `skip`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.config.settings import TfidfLrHpoConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag
from banking77_intent_ml.models import hpo

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "hpo_tfidf_lr.py"


@pytest.fixture
def toy_data() -> TrainVal:
    """Sztuczne dane: 3 klasy o rozróżnialnym słownictwie."""
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


def _hpo_settings(settings, *, trials: int, sampler: str = "tpe"):
    tfidf_lr = replace(
        settings.models.tfidf_lr,
        hpo_trials=trials,
        hpo=TfidfLrHpoConfig(sampler=sampler, tie_tolerance=0.001, space={}),
    )
    return replace(settings, models=replace(settings.models, tfidf_lr=tfidf_lr))


def _fake_suggest(trial, settings):
    """Zastępcza przestrzeń (sam `C`) — testuje pętlę, nie przestrzeń autora."""
    base = settings.models.tfidf_lr
    return replace(base, logreg={**base.logreg, "C": trial.suggest_categorical("C", [0.5, 2.0])})


def test_every_trial_is_a_tagged_mlflow_run(toy_data, tmp_mlflow_settings, monkeypatch):
    settings = _hpo_settings(tmp_mlflow_settings, trials=3)
    monkeypatch.setattr(hpo, "suggest_tfidf_lr", _fake_suggest)
    monkeypatch.setattr(hpo, "objective_value", lambda result, settings: result.macro_f1)

    trials = hpo.run_tfidf_lr_hpo(toy_data, settings)

    assert [trial.number for trial in trials] == [0, 1, 2]
    assert count_runs_with_tag(settings, hpo.STUDY_TAG, hpo.STUDY_NAME) == 3
    client = MlflowClient(settings.mlflow.tracking_uri)
    for trial in trials:
        run = client.get_run(trial.run_id)
        assert run.data.tags["mlflow.runName"] == f"tfidf-lr-hpo-{trial.number:02d}"
        assert run.data.tags["hpo_trial"] == str(trial.number)
        assert run.data.tags["model_family"] == "tfidf_lr"
        assert run.data.params["logreg__C"] == str(trial.config.logreg["C"])
        assert trial.objective == trial.result.macro_f1
        assert trial.n_features > 0


def test_hpo_requires_tpe_sampler(toy_data, tmp_mlflow_settings):
    settings = _hpo_settings(tmp_mlflow_settings, trials=1, sampler="grid")
    with pytest.raises(ValueError, match="tpe"):
        hpo.run_tfidf_lr_hpo(toy_data, settings)


def test_count_runs_with_tag_on_fresh_store(tmp_mlflow_settings):
    assert count_runs_with_tag(tmp_mlflow_settings, hpo.STUDY_TAG, hpo.STUDY_NAME) == 0


def test_cli_refuses_second_study_before_training(tmp_mlflow_settings, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("hpo_tfidf_lr_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: tmp_mlflow_settings)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda *args: 16)
    monkeypatch.setattr(
        cli, "load_train_val", lambda *args: pytest.fail("dane wczytane mimo zużytego budżetu")
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "budżet" in capsys.readouterr().out


# --- Testy autora (W2 S1): treść pisze autor, potem usuwa `skip` -------------------------------


@pytest.mark.skip(reason="W2 S1: test pisze autor")
def test_suggest_stays_within_configured_space():
    """Każda próba ma parametry z przestrzeni z configu; `ngram_range` jest krotką."""


@pytest.mark.skip(reason="W2 S1: test pisze autor")
def test_objective_is_selection_metric():
    """Cel = `evaluation.selection_metric` z configu, liczony na walidacji."""


@pytest.mark.skip(reason="W2 S1: test pisze autor")
def test_select_best_prefers_fewer_features_on_tie():
    """Różnica celu < `tie_tolerance` -> wygrywa próba z mniejszym `n_features`."""
