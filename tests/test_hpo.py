"""Testy HPO TF-IDF + LR (models/hpo.py, W2 S1).

Infra: każda próba = osobny run MLflow z tagami badania; blokada budżetu w CLI (drugie badanie
odmawia przed treningiem). Przestrzeń, cel i wybór zwycięzcy sprawdzane bez danych BANKING77.
"""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from dataclasses import replace
from itertools import permutations, product
from pathlib import Path

import optuna
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.config.settings import TfidfLrHpoConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import evaluate
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
        assert run.data.metrics["n_features"] == trial.n_features


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


# --- Przestrzeń, cel i reguła wyboru (W2 S1) --------------------------------------------------


@pytest.mark.parametrize("custom_space", [False, True])
def test_suggest_stays_within_configured_space(custom_space):
    """Odczyt przestrzeni, rozkład log(C), krotki i brak mutacji konfiguracji bazowej."""
    settings = load_settings()
    base = settings.models.tfidf_lr
    if custom_space:
        space = {
            "representation": {"word_trigrams": {"analyzer": "word", "ngram_range": [1, 3]}},
            "min_df": [3],
            "sublinear_tf": [True],
            "C": {"low": 0.25, "high": 4.0},
            "class_weight": ["balanced"],
        }
        base = replace(base, hpo=replace(base.hpo, space=space))
        settings = replace(settings, models=replace(settings.models, tfidf_lr=base))
    space = base.hpo.space
    original = deepcopy(base)
    for representation, min_df, sublinear_tf, c, class_weight in product(
        space["representation"], space["min_df"], space["sublinear_tf"],
        [space["C"]["low"], space["C"]["high"]], space["class_weight"],
    ):
        trial = optuna.trial.FixedTrial({
            "representation": representation, "min_df": min_df, "sublinear_tf": sublinear_tf,
            "C": c, "class_weight": class_weight,
        })
        config = hpo.suggest_tfidf_lr(trial, settings)

        expected_tfidf = {
            **original.tfidf, **space["representation"][representation],
            "ngram_range": tuple(space["representation"][representation]["ngram_range"]),
            "min_df": min_df, "sublinear_tf": sublinear_tf,
        }
        expected_logreg = {**original.logreg, "C": c, "class_weight": class_weight}
        assert config == replace(original, tfidf=expected_tfidf, logreg=expected_logreg)
        assert config is not base
        assert config.tfidf is not base.tfidf
        assert config.logreg is not base.logreg
        assert base == original
        assert isinstance(config.tfidf["ngram_range"], tuple)
        for name in ("representation", "min_df", "sublinear_tf", "class_weight"):
            assert trial.distributions[name] == optuna.distributions.CategoricalDistribution(
                list(space[name])
            )
        assert trial.distributions["C"] == optuna.distributions.FloatDistribution(
            space["C"]["low"], space["C"]["high"], log=True
        )


@pytest.mark.parametrize("metric, expected", [("macro_f1", 3 / 7), ("accuracy", 3 / 4)])
def test_objective_is_selection_metric(metric, expected):
    """Cel = `evaluation.selection_metric` z configu, liczony na walidacji."""
    settings = load_settings()
    settings = replace(settings, evaluation=replace(settings.evaluation, selection_metric=metric))
    result = evaluate([0, 0, 0, 1], [0, 0, 0, 0], {"a": 0, "b": 1})

    assert hpo.objective_value(result, settings) == pytest.approx(expected)


@pytest.mark.parametrize(("scores", "features", "tolerance", "winner"), [
    ([0.9, 0.8995, 0.8988], [100, 80, 10], 0.001, 1),  # bez łańcucha remisów
    ([0.9, 0.898], [100, 10], 0.001, 0),  # cel ważniejszy od rozmiaru
    ([0.9, 0.9], [100, 10], 0.001, 1),  # dokładny remis
    ([0.875, 0.75], [100, 10], 0.125, 0),  # granica: <, nie <= (dokładne floaty)
    ([0.9], [100], 0.001, 0),
    ([0.9, 0.898], [100, 10], 0.0, 0),  # zerowa tolerancja zachowuje maksimum
])
def test_select_best_prefers_fewer_features_on_tie(scores, features, tolerance, winner):
    """Remis względem globalnego maksimum; wynik niezależny od kolejności prób."""
    settings = load_settings()
    result = evaluate([0, 1], [0, 1], {"a": 0, "b": 1})
    trials = [
        hpo.HpoTrial(i, f"run-{i}", settings.models.tfidf_lr, result, score, n_features)
        for i, (score, n_features) in enumerate(zip(scores, features, strict=True))
    ]

    for ordered in permutations(trials):
        assert hpo.select_best(list(ordered), tolerance=tolerance) is trials[winner]


def test_select_best_rejects_empty_trials():
    with pytest.raises(ValueError):
        hpo.select_best([], tolerance=0.001)
