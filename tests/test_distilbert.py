"""Testy pętli DistilBERT (models/distilbert.py, W2 S3) — malutki model na CPU, bez pobierania.

Infra: run MLflow na wariant z krzywymi per epoka, checkpoint każdej epoki w cache, odmowa przy
brudnym katalogu, blokada budżetu w CLI. Funkcje autora zastąpione w tych testach prostymi
odpowiednikami; ich własne testy (konfiguracja, metryki, reguła wyboru) — `skip`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

# CI instaluje tylko extra `dev` (bez torch/transformers) — tam cały moduł jest pomijany.
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from transformers import (  # noqa: E402
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    TrainingArguments,
)

from banking77_intent_ml.data.datasets import TrainVal  # noqa: E402
from banking77_intent_ml.evaluation.metrics import evaluate  # noqa: E402
from banking77_intent_ml.evaluation.tracking import count_runs_with_tag  # noqa: E402
from banking77_intent_ml.models import distilbert  # noqa: E402
from banking77_intent_ml.models.hpo import STUDY_TAG  # noqa: E402

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "train_distilbert.py"
_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "top", "up", "card", "rate"]


@pytest.fixture
def toy_data() -> TrainVal:
    return TrainVal(
        x_train=pd.Series(["top up", "top up now", "card", "card late", "rate", "rate fee"]),
        y_train=pd.Series([0, 0, 1, 1, 2, 2]),
        x_val=pd.Series(["top", "card now", "rate today"]),
        y_val=pd.Series([0, 1, 2]),
        label_map={"a": 0, "b": 1, "c": 2},
    )


@pytest.fixture
def tiny(tmp_path, tmp_mlflow_settings, monkeypatch):
    """Malutki tokenizer i model zamiast pobierania; funkcje autora zastąpione prostymi."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(_VOCAB), encoding="utf-8")
    monkeypatch.setattr(
        distilbert, "_load_tokenizer", lambda settings: BertTokenizerFast(vocab_file=str(vocab))
    )

    def load_model(settings, label_map):
        config = DistilBertConfig(
            vocab_size=len(_VOCAB), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
            max_position_embeddings=32, num_labels=len(label_map),
        )
        return DistilBertForSequenceClassification(config)

    def cpu_args(variant, settings, output_dir):
        return TrainingArguments(
            output_dir=str(output_dir), eval_strategy="epoch", save_strategy="epoch",
            save_only_model=True, num_train_epochs=2, per_device_train_batch_size=2,
            per_device_eval_batch_size=4, learning_rate=variant["lr"], report_to="none",
            use_cpu=True, seed=settings.models.seed, disable_tqdm=True,
            dataloader_num_workers=0,
        )

    def metrics(eval_pred, *, settings, label_map):
        logits = np.asarray(eval_pred.predictions)
        return evaluate(eval_pred.label_ids, logits.argmax(axis=-1), label_map).metrics()

    monkeypatch.setattr(distilbert, "_load_model", load_model)
    monkeypatch.setattr(distilbert, "training_arguments", cpu_args)
    monkeypatch.setattr(distilbert, "compute_metrics", metrics)
    model_cfg = replace(
        tmp_mlflow_settings.models.distilbert,
        variants=({"lr": 1e-3}, {"lr": 5e-4}), configs=2, max_len=16,
    )
    return replace(
        tmp_mlflow_settings,
        paths=replace(tmp_mlflow_settings.paths, cache_dir=tmp_path / "cache"),
        models=replace(tmp_mlflow_settings.models, distilbert=model_cfg),
    )


def test_variant_logs_curves_and_keeps_epoch_checkpoints(toy_data, tiny):
    run = distilbert.run_variant(0, toy_data, tiny)

    assert [score.epoch for score in run.epochs] == [1, 2]
    assert all(score.checkpoint.is_dir() for score in run.epochs)
    client = MlflowClient(tiny.mlflow.tracking_uri)
    logged = client.get_run(run.run_id)
    assert logged.data.tags["model_family"] == "distilbert"
    assert logged.data.tags[STUDY_TAG] == distilbert.STUDY_NAME
    assert logged.data.params["lr"] == str(1e-3)
    history = client.get_metric_history(run.run_id, "val_macro_f1")
    assert [m.step for m in sorted(history, key=lambda m: m.step)] == [1, 2]
    by_step = {m.step: m.value for m in history}
    assert [score.objective for score in run.epochs] == pytest.approx([by_step[1], by_step[2]])


def test_variant_refuses_dirty_output_dir(toy_data, tiny):
    leftover = tiny.paths.cache_dir / distilbert.CACHE_SUBDIR / "variant-0" / "checkpoint-1"
    leftover.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="nie jest pusty"):
        distilbert.run_variant(0, toy_data, tiny)


def test_all_variants_are_logged(toy_data, tiny):
    runs = distilbert.run_distilbert(toy_data, tiny)

    assert [run.variant for run in runs] == [0, 1]
    assert count_runs_with_tag(tiny, STUDY_TAG, distilbert.STUDY_NAME) == 2


def test_cli_refuses_second_study_before_training(tmp_mlflow_settings, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("train_distilbert_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: tmp_mlflow_settings)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda *args: 5)
    monkeypatch.setattr(
        cli, "load_train_val", lambda *args: pytest.fail("dane wczytane mimo zużytego budżetu")
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "budżet" in capsys.readouterr().out


# --- Testy autora (W2 S3): treść pisze autor, potem usuwa `skip` -------------------------------


@pytest.mark.skip(reason="W2 S3: test pisze autor")
def test_training_arguments_follow_config_and_variant():
    """LR z wariantu, reszta z configu; ocena i zapis co epokę, same wagi, bez load_best."""


@pytest.mark.skip(reason="W2 S3: test pisze autor")
def test_compute_metrics_uses_common_evaluate():
    """Logity -> id klas -> `evaluate()`; klucze jak w `result.metrics()`."""


@pytest.mark.skip(reason="W2 S3: test pisze autor")
def test_select_best_distilbert_rule():
    """Najwyższy cel; remis (< tolerancji) -> mniej epok; niezależnie od kolejności."""
