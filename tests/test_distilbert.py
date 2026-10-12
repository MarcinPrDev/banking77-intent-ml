"""Testy pętli DistilBERT (models/distilbert.py, W2 S3) — malutki model na CPU, bez pobierania.

Infra: run MLflow na wariant z krzywymi per epoka, checkpoint każdej epoki w cache, odmowa przy
brudnym katalogu, blokada budżetu w CLI. Konfiguracja, metryki i reguła wyboru sprawdzane
bez pobierania wag i bez danych BANKING77.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from itertools import permutations
from math import ceil
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
    EvalPrediction,
    TrainingArguments,
)

from banking77_intent_ml.config import load_settings  # noqa: E402
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
    """Malutki tokenizer i model na CPU; rzeczywiste argumenty i metryki projektu."""
    # transformers 5: słownik przez `vocab` (dict); `vocab_file` w konstruktorze jest pomijany
    vocab = {token: index for index, token in enumerate(_VOCAB)}
    monkeypatch.setattr(
        distilbert, "_load_tokenizer", lambda settings: BertTokenizerFast(vocab=vocab)
    )

    def load_model(settings, label_map):
        config = DistilBertConfig(
            vocab_size=len(_VOCAB), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
            max_position_embeddings=32, num_labels=len(label_map),
        )
        return DistilBertForSequenceClassification(config)

    make_args = distilbert.training_arguments

    def cpu_args(variant, settings, output_dir):
        return replace(make_args(variant, settings, output_dir), disable_tqdm=True)

    monkeypatch.setattr(distilbert, "_load_model", load_model)
    monkeypatch.setattr(distilbert.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(distilbert, "training_arguments", cpu_args)
    model_cfg = replace(
        tmp_mlflow_settings.models.distilbert,
        variants=({"lr": 1e-3}, {"lr": 5e-4}), configs=2, max_len=16, epochs=2, batch_size=2,
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
    for score in run.epochs:
        assert (score.checkpoint / "model.safetensors").is_file()
        assert (score.checkpoint / "tokenizer_config.json").is_file()
        assert not (score.checkpoint / "optimizer.pt").exists()
        assert not (score.checkpoint / "scheduler.pt").exists()
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


# --- Konfiguracja, metryki i reguła wyboru (W2 S3) --------------------------------------------


@pytest.mark.parametrize("custom_config", [False, True])
@pytest.mark.parametrize("precision", ["native", "cpu", "no_bf16"])
def test_training_arguments_follow_config_and_variant(
    tmp_path, monkeypatch, custom_config, precision
):
    """LR z wariantu, reszta z configu; ocena i zapis co epokę, same wagi, bez load_best."""
    settings = load_settings()
    if custom_config:
        cfg = replace(
            settings.models.distilbert, epochs=3, batch_size=7, warmup_ratio=0.25,
            weight_decay=0.03, lr=0.05,
        )
        settings = replace(settings, models=replace(settings.models, seed=17, distilbert=cfg))
    cfg = settings.models.distilbert
    variant = {"lr": 7e-5}
    if precision == "cpu":
        monkeypatch.setattr(distilbert.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            distilbert.torch.cuda, "is_bf16_supported",
            lambda: pytest.fail("sprawdzenie bf16 CUDA mimo braku CUDA"),
        )
    elif precision == "no_bf16":
        monkeypatch.setattr(distilbert.torch.cuda, "is_bf16_supported", lambda: False)
    cuda = distilbert.torch.cuda.is_available()
    expected_bf16 = cuda and distilbert.torch.cuda.is_bf16_supported()

    args = distilbert.training_arguments(variant, settings, tmp_path)

    assert isinstance(args, TrainingArguments)
    assert args.output_dir == str(tmp_path)
    assert args.learning_rate == variant["lr"]
    assert args.num_train_epochs == cfg.epochs
    assert args.per_device_train_batch_size == cfg.batch_size
    assert args.per_device_eval_batch_size == cfg.batch_size
    assert args.warmup_steps == cfg.warmup_ratio
    assert getattr(args, "warmup_ratio", None) is None
    assert args.get_warmup_steps(100) == ceil(100 * cfg.warmup_ratio)
    assert args.weight_decay == cfg.weight_decay
    assert args.seed == args.data_seed == settings.models.seed
    assert args.eval_strategy == args.save_strategy == "epoch"
    assert args.save_only_model is True
    assert args.save_total_limit is None
    assert args.load_best_model_at_end is False
    assert args.report_to == []
    assert args.dataloader_num_workers == 0
    assert args.bf16 is expected_bf16
    assert args.fp16 is False
    assert args.use_cpu is (not cuda)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_compute_metrics_uses_common_evaluate(as_tuple):
    """Logity -> id klas -> `evaluate()`; klucze jak w `result.metrics()`."""
    logits = np.array([[8, -2, -5], [3, 2, -4], [7, 1, -3], [-2, 4, -8]])
    labels = np.array([0, 0, 1, 1])
    label_map = {"b": 1, "absent": 2, "a": 0}
    predictions = (logits, np.zeros((4, 2))) if as_tuple else logits
    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

    metrics = distilbert.compute_metrics(eval_pred, settings=load_settings(), label_map=label_map)

    assert metrics == evaluate(labels, [0, 0, 0, 1], label_map).metrics()
    assert metrics == pytest.approx({"accuracy": 3 / 4, "macro_f1": 22 / 45})


@pytest.mark.parametrize(("points", "tolerance", "winner"), [
    ([(0, 5, 0.9), (1, 3, 0.8995), (2, 1, 0.8988)], 0.001, 1),  # bez łańcucha remisów
    ([(0, 5, 0.9), (1, 1, 0.898)], 0.001, 0),  # jakość ponad liczbą epok
    ([(0, 5, 0.9), (1, 3, 0.9)], 0.001, 1),
    ([(0, 5, 0.875), (1, 1, 0.75)], 0.125, 0),  # ścisła granica na dokładnych floatach
    ([(0, 3, 0.8995), (1, 3, 0.9)], 0.001, 1),  # przy tylu samych epokach wyższy cel
    ([(1, 3, 0.9), (0, 3, 0.9)], 0.001, 1),  # potem niższy numer wariantu
    ([(0, 3, 0.9), (0, 3, 0.9)], 0.001, 0),  # potem run_id
    ([(0, 5, 0.9), (1, 1, 0.8995)], 0.0, 0),
    ([(0, 5, 0.9), (1, 3, 0.9)], 0.0, 1),
    ([(0, 1, 0.9)], 0.001, 0),
    ([], 0.001, None),
])
def test_select_best_distilbert_rule(points, tolerance, winner):
    """Najwyższy cel; remis (< tolerancji) -> mniej epok; niezależnie od kolejności."""
    scores = [
        distilbert.EpochScore(variant, 2e-5, epoch, objective, Path(f"checkpoint-{i}"), f"run-{i}")
        for i, (variant, epoch, objective) in enumerate(points)
    ]
    if winner is None:
        with pytest.raises(ValueError):
            distilbert.select_best_distilbert(scores, tolerance=tolerance)
        return

    for ordered in permutations(scores):
        best = distilbert.select_best_distilbert(list(ordered), tolerance=tolerance)
        assert best is scores[winner]
