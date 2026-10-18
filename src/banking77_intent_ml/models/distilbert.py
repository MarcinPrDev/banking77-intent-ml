"""DistilBERT fine-tuning (głowica 77 klas) na zamrożonej walidacji (W2, S3) — run na wariant.

Decyzje autora (12.10.2026): warianty `models.distilbert.variants` (LR {2, 3, 4, 5, 6}e-5,
5 epok; budżet zmieniony z 2–3 na 5 przed wynikami), reszta jak Qi i in. 2021 z configu. Każda
epoka oceniana na walidacji; zwycięzca spośród par (wariant, epoka): najwyższy
`evaluation.selection_metric`, remis (< `tie_tolerance`) -> mniej epok.

Infra (`run_variant`, `run_distilbert`): tokenizacja (`max_len`, dynamiczny padding), Trainer,
wagi po każdej epoce w cache (`paths.cache_dir/distilbert/variant-N/checkpoint-*`, same wagi
z tokenizerem), krzywe per epoka i wynik najlepszej epoki wariantu w MLflow (tabele per klasa
przez wspólne `evaluate()` na predykcjach z ewaluacji tej epoki). Reguła wyboru checkpointu
ustalona przed treningiem; budżet pilnuje CLI (`scripts/train_distilbert.py`).

Stretch 1 (decyzja autora 18.10.2026): `run_seed` / `run_seed_study` — zamrożona konfiguracja
z innymi seedami (`models.distilbert.seed_study` + kontrolnie `models.seed`), wynik po zamrożonej
epoce na walidacji, bez zapisu checkpointów; budżet pilnuje `scripts/seed_study_distilbert.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from mlflow.tracking import MlflowClient
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import evaluate
from banking77_intent_ml.evaluation.tracking import log_validation_run
from banking77_intent_ml.models.hpo import STUDY_TAG

STUDY_NAME = "distilbert_w2"  # wartość tagu `hpo_study` — jedno badanie = wszystkie warianty
CACHE_SUBDIR = "distilbert"
SEED_STUDY_TAG = "seed_study"
SEED_STUDY_NAME = "distilbert_val_seeds"  # jedno badanie = models.seed + seed_study
SEED_STUDY_SUBDIR = "seed-study"


@dataclass(frozen=True)
class EpochScore:
    variant: int  # indeks wariantu w `models.distilbert.variants`
    lr: float
    epoch: int
    objective: float  # evaluation.selection_metric na walidacji po tej epoce
    checkpoint: Path  # wagi (+ tokenizer) po tej epoce, w cache
    run_id: str


@dataclass(frozen=True)
class DistilbertRun:
    variant: int
    lr: float
    run_id: str
    epochs: list[EpochScore]
    train_seconds: float  # trening z ewaluacją po każdej epoce
    peak_vram_gb: float


def training_arguments(
    variant: dict[str, Any], settings: Settings, output_dir: Path
) -> TrainingArguments:
    """Konfiguracja Trainera dla jednego wariantu.

    LR z wariantu, batch/epoki/warmup/weight decay z configu; seed treningu i danych
    z `models.seed`. Ocena i zapis samych wag co epokę, wszystkie checkpointy zachowane.
    Wybór epoki i MLflow obsługuje pętla projektu. Na CUDA używamy BF16, jeśli jest
    obsługiwane; w pozostałych przypadkach FP32, bez CUDA trening odbywa się na CPU.
    Brak procesów roboczych DataLoadera pozwala uruchamiać trening na Windows.
    """
    cfg = settings.models.distilbert
    cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=variant["lr"],
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        warmup_steps=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=None,
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=settings.models.seed,
        data_seed=settings.models.seed,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=False,
        use_cpu=not cuda,
    )


def compute_metrics(
    eval_pred: EvalPrediction, *, settings: Settings, label_map: dict[str, int]
) -> dict[str, float]:
    """Metryki jednej ewaluacji (po epoce) dla Trainera.

    Pierwszy element krotki predykcji to logity, jeśli model zwraca dodatkowe wyjścia.
    Argmax po klasach daje id z mapy; wspólne `evaluate()` uwzględnia wszystkie klasy.
    Zwraca `result.metrics()` — prefiks `eval_` dodaje Trainer.
    """
    logits = eval_pred.predictions
    logits = logits[0] if isinstance(logits, tuple) else logits
    predicted_ids = np.asarray(logits).argmax(axis=-1)
    return evaluate(eval_pred.label_ids, predicted_ids, label_map).metrics()


def select_best_distilbert(scores: list[EpochScore], *, tolerance: float) -> EpochScore:
    """Zwycięzca spośród par (wariant, epoka).

    Reguła ustalona przed treningiem: najwyższy `objective`; różnica < `tolerance` względem
    globalnego maksimum oznacza remis, w którym wygrywa mniej epok. Dalej rozstrzyga wyższy
    cel, niższy numer wariantu, leksykograficznie mniejszy `run_id`, potem ścieżka checkpointu.
    Przy zerowej tolerancji rozpatrujemy tylko dokładne maksima. Kolejność wejścia nie ma
    znaczenia; pusta lista zgłasza ValueError.
    """
    best_value = max(score.objective for score in scores)
    tied = [
        score for score in scores
        if score.objective == best_value or best_value - score.objective < tolerance
    ]
    return min(
        tied,
        key=lambda score: (
            score.epoch, -score.objective, score.variant, score.run_id, score.checkpoint.as_posix()
        ),
    )


def _load_tokenizer(settings: Settings):
    cfg = settings.models.distilbert
    return AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)


def _load_model(settings: Settings, label_map: dict[str, int]):
    cfg = settings.models.distilbert
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        num_labels=len(label_map),
        id2label={index: name for name, index in label_map.items()},
        label2id=dict(label_map),
    )


def _features(texts: pd.Series, labels: pd.Series, tokenizer, max_len: int) -> list[dict]:
    encoded = tokenizer(texts.tolist(), truncation=True, max_length=max_len)
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": int(label)}
        for ids, mask, label in zip(
            encoded["input_ids"], encoded["attention_mask"], labels.tolist(), strict=True
        )
    ]


def run_variant(index: int, data: TrainVal, settings: Settings) -> DistilbertRun:
    """Jeden wariant: trening z oceną po każdej epoce, checkpointy w cache, run w MLflow."""
    cfg = settings.models.distilbert
    variant = cfg.variants[index]
    output_dir = settings.paths.cache_dir / CACHE_SUBDIR / f"variant-{index}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} nie jest pusty — usuń pozostałości po treningu")

    set_seed(settings.models.seed)
    tokenizer = _load_tokenizer(settings)
    args = training_arguments(variant, settings, output_dir)
    predictions: list[np.ndarray] = []  # predykcje każdej ewaluacji (po epoce), w kolejności

    def metrics_fn(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions.append(np.asarray(logits).argmax(axis=-1))
        return compute_metrics(eval_pred, settings=settings, label_map=data.label_map)

    trainer = Trainer(
        model=_load_model(settings, data.label_map),
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=_features(data.x_train, data.y_train, tokenizer, cfg.max_len),
        eval_dataset=_features(data.x_val, data.y_val, tokenizer, cfg.max_len),
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
    )
    cuda = torch.cuda.is_available() and not args.use_cpu
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0

    selection = settings.evaluation.selection_metric
    evals = [entry for entry in trainer.state.log_history if f"eval_{selection}" in entry]
    if len(evals) != len(predictions):
        raise RuntimeError(f"ewaluacji {len(evals)}, predykcji {len(predictions)} — rozjazd")
    best = max(range(len(evals)), key=lambda i: evals[i][f"eval_{selection}"])
    result = evaluate(data.y_val, predictions[best], data.label_map)
    curve_names = sorted({settings.evaluation.primary_metric, selection, "loss"})
    run_id = log_validation_run(
        run_name=f"distilbert-v{index}-lr{variant['lr']:g}",
        params={
            "model": "distilbert",
            "model_id": cfg.model_id,
            "revision": cfg.revision,
            "variant": index,
            "lr": args.learning_rate,
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "max_len": cfg.max_len,
            "warmup_steps": args.warmup_steps,  # ułamek w [0, 1) = proporcja kroków (5.x)
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "bf16": args.bf16,
            "best_epoch": int(round(evals[best]["epoch"])),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        result=result,
        settings=settings,
        tags={"model_family": "distilbert", STUDY_TAG: STUDY_NAME},
        fit_seconds=train_seconds,
        extra_metrics={"peak_vram_gb": peak_vram_gb},
        curves={f"val_{name}": [entry[f"eval_{name}"] for entry in evals] for name in curve_names},
    )
    epochs = [
        EpochScore(
            variant=index,
            lr=float(variant["lr"]),
            epoch=int(round(entry["epoch"])),
            objective=float(entry[f"eval_{selection}"]),
            checkpoint=output_dir / f"checkpoint-{entry['step']}",
            run_id=run_id,
        )
        for entry in evals
    ]
    missing = [score.checkpoint for score in epochs if not score.checkpoint.is_dir()]
    if missing:
        raise RuntimeError(f"brak checkpointów epok: {missing} — save_strategy='epoch'?")
    return DistilbertRun(index, float(variant["lr"]), run_id, epochs, train_seconds, peak_vram_gb)


def run_distilbert(data: TrainVal, settings: Settings) -> list[DistilbertRun]:
    """Wszystkie warianty z configu, po kolei."""
    variants = settings.models.distilbert.variants
    return [run_variant(index, data, settings) for index in range(len(variants))]


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_id: str
    frozen_epoch: int
    metrics: dict[str, float]  # metryki walidacji po zamrożonej epoce (= metryki runu)
    curve: list[float]  # evaluation.selection_metric po kolejnych epokach (informacyjnie)
    train_seconds: float


def run_seed(seed: int, data: TrainVal, settings: Settings) -> SeedRun:
    """Zamrożona konfiguracja z innym seedem treningu i danych (stretch 1).

    Ten sam przepis co `run_variant` dla LR z `frozen.lr`: harmonogram na `epochs` epok, ocena
    po każdej epoce; wynik runu = metryki po epoce `frozen.epoch`, bez wyboru epoki per seed.
    Bez zapisu checkpointów (logi Trainera w `paths.cache_dir/distilbert/seed-study/seed-<seed>`).
    Dla `models.seed` odtwarza zamrożony run W2 — kontrola pętli (`frozen_reference`).
    """
    cfg = settings.models.distilbert
    if cfg.frozen is None:
        raise ValueError("models.distilbert.frozen nie jest ustawione — brak konfiguracji")
    frozen_epoch = int(cfg.frozen["epoch"])
    if not 1 <= frozen_epoch <= cfg.epochs:
        raise ValueError(f"frozen.epoch {frozen_epoch} poza zakresem 1..{cfg.epochs}")
    seeded = replace(settings, models=replace(settings.models, seed=seed))
    output_dir = settings.paths.cache_dir / CACHE_SUBDIR / SEED_STUDY_SUBDIR / f"seed-{seed}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} nie jest pusty — usuń pozostałości po treningu")

    set_seed(seed)
    tokenizer = _load_tokenizer(seeded)
    variant = {"lr": float(cfg.frozen["lr"])}
    args = replace(training_arguments(variant, seeded, output_dir), save_strategy="no")
    predictions: list[np.ndarray] = []

    def metrics_fn(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        predictions.append(np.asarray(logits).argmax(axis=-1))
        return compute_metrics(eval_pred, settings=seeded, label_map=data.label_map)

    trainer = Trainer(
        model=_load_model(seeded, data.label_map),
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=_features(data.x_train, data.y_train, tokenizer, cfg.max_len),
        eval_dataset=_features(data.x_val, data.y_val, tokenizer, cfg.max_len),
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
    )
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start

    selection = settings.evaluation.selection_metric
    evals = [entry for entry in trainer.state.log_history if f"eval_{selection}" in entry]
    epochs = [int(round(entry["epoch"])) for entry in evals]
    if len(evals) != len(predictions) or frozen_epoch not in epochs:
        raise RuntimeError(f"ewaluacje po epokach {epochs}, predykcji {len(predictions)}")
    at = epochs.index(frozen_epoch)
    result = evaluate(data.y_val, predictions[at], data.label_map)
    curve_names = sorted({settings.evaluation.primary_metric, selection, "loss"})
    run_id = log_validation_run(
        run_name=f"distilbert-seed{seed}",
        params={
            "model": "distilbert",
            "model_id": cfg.model_id,
            "revision": cfg.revision,
            "lr": args.learning_rate,
            "epochs": args.num_train_epochs,
            "frozen_epoch": frozen_epoch,
            "batch_size": args.per_device_train_batch_size,
            "max_len": cfg.max_len,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "bf16": args.bf16,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        result=result,
        settings=seeded,
        tags={"model_family": "distilbert", SEED_STUDY_TAG: SEED_STUDY_NAME},
        fit_seconds=train_seconds,
        curves={f"val_{name}": [entry[f"eval_{name}"] for entry in evals] for name in curve_names},
    )
    return SeedRun(
        seed=seed,
        run_id=run_id,
        frozen_epoch=frozen_epoch,
        metrics=result.metrics(),
        curve=[float(entry[f"eval_{selection}"]) for entry in evals],
        train_seconds=train_seconds,
    )


def run_seed_study(data: TrainVal, settings: Settings) -> list[SeedRun]:
    """Kontrolnie `models.seed`, potem `models.distilbert.seed_study`, po kolei."""
    seeds = (settings.models.seed, *settings.models.distilbert.seed_study)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seedy badania muszą być różne: {seeds}")
    return [run_seed(seed, data, settings) for seed in seeds]


def frozen_reference(settings: Settings) -> float:
    """`selection_metric` zamrożonego runu W2 po zamrożonej epoce (wartość z krzywej w MLflow)."""
    cfg = settings.models.distilbert
    client = MlflowClient(settings.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow.experiment)
    if experiment is None:
        raise LookupError(f"brak eksperymentu {settings.mlflow.experiment} w MLflow")
    lr = float(cfg.frozen["lr"])
    runs = [
        run
        for run in client.search_runs(
            [experiment.experiment_id], filter_string=f"tags.{STUDY_TAG} = '{STUDY_NAME}'"
        )
        if float(run.data.params["lr"]) == lr
    ]
    if len(runs) != 1:
        raise LookupError(f"oczekiwano 1 runu {STUDY_NAME} z lr {lr:g}, jest {len(runs)}")
    name = f"val_{settings.evaluation.selection_metric}"
    history = client.get_metric_history(runs[0].info.run_id, name)
    values = [m.value for m in history if m.step == int(cfg.frozen["epoch"])]
    if len(values) != 1:
        raise LookupError(f"brak {name} po epoce {cfg.frozen['epoch']} w {runs[0].info.run_id}")
    return values[0]
