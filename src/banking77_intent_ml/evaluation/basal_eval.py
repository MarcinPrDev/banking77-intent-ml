"""Dodatek W5: zero-shot Basal — kandydaci na walidacji, wybór regułą z prerejestracji, test raz.

Prerejestracja (config `basal`, decyzje autora 30.10.2026, commit przed obliczeniami): kandydaci =
modele `basal.models` × warianty opisów `basal.label_variants`; każdy raz na tej samej próbce
walidacji co LLM (`llm_eval.val_sample`). Wybór: najwyższe macro-F1; kandydaci bliżej najlepszego
niż `tie_tolerance` = remis → tańszy (mniejszy `cost_rank`, potem wariant opisów w kolejności
nazw, czyli v1 przed v2). Zamrożony kandydat (`frozen_model`, `frozen_labels` — wpis autora) raz
na teście: pełne 3 080 (`test_*`) i 385 z W1 (`llm385_*`). Teksty nie trafiają do MLflow.

Badanie 2 (config `basal_ultra`, decyzje autora 31.10.2026, commit przed obliczeniami): ten sam
protokół dla basal-omni-1.0-ultra (serwer na wynajętym GPU, klient przez tunel SSH). Badania
rozdziela tag runów walidacyjnych (`<badanie>_val_candidates`), a budżet testu (raz) liczy się
na kandydatach danego badania — run testowy 4,5B nie blokuje ultra i odwrotnie.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.config.settings import BasalConfig
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.data.splits import ids_sha256
from banking77_intent_ml.evaluation.final import load_test_set
from banking77_intent_ml.evaluation.llm_eval import val_sample
from banking77_intent_ml.evaluation.metrics import EvaluationResult, evaluate
from banking77_intent_ml.evaluation.tracking import configure_mlflow, log_validation_run
from banking77_intent_ml.models.basal import BasalClient, BasalPredictions, predict_texts

FAMILY = "basal"  # tag model_family runów Basal (walidacja i test)
STUDIES = ("basal", "basal_ultra")  # sekcje configu: badanie 1 (4,5B), badanie 2 (ultra)
VAL_STUDY_TAG = "basal_study"  # wartość: `val_study_name(badanie)`; każdy kandydat raz
TEST_STUDY_TAG = "basal_test_study"  # badanie runu testowego (inny klucz niż w walidacji)
TEST_PREFIX_FULL = "test"  # pełne 3 080
TEST_PREFIX_LLM = "llm385"  # te same 385 co LLM i trzy modele
TEST_PREDICTIONS_ARTIFACT = "predictions_basal_test.csv"


def study_config(settings: Settings, study: str = "basal") -> BasalConfig:
    """Sekcja configu badania; nieznane badanie albo brak sekcji = odmowa."""
    if study not in STUDIES:
        raise ValueError(f"nieznane badanie {study!r}, dostępne: {STUDIES}")
    cfg = getattr(settings, study)
    if cfg is None:
        raise ValueError(f"brak sekcji {study} w configu")
    return cfg


def val_study_name(study: str = "basal") -> str:
    """Tag runów walidacyjnych badania; dla badania 1 = `basal_val_candidates` (jego runy)."""
    return f"{study}_val_candidates"


def _label_names(settings: Settings) -> list[str]:
    labels = json.loads(settings.dataset.label_map_path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]


def label_criteria(settings: Settings, variant: str, study: str = "basal") -> dict[str, str]:
    """Opisy 77 intencji wariantu; klucze = nazwy z mapy etykiet, puste opisy = odmowa."""
    variants = study_config(settings, study).label_variants
    if variant not in variants:
        raise ValueError(f"brak wariantu {variant!r} w {study}.label_variants: {sorted(variants)}")
    path = variants[variant]
    criteria = json.loads(path.read_text(encoding="utf-8"))
    if list(criteria) != _label_names(settings):
        raise ValueError(f"{path.name}: klucze inne niż nazwy z mapy etykiet (kolejność id)")
    empty = [name for name, text in criteria.items() if not str(text).strip()]
    if empty:
        raise ValueError(f"{path.name}: {len(empty)} pustych opisów, np. {empty[0]!r}")
    return {name: str(text).strip() for name, text in criteria.items()}


def instructions(settings: Settings, study: str = "basal") -> str:
    """Treść pytania z configu; pusta = odmowa (pisze autor)."""
    text = study_config(settings, study).instructions.strip()
    if not text:
        raise ValueError(f"{study}.instructions puste — treść pytania pisze autor")
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params(settings: Settings, candidate: str, variant: str, study: str) -> dict[str, Any]:
    cfg = study_config(settings, study)
    model = cfg.models[candidate]
    criteria = label_criteria(settings, variant, study)
    params = {
        "candidate": candidate,
        "repo": model.repo,
        "revision": model.revision,
        "cost_rank": model.cost_rank,
        "labels": variant,
        "labels_sha256": _sha256(json.dumps(criteria, ensure_ascii=False)),
        "instructions_sha256": _sha256(instructions(settings, study)),
        "engine_version": cfg.engine_version,
        "serve_args": cfg.serve_args,
    }
    if cfg.hardware:
        params["hardware"] = cfg.hardware
    return params


def latency_metrics(predictions: BasalPredictions) -> dict[str, float]:
    """p50/p95 czasu żądania (klient, HTTP) i czasu po stronie serwera, w ms."""
    wall = np.asarray(predictions.wall_ms)
    server = np.asarray(predictions.server_ms)
    return {
        "latency_wall_p50_ms": float(np.percentile(wall, 50)),
        "latency_wall_p95_ms": float(np.percentile(wall, 95)),
        "latency_server_p50_ms": float(np.nanpercentile(server, 50)),
        "latency_server_p95_ms": float(np.nanpercentile(server, 95)),
    }


def count_val_runs(settings: Settings, candidate: str, variant: str, study: str = "basal") -> int:
    """Runy walidacyjne kandydata w badaniu (budżet: raz)."""
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=(
            f"tags.{VAL_STUDY_TAG} = '{val_study_name(study)}' and tags.candidate = '{candidate}' "
            f"and tags.labels = '{variant}'"
        ),
    )
    return len(runs)


def count_basal_test_runs(settings: Settings, study: str = "basal") -> int:
    """Runy testowe kandydatów badania (budżet: raz na badanie)."""
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=f"tags.split = 'test' and tags.model_family = '{FAMILY}'",
    )
    if runs.empty:
        return 0
    return int(runs["tags.candidate"].isin(list(study_config(settings, study).models)).sum())


def run_val(
    candidate: str,
    variant: str,
    data: TrainVal,
    client: BasalClient,
    settings: Settings,
    study: str = "basal",
) -> tuple[str, EvaluationResult, dict[str, float]]:
    """Kandydat na próbce walidacji; run MLflow z tagiem badania, kandydata i wariantu opisów."""
    if candidate not in study_config(settings, study).models:
        raise ValueError(f"brak kandydata {candidate!r} w {study}.models")
    criteria = label_criteria(settings, variant, study)
    ids = val_sample(data.y_val, settings)
    predictions = predict_texts(
        data.x_val.loc[ids].tolist(), criteria, instructions(settings, study), client,
        data.label_map,
    )
    result = evaluate(data.y_val.loc[ids].to_numpy(), predictions.ids, data.label_map)
    latency = latency_metrics(predictions)
    run_id = log_validation_run(
        run_name=f"basal-{candidate}-{variant}-val",
        params={
            **_params(settings, candidate, variant, study),
            "n_val_sample": len(ids),
            "sha256_val_sample": ids_sha256(ids),
        },
        result=result,
        settings=settings,
        tags={
            "model_family": FAMILY, VAL_STUDY_TAG: val_study_name(study),
            "candidate": candidate, "labels": variant,
        },
        extra_metrics=latency,
    )
    return run_id, result, latency


@dataclass(frozen=True)
class Candidate:
    candidate: str
    labels: str
    macro_f1: float
    accuracy: float
    cost_rank: int
    run_id: str


def choose(candidates: Sequence[Candidate], tie_tolerance: float) -> Candidate:
    """Reguła z prerejestracji: najwyższe macro-F1; remis (< `tie_tolerance`) → tańszy, potem v1."""
    if not candidates:
        raise ValueError("brak kandydatów")
    best = max(candidate.macro_f1 for candidate in candidates)
    tied = [candidate for candidate in candidates if best - candidate.macro_f1 < tie_tolerance]
    return min(tied, key=lambda candidate: (candidate.cost_rank, candidate.labels))


def val_candidates(settings: Settings, study: str = "basal") -> list[Candidate]:
    """Wyniki walidacji badania z MLflow; brak albo powtórzenie któregoś kandydata = odmowa."""
    cfg = study_config(settings, study)
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=f"tags.{VAL_STUDY_TAG} = '{val_study_name(study)}'",
    )
    expected = {(candidate, variant) for candidate in cfg.models for variant in cfg.label_variants}
    found: dict[tuple[str, str], Candidate] = {}
    for _, run in runs.iterrows():
        key = (run["tags.candidate"], run["tags.labels"])
        if key in found:
            raise ValueError(f"kandydat {key} ma więcej niż jeden run walidacyjny")
        found[key] = Candidate(
            candidate=key[0],
            labels=key[1],
            macro_f1=float(run["metrics.macro_f1"]),
            accuracy=float(run["metrics.accuracy"]),
            cost_rank=cfg.models[key[0]].cost_rank,
            run_id=str(run["run_id"]),
        )
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"brak runów walidacyjnych dla: {missing}")
    return sorted(found.values(), key=lambda candidate: -candidate.macro_f1)


def run_test(
    client: BasalClient, settings: Settings, study: str = "basal"
) -> tuple[str, dict[str, EvaluationResult], dict[str, float]]:
    """Zamrożony kandydat badania raz na teście: pełne 3 080 i podzbiór 385; predykcje jako id."""
    cfg = study_config(settings, study)
    candidate, variant = cfg.frozen_model, cfg.frozen_labels
    if not candidate or not variant:
        raise ValueError(f"{study}.frozen_model / frozen_labels nie ustawione — najpierw walidacja")
    criteria = label_criteria(settings, variant, study)
    test = load_test_set(settings)
    predictions = predict_texts(
        test.x.tolist(), criteria, instructions(settings, study), client, test.label_map
    )
    y_true = test.y.to_numpy()
    y_pred = np.asarray(predictions.ids)
    in_llm385 = test.x.index.isin(test.llm_ids)
    results = {
        TEST_PREFIX_FULL: evaluate(y_true, y_pred, test.label_map),
        TEST_PREFIX_LLM: evaluate(y_true[in_llm385], y_pred[in_llm385], test.label_map),
    }
    latency = latency_metrics(predictions)
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"test-basal-{candidate}-{variant}") as run:
        mlflow.set_tags({
            "model_family": FAMILY,
            "split": "test",
            "stage": "extension",
            TEST_STUDY_TAG: study,
            "candidate": candidate,
            "labels": variant,
            "data_commit": settings.dataset.commit,
            "sha256_llm_eval_ids": manifest["sha256_llm_eval_ids"],
        })
        mlflow.log_params({
            **_params(settings, candidate, variant, study),
            f"n_{TEST_PREFIX_FULL}": len(y_true),
            f"n_{TEST_PREFIX_LLM}": int(in_llm385.sum()),
        })
        mlflow.log_metrics({
            **{
                f"{prefix}_{key}": value
                for prefix, result in results.items()
                for key, value in result.metrics().items()
            },
            **latency,
        })
        for prefix, result in results.items():
            mlflow.log_table(
                result.per_class.reset_index(), artifact_file=f"per_class_basal_{prefix}.json"
            )
        frame = pd.DataFrame({
            "row_id": test.x.index.to_numpy(),
            "true_id": y_true,
            "predicted_id": y_pred,
            "score": predictions.scores,
        })
        with TemporaryDirectory(prefix="basal-predictions-") as temporary:
            path = Path(temporary) / TEST_PREDICTIONS_ARTIFACT
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        return run.info.run_id, results, latency
