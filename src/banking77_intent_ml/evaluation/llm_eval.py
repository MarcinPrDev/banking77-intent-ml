"""Ocena LLM zero-shot (stretch 2, W4): dobór promptu na walidacji, potem raz test 385.

Decyzje autora (18.10.2026): 2 warianty promptu (`llm.prompts`), każdy raz na 385 zdaniach
walidacji (`llm.val_per_class` na klasę, seed `split.random_state`); zamrożony wariant
(`llm.frozen_prompt`) raz na 385 wiadomościach testu (`llm.eval_ids_path`, wybrane w W1) —
metryki `llm385_*` porównywalne z runami testowymi trzech modeli. Błąd formatu = pomyłka
(`evaluate(..., invalid=INVALID_ID)`). Teksty nie trafiają do MLflow — predykcje jako id.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.data.splits import ids_sha256
from banking77_intent_ml.evaluation.final import count_test_runs, load_test_set
from banking77_intent_ml.evaluation.metrics import EvaluationResult, evaluate
from banking77_intent_ml.evaluation.tracking import configure_mlflow, log_validation_run
from banking77_intent_ml.models.llm import INVALID_ID, LLMClient, predict_texts, usage_metrics

LLM_FAMILY = "llm"  # tag model_family runów LLM (walidacja i test)
VAL_STUDY_TAG = "llm_study"
VAL_STUDY_NAME = "llm_val_prompts"  # badanie doboru promptu: każdy wariant raz
TEST_PREFIX = "llm385"  # jak w runach testowych trzech modeli
TEST_PREDICTIONS_ARTIFACT = "predictions_llm385.csv"


def prompt_template(settings: Settings, name: str) -> str:
    """Szablon wariantu z configu; pusty albo bez `{labels}` / `{text}` = odmowa."""
    prompts = settings.llm.prompts
    if name not in prompts:
        raise ValueError(f"brak wariantu {name!r} w llm.prompts (są: {sorted(prompts)})")
    template = prompts[name]
    if "{labels}" not in template or "{text}" not in template:
        raise ValueError(f"llm.prompts.{name}: szablon pusty albo bez pól {{labels}} / {{text}}")
    return template


def val_sample(y_val: pd.Series, settings: Settings) -> list[int]:
    """Id wierszy walidacji: `llm.val_per_class` na klasę, seed `split.random_state`."""
    per_class = settings.llm.val_per_class
    chosen: list[int] = []
    for label, group in y_val.groupby(y_val, sort=True):
        if len(group) < per_class:
            raise ValueError(f"klasa {label}: {len(group)} zdań walidacji < {per_class}")
        chosen.extend(group.sample(n=per_class, random_state=settings.split.random_state).index)
    return sorted(int(row_id) for row_id in chosen)


def count_val_runs(settings: Settings, name: str) -> int:
    """Runy walidacyjne wariantu promptu (budżet: raz)."""
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=f"tags.{VAL_STUDY_TAG} = '{VAL_STUDY_NAME}' and tags.prompt = '{name}'",
    )
    return len(runs)


def count_llm_test_runs(settings: Settings) -> int:
    return count_test_runs(settings, LLM_FAMILY)


def _params(settings: Settings, name: str, template: str) -> dict[str, Any]:
    cfg = settings.llm
    try:
        litellm_version = version("litellm")
    except PackageNotFoundError:
        litellm_version = "brak"
    return {
        "model": cfg.model,
        "prompt": name,
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "temperature": cfg.temperature,
        "reasoning_effort": cfg.reasoning_effort,
        "include_reasoning": cfg.include_reasoning,
        "max_completion_tokens": cfg.max_completion_tokens,
        "litellm": litellm_version,
    }


def run_val(
    name: str, data: TrainVal, client: LLMClient, settings: Settings
) -> tuple[str, EvaluationResult, dict[str, float]]:
    """Wariant promptu na próbce walidacji; run MLflow z tagiem badania i wariantu."""
    template = prompt_template(settings, name)
    ids = val_sample(data.y_val, settings)
    predictions = predict_texts(data.x_val.loc[ids].tolist(), template, client, data.label_map)
    result = evaluate(
        data.y_val.loc[ids].to_numpy(), predictions.ids, data.label_map, invalid=INVALID_ID
    )
    usage = usage_metrics(predictions, settings)
    run_id = log_validation_run(
        run_name=f"llm-{name}-val",
        params={
            **_params(settings, name, template),
            "n_val_sample": len(ids),
            "sha256_val_sample": ids_sha256(ids),
        },
        result=result,
        settings=settings,
        tags={"model_family": LLM_FAMILY, VAL_STUDY_TAG: VAL_STUDY_NAME, "prompt": name},
        extra_metrics=usage,
    )
    return run_id, result, usage


def run_test(
    client: LLMClient, settings: Settings
) -> tuple[str, EvaluationResult, dict[str, float]]:
    """Zamrożony prompt raz na 385 wiadomościach testu; metryki `llm385_*`, predykcje jako id."""
    name = settings.llm.frozen_prompt
    if not name:
        raise ValueError("llm.frozen_prompt nie ustawiony — najpierw dobór promptu na walidacji")
    template = prompt_template(settings, name)
    test = load_test_set(settings)
    ids = test.llm_ids
    predictions = predict_texts(test.x.loc[ids].tolist(), template, client, test.label_map)
    y_true = test.y.loc[ids].to_numpy()
    result = evaluate(y_true, predictions.ids, test.label_map, invalid=INVALID_ID)
    usage = usage_metrics(predictions, settings)
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"test-llm-{name}") as run:
        mlflow.set_tags({
            "model_family": LLM_FAMILY,
            "split": "test",
            "stage": "final",
            "prompt": name,
            "data_commit": settings.dataset.commit,
            "sha256_llm_eval_ids": manifest["sha256_llm_eval_ids"],
        })
        mlflow.log_params({**_params(settings, name, template), f"n_{TEST_PREFIX}": len(ids)})
        mlflow.log_metrics({
            **{f"{TEST_PREFIX}_{key}": value for key, value in result.metrics().items()},
            **{f"{TEST_PREFIX}_{key}": value for key, value in usage.items()},
        })
        mlflow.log_table(result.per_class.reset_index(), artifact_file="per_class_llm385.json")
        mlflow.log_table(result.confusion.reset_index(), artifact_file="confusion_llm385.json")
        frame = pd.DataFrame(
            {"row_id": ids, "true_id": y_true, "predicted_id": predictions.ids}
        )
        with TemporaryDirectory(prefix="llm-predictions-") as temporary:
            path = Path(temporary) / TEST_PREDICTIONS_ARTIFACT
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        return run.info.run_id, result, usage
