"""Testy LLM zero-shot (models/llm.py, evaluation/llm_eval.py, CLI) — atrapa klienta, bez sieci.

Dowodzą: parsowanie przyjmuje wyłącznie nazwę etykiety (po normalizacji), inne odpowiedzi są błędem
formatu liczonym jako pomyłka; szablon wymaga `{labels}` i `{text}`; cache nie powtarza zapytań;
próbka walidacji = N na klasę, deterministyczna; runy MLflow mają tagi badania i metryki użycia;
CLI odmawia przed jakimkolwiek zapytaniem. Nie dowodzą jakości promptu ani modelu.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation import llm_eval
from banking77_intent_ml.evaluation.final import TestSet as _TestSet  # bez zbierania przez pytest
from banking77_intent_ml.models.llm import (
    INVALID_ID,
    CachedClient,
    LlmPredictions,
    LlmReply,
    parse_label,
    render_prompt,
    usage_metrics,
)

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "llm_zero_shot.py"
TEMPLATE = "Etykiety:\n{labels}\nWiadomość: {text}"
LABELS = {"card_arrival": 0, "top_up_failed": 1, "exchange_rate": 2}


class FakeClient:
    """Odpowiada nazwą wg funkcji tekstu wiadomości; liczy zapytania."""

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> LlmReply:
        self.prompts.append(prompt)
        text = prompt.rsplit("Wiadomość: ", 1)[-1]
        return LlmReply(self.answer_for(text), prompt_tokens=100, completion_tokens=10,
                        latency_s=0.5)


@pytest.fixture
def llm_settings(tmp_mlflow_settings, tmp_path):
    llm = replace(
        tmp_mlflow_settings.llm,
        prompts={"v1": TEMPLATE, "v2": ""},
        val_per_class=2,
        price_input_usd_per_mtok=0.075,
        price_output_usd_per_mtok=0.30,
    )
    paths = replace(tmp_mlflow_settings.paths, cache_dir=tmp_path / "cache")
    return replace(tmp_mlflow_settings, llm=llm, paths=paths)


@pytest.fixture
def toy_data() -> TrainVal:
    names = {index: name for name, index in LABELS.items()}
    y_val = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], index=range(100, 109))
    x_val = pd.Series([f"{names[label]} #{row}" for row, label in y_val.items()], index=y_val.index)
    return TrainVal(
        x_train=pd.Series(["x"]), y_train=pd.Series([0]), x_val=x_val, y_val=y_val,
        label_map=LABELS,
    )


def test_parse_label_accepts_only_a_label_name_after_normalization():
    assert parse_label(' "Card_Arrival". ', LABELS) == 0
    assert parse_label("`top_up_failed`", LABELS) == 1
    assert parse_label("exchange_rate\n", LABELS) == 2
    for answer in ("card arrival", "card_arrival, bo karta", "", "category: card_arrival"):
        assert parse_label(answer, LABELS) == INVALID_ID


def test_render_prompt_lists_labels_and_requires_both_fields():
    prompt = render_prompt(TEMPLATE, "where is my card?", ["card_arrival", "top_up_failed"])
    assert "card_arrival\ntop_up_failed" in prompt and prompt.endswith("where is my card?")
    assert "{" not in prompt
    with pytest.raises(ValueError, match="bez pól"):
        render_prompt("Etykiety: {labels}", "x", ["a"])


def test_cached_client_reuses_replies_and_persists(llm_settings, tmp_path):
    fake = FakeClient(lambda text: "card_arrival")
    path = tmp_path / "cache.jsonl"
    cached = CachedClient(fake, path, llm_settings)

    first = cached.complete("prompt A")
    assert cached.complete("prompt A") == first and len(fake.prompts) == 1
    reloaded = CachedClient(fake, path, llm_settings)
    assert reloaded.complete("prompt A") == first and len(fake.prompts) == 1
    other = replace(llm_settings, llm=replace(llm_settings.llm, temperature=0.6))
    CachedClient(fake, path, other).complete("prompt A")
    assert len(fake.prompts) == 2  # inne parametry = inny klucz
    assert "prompt A" not in path.read_text(encoding="utf-8")  # w cache tylko hash promptu


def test_usage_metrics_count_format_errors_tokens_latency_and_cost(llm_settings):
    replies = [LlmReply("a", 100, 10, 0.2), LlmReply("?", 300, 30, 0.6)]
    usage = usage_metrics(LlmPredictions(np.array([0, INVALID_ID]), replies), llm_settings)

    assert usage["format_errors"] == 1
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (400, 40)
    assert usage["latency_p50_s"] == pytest.approx(0.4)
    assert usage["cost_usd"] == pytest.approx(400 / 1e6 * 0.075 + 40 / 1e6 * 0.30)


def test_val_sample_is_per_class_from_validation_and_deterministic(toy_data, llm_settings):
    ids = llm_eval.val_sample(toy_data.y_val, llm_settings)

    assert ids == llm_eval.val_sample(toy_data.y_val, llm_settings)
    assert set(ids) <= set(toy_data.y_val.index)
    assert toy_data.y_val.loc[ids].value_counts().to_dict() == {0: 2, 1: 2, 2: 2}
    too_many = replace(llm_settings, llm=replace(llm_settings.llm, val_per_class=4))
    with pytest.raises(ValueError, match="< 4"):
        llm_eval.val_sample(toy_data.y_val, too_many)


def test_empty_or_incomplete_template_is_refused(llm_settings):
    with pytest.raises(ValueError, match="szablon pusty"):
        llm_eval.prompt_template(llm_settings, "v2")
    with pytest.raises(ValueError, match="brak wariantu"):
        llm_eval.prompt_template(llm_settings, "v3")


def test_run_val_counts_format_errors_as_misses_and_tags_the_study(toy_data, llm_settings):
    def answer(text):
        name = text.split(" #")[0]
        return "nie wiem" if name == "exchange_rate" and text.endswith(("106", "107")) else name

    run_id, result, usage = llm_eval.run_val("v1", toy_data, FakeClient(answer), llm_settings)

    sample = llm_eval.val_sample(toy_data.y_val, llm_settings)
    errors = sum(1 for row in sample if toy_data.y_val[row] == 2 and row in (106, 107))
    assert usage["format_errors"] == errors
    assert result.accuracy == pytest.approx((len(sample) - errors) / len(sample))
    logged = MlflowClient(llm_settings.mlflow.tracking_uri).get_run(run_id)
    assert logged.data.tags["model_family"] == "llm"
    assert logged.data.tags[llm_eval.VAL_STUDY_TAG] == llm_eval.VAL_STUDY_NAME
    assert logged.data.tags["prompt"] == "v1"
    assert logged.data.metrics["format_errors"] == errors
    assert "#" not in " ".join(logged.data.params.values())  # bez tekstów w parametrach
    assert llm_eval.count_val_runs(llm_settings, "v1") == 1
    assert llm_eval.count_val_runs(llm_settings, "v2") == 0


def test_run_test_logs_llm385_metrics_once_for_the_frozen_prompt(llm_settings, monkeypatch):
    settings = replace(llm_settings, llm=replace(llm_settings.llm, frozen_prompt="v1"))
    names = {index: name for name, index in LABELS.items()}
    y = pd.Series([0, 1, 2, 0], index=[10, 11, 12, 13])
    fake_test = _TestSet(
        x=pd.Series([f"{names[v]} #{k}" for k, v in y.items()], index=y.index), y=y,
        clean_ids=[10, 11, 12, 13], llm_ids=[10, 11, 12], label_map=LABELS,
    )
    monkeypatch.setattr(llm_eval, "load_test_set", lambda s: fake_test)

    run_id, result, _ = llm_eval.run_test(FakeClient(lambda t: t.split(" #")[0]), settings)

    assert result.n_samples == 3 and result.accuracy == 1.0
    logged = MlflowClient(settings.mlflow.tracking_uri).get_run(run_id)
    assert logged.data.tags["split"] == "test" and logged.data.tags["model_family"] == "llm"
    assert logged.data.metrics["llm385_accuracy"] == 1.0
    assert "llm385_cost_usd" in logged.data.metrics
    assert llm_eval.count_llm_test_runs(settings) == 1


def _cli(monkeypatch, settings):
    spec = importlib.util.spec_from_file_location("llm_zero_shot_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "_client", lambda s: pytest.fail("klient utworzony mimo odmowy"))
    monkeypatch.setattr(cli, "load_train_val", lambda s: pytest.fail("dane wczytane mimo odmowy"))
    return cli


@pytest.mark.parametrize(
    ("argv", "frozen", "val_runs", "test_runs", "message"),
    [
        (["--stage", "val", "--prompt", "v1"], None, 1, 0, "budżet"),
        (["--stage", "val", "--prompt", "v2"], None, 0, 0, "szablon pusty"),
        (["--stage", "test"], None, 0, 0, "brak llm.frozen_prompt"),
        (["--stage", "test"], "v1", 0, 0, "nie ma runu walidacyjnego"),
        (["--stage", "test"], "v1", 1, 1, "raz"),
    ],
)
def test_cli_refuses_before_any_request(
    llm_settings, monkeypatch, capsys, argv, frozen, val_runs, test_runs, message
):
    settings = replace(llm_settings, llm=replace(llm_settings.llm, frozen_prompt=frozen))
    cli = _cli(monkeypatch, settings)
    monkeypatch.setattr(cli, "count_val_runs", lambda s, name: val_runs)
    monkeypatch.setattr(cli, "count_llm_test_runs", lambda s: test_runs)
    monkeypatch.setattr("sys.argv", ["llm_zero_shot.py", *argv])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert message in capsys.readouterr().out
