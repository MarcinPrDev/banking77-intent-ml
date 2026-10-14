"""Testy pomiarów kosztu (evaluation/cost.py, cost_worker.py, W2 N2) — bez prawdziwych artefaktów.

Losowanie zdań, podsumowania (percentyle, mediany), środowisko procesu (wątki, tylko CPU), rozmiar
katalogu, run MLflow; integracyjnie: prawdziwe procesy pomiarowe na malutkim artefakcie TF-IDF+LR
(bez torch, działa w CI). CLI odmawia przed wczytaniem danych, gdy pomiar oficjalny już był.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation import cost
from banking77_intent_ml.models.classifiers import TfidfLrClassifier

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "measure_cost.py"


def _with_cost(settings, **changes):
    cost_cfg = replace(settings.evaluation.cost, **changes)
    return replace(settings, evaluation=replace(settings.evaluation, cost=cost_cfg))


def test_sample_is_seeded_subset_of_val_in_val_order():
    settings = _with_cost(load_settings(), n_texts=4)
    index = [100, 104, 108, 112, 116, 120, 124]  # id wierszy jak w TrainVal, nie 0..n-1
    data = TrainVal(
        x_train=pd.Series(["t"]),
        y_train=pd.Series([0]),
        x_val=pd.Series([f"v{i}" for i in index], index=index),
        y_val=pd.Series([0] * len(index), index=index),
        label_map={"a": 0},
    )

    first = cost.sample_texts(data, settings)

    pd.testing.assert_series_equal(first, cost.sample_texts(data, settings))
    assert len(first) == 4
    assert first.index.is_monotonic_increasing
    assert all(first[i] == f"v{i}" for i in first.index)


def test_summaries_use_percentiles_and_medians():
    warm = cost.summarize_warm([int(ms * 1e6) for ms in range(1, 101)])
    cold = cost.summarize_cold([
        {"import_s": 1.0, "load_s": 2.0, "first_predict_s": 0.1},
        {"import_s": 3.0, "load_s": 1.0, "first_predict_s": 0.2},
        {"import_s": 2.0, "load_s": 2.0, "first_predict_s": 0.3},
    ])

    assert warm["warm_p50_ms"] == pytest.approx(np.percentile(np.arange(1, 101), 50))
    assert warm["warm_p95_ms"] == pytest.approx(np.percentile(np.arange(1, 101), 95))
    assert warm["warm_max_ms"] == pytest.approx(100)
    assert cold["cold_median_s"] == pytest.approx(4.2)  # sumy: 3,1 / 4,2 / 4,3
    assert cold["cold_max_s"] == pytest.approx(4.3)
    assert cold["cold_import_median_s"] == pytest.approx(2.0)
    assert cold["cold_load_median_s"] == pytest.approx(2.0)
    assert cold["cold_first_predict_median_s"] == pytest.approx(0.2)


def test_worker_env_limits_threads_and_hides_gpu():
    env = cost.worker_env(3)

    assert all(env[name] == "3" for name in cost.THREAD_ENV)
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_worker_env_really_hides_cuda_from_torch():
    """Regresja 14.10: na Windows `CUDA_VISIBLE_DEVICES=""` nie ukrywało GPU przed torch."""
    pytest.importorskip("torch")
    code = "import torch; print(torch.cuda.is_available())"

    completed = subprocess.run(
        [sys.executable, "-c", code], env=cost.worker_env(1), capture_output=True, text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"


def test_directory_bytes_counts_every_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)

    assert cost.directory_bytes(tmp_path) == 15


def test_measure_runs_real_worker_processes_on_cpu(tmp_path):
    settings = _with_cost(load_settings(), warmup=2, passes=2, cold_runs=2)
    texts = [
        "where is my card", "my card has not arrived", "card still not here",
        "top up failed", "top up declined", "my top up did not work",
    ]
    TfidfLrClassifier(settings).fit(texts, [0, 0, 0, 1, 1, 1]).save(tmp_path / "tfidf_lr")
    queries = ["card not here yet", "top up not working", "card late"]

    result = cost.measure("tfidf_lr", 1, queries, settings, directory=tmp_path / "tfidf_lr")

    assert len(result.raw["warm_ns"]) == 2 * len(queries)
    assert len(result.raw["cold"]) == 2
    assert result.torch_threads is None  # ścieżka TF-IDF nie ładuje torch
    m = result.metrics
    assert 0 < m["warm_p50_ms"] <= m["warm_p95_ms"] <= m["warm_max_ms"]
    assert 0 < m["cold_median_s"] <= m["cold_max_s"]


def test_log_cost_run_records_params_metrics_and_raw_timings(tmp_mlflow_settings):
    result = cost.CostResult(
        model="tfidf_lr", threads=1, torch_threads=None,
        metrics={"warm_p50_ms": 0.5, "warm_p95_ms": 0.9, "cold_median_s": 1.2},
        raw={"warm_ns": [1, 2], "cold": []},
    )

    run_id = cost.log_cost_run(
        result, tmp_mlflow_settings, sample_ids_sha256="abc", artifact_size=5_000_000
    )

    client = MlflowClient(tmp_mlflow_settings.mlflow.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags[cost.STUDY_TAG] == cost.STUDY_NAME
    assert run.data.tags["model_family"] == "tfidf_lr"
    assert run.data.params["threads"] == "1"
    assert run.data.params["device"] == "cpu"
    assert run.data.params["sample_ids_sha256"] == "abc"
    assert run.data.metrics["artifact_mb"] == pytest.approx(5.0)
    assert run.data.metrics["warm_p95_ms"] == pytest.approx(0.9)
    assert cost.RAW_ARTIFACT in [artifact.path for artifact in client.list_artifacts(run_id)]


def test_cli_refuses_second_measurement_before_loading_data(
    tmp_mlflow_settings, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location("measure_cost_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_settings", lambda: tmp_mlflow_settings)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda *args: 6)
    monkeypatch.setattr(
        cli, "load_train_val", lambda *args: pytest.fail("dane wczytane mimo pomiaru")
    )
    monkeypatch.setattr("sys.argv", ["measure_cost.py"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "pomiar" in capsys.readouterr().out


def test_cli_final_uses_final_artifacts_and_separate_study(
    tmp_path, tmp_mlflow_settings, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location("measure_cost_cli_final", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    settings = replace(
        tmp_mlflow_settings,
        paths=replace(tmp_mlflow_settings.paths, final_models_dir=tmp_path / "final"),
    )
    studies = []
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "count_runs_with_tag", lambda s, key, value: studies.append(value))
    monkeypatch.setattr(cli, "load_train_val", lambda *args: pytest.fail("dane wczytane"))
    monkeypatch.setattr("sys.argv", ["measure_cost.py", "--final"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert studies == [cost.STUDY_NAME_FINAL]
    out = capsys.readouterr().out
    assert str(tmp_path / "final") in out
    assert "refit_final.py" in out


def test_artifact_bytes_reads_given_root(tmp_path):
    (tmp_path / "tfidf_lr").mkdir()
    (tmp_path / "tfidf_lr" / "model.joblib").write_bytes(b"x" * 7)

    assert cost.artifact_bytes("tfidf_lr", load_settings(), tmp_path) == 7
