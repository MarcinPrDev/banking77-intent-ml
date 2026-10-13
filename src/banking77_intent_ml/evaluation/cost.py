"""Pomiary kosztu CPU (W2 N2): latencja batch=1 przez wspólny interfejs, zimny start, rozmiar.

Decyzje autora (14.10.2026, `evaluation.cost` w configu): wątki 1 i 8; latencja rozgrzana =
`warmup` predykcji rozgrzewki, potem `n_texts` zdań z walidacji × `passes` przebiegów, batch=1,
p50/p95 z `numpy.percentile`; zimny start = `cold_runs` nowych procesów na ścieżkę (mediana
i max); poziom = predyktor (`predict_proba` na jednym tekście, jak w API), bez HTTP.

Każdy pomiar w osobnym procesie (`evaluation.cost_worker`) ze środowiskiem ustawionym przed
startem Pythona: OMP/MKL/OpenBLAS = liczba wątków i `CUDA_VISIBLE_DEVICES=""` — tylko CPU, także
dla enkodera MiniLM. Zdania: `n_texts` losowanych z walidacji seedem `models.seed`, te same dla
każdej ścieżki i liczby wątków (hash ich id w runie). Rozmiar = cały katalog
`paths.models_dir/<nazwa>`; dla MiniLM+kNN plus wagi enkodera z cache Hugging Face (przypięta
rewizja), bez których indeks nie działa. MB = 10^6 B.
Ograniczenie: po pierwszym uruchomieniu pliki modelu są w cache systemu plików, więc zimny start
mierzy nowy proces, a nie pierwszy odczyt z dysku.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.tracking import configure_mlflow, protocol_tags

STUDY_TAG = "cost_study"
STUDY_NAME = "cpu_cost_w2"  # wartość tagu — jeden pomiar oficjalny wszystkich ścieżek
WORKER_MODULE = "banking77_intent_ml.evaluation.cost_worker"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
RAW_ARTIFACT = "cost_raw.json"


@dataclass(frozen=True)
class CostResult:
    model: str
    threads: int
    metrics: dict[str, float]  # warm_* [ms], cold_* [s]
    torch_threads: int | None  # faktyczna liczba wątków torch; None, gdy ścieżka nie używa torch
    raw: dict[str, Any]  # surowe pomiary: czasy rozgrzane [ns], składowe zimnych startów [s]


def sample_texts(data: TrainVal, settings: Settings) -> pd.Series:
    """`n_texts` zdań z walidacji (seed `models.seed`), w kolejności pozycji w walidacji."""
    rng = np.random.default_rng(settings.models.seed)
    n_texts = settings.evaluation.cost.n_texts
    positions = np.sort(rng.choice(len(data.x_val), size=n_texts, replace=False))
    return data.x_val.iloc[positions]


def worker_env(threads: int) -> dict[str, str]:
    env = {**os.environ, **{name: str(threads) for name in THREAD_ENV}}
    env["CUDA_VISIBLE_DEVICES"] = ""  # tylko CPU — także enkoder MiniLM
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_worker(job: dict[str, Any], threads: int) -> dict[str, Any]:
    """Nowy proces Pythona z `cost_worker`; wynik = ostatnia linia JSON ze stdout."""
    completed = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, json.dumps(job)],
        env=worker_env(threads), capture_output=True, encoding="utf-8", check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"proces pomiarowy zakończony kodem {completed.returncode}:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def summarize_warm(timings_ns: list[int]) -> dict[str, float]:
    ms = np.asarray(timings_ns, dtype=float) / 1e6
    return {
        "warm_p50_ms": float(np.percentile(ms, 50)),
        "warm_p95_ms": float(np.percentile(ms, 95)),
        "warm_mean_ms": float(ms.mean()),
        "warm_max_ms": float(ms.max()),
    }


def summarize_cold(runs: list[dict[str, float]]) -> dict[str, float]:
    totals = [run["import_s"] + run["load_s"] + run["first_predict_s"] for run in runs]
    return {
        "cold_median_s": median(totals),
        "cold_max_s": max(totals),
        "cold_import_median_s": median(run["import_s"] for run in runs),
        "cold_load_median_s": median(run["load_s"] for run in runs),
        "cold_first_predict_median_s": median(run["first_predict_s"] for run in runs),
    }


def measure(
    name: str, threads: int, texts: list[str], settings: Settings, *, directory: Path | None = None,
) -> CostResult:
    """Latencja rozgrzana (jeden proces) i `cold_runs` zimnych startów jednej ścieżki."""
    cfg = settings.evaluation.cost
    with TemporaryDirectory(prefix="cost-") as temporary:
        texts_path = Path(temporary) / "texts.json"
        texts_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        job = {
            "model": name, "threads": threads, "texts": str(texts_path),
            "directory": str(directory) if directory else None,
        }
        warm_job = {**job, "mode": "warm", "warmup": cfg.warmup, "passes": cfg.passes}
        warm = run_worker(warm_job, threads)
        cold = [run_worker({**job, "mode": "cold"}, threads) for _ in range(cfg.cold_runs)]
    if any(run["cuda"] for run in [warm, *cold]):
        raise RuntimeError("proces pomiarowy widział CUDA — pomiar ma być wyłącznie na CPU")
    cold_parts = [
        {key: run[key] for key in ("import_s", "load_s", "first_predict_s")} for run in cold
    ]
    return CostResult(
        model=name,
        threads=threads,
        metrics={**summarize_warm(warm["timings_ns"]), **summarize_cold(cold_parts)},
        torch_threads=warm["torch_threads"],
        raw={"warm_ns": warm["timings_ns"], "cold": cold_parts},
    )


def directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def encoder_bytes(settings: Settings) -> int:
    """Wagi enkodera MiniLM z przypiętej rewizji w cache Hugging Face (część artefaktu kNN)."""
    from huggingface_hub import scan_cache_dir

    cfg = settings.models.minilm_knn
    for repo in scan_cache_dir().repos:
        if repo.repo_id == cfg.model_id:
            for revision in repo.revisions:
                if revision.commit_hash == cfg.revision:
                    return revision.size_on_disk
    raise FileNotFoundError(f"brak {cfg.model_id}@{cfg.revision} w cache Hugging Face")


def artifact_bytes(name: str, settings: Settings) -> int:
    size = directory_bytes(settings.paths.models_dir / name)
    if name == "minilm_knn":
        size += encoder_bytes(settings)
    return size


def log_cost_run(
    result: CostResult, settings: Settings, *, sample_ids_sha256: str, artifact_size: int,
) -> str:
    """Run MLflow na ścieżkę × wątki: parametry pomiaru, metryki, surowe czasy jako artefakt."""
    cfg = settings.evaluation.cost
    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"cost-{result.model}-t{result.threads}") as run:
        mlflow.set_tags({
            **protocol_tags(settings), "model_family": result.model, STUDY_TAG: STUDY_NAME,
        })
        mlflow.log_params({
            "threads": result.threads,
            "torch_threads": result.torch_threads,
            "n_texts": cfg.n_texts,
            "passes": cfg.passes,
            "warmup": cfg.warmup,
            "cold_runs": cfg.cold_runs,
            "sample_ids_sha256": sample_ids_sha256,
            "device": "cpu",
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        })
        mlflow.log_metrics({**result.metrics, "artifact_mb": artifact_size / 1e6})
        mlflow.log_dict(result.raw, RAW_ARTIFACT)
        return run.info.run_id
