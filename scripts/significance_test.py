"""Test istotności różnic na zapisanych predykcjach testu (dodatek W5) — cienkie CLI.

Uruchomienie (raz, po zielonych testach; parametry z prerejestracji w configu):
uv run python scripts/significance_test.py
Wypisuje tylko liczby (accuracy, b/c, p, p po Holmie, przedziały) — bez tekstów testu.
Zapis do MLflow (decyzja autora 29.10): osobny run z tagiem `analysis=significance` — nie
`split=test`, bo to analiza zapisanych predykcji, nie ocena końcowa; parametry z configu,
metryki per para, tabela jako artefakt CSV. Analiza raz: istniejący run z tym tagiem = odmowa.
Logika: src/banking77_intent_ml/evaluation/significance.py. Kod CLI: asystent.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.evaluation.error_analysis import test_artifacts
from banking77_intent_ml.evaluation.significance import significance_table
from banking77_intent_ml.evaluation.tracking import configure_mlflow, count_runs_with_tag

TAG = ("analysis", "significance")
TABLE_ARTIFACT = "significance.csv"


def main() -> None:
    settings = load_settings()
    configure_mlflow(settings)
    if count_runs_with_tag(settings, *TAG):
        raise SystemExit(f"[STOP] run z tagiem {TAG[0]}={TAG[1]} już istnieje — analiza raz")

    artifacts = test_artifacts(settings)
    predictions = {name: art.predictions for name, art in artifacts.items()}
    table = significance_table(predictions, settings)
    n = len(next(iter(predictions.values())))
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(f"N = {n}")
        print(table.to_string(index=False))

    cfg = settings.evaluation.significance
    metrics = {}
    for row in table.itertuples(index=False):
        key = f"{row.a}_vs_{row.b}"
        metrics.update({
            f"{key}_diff_pp": row.diff_pp, f"{key}_p": row.p, f"{key}_p_holm": row.p_holm,
            f"{key}_reject": float(row.reject), f"{key}_b_only": row.b_only,
            f"{key}_c_only": row.c_only, f"{key}_ci_low_pp": row.ci_low_pp,
            f"{key}_ci_high_pp": row.ci_high_pp,
        })
    with mlflow.start_run(run_name="significance") as run:
        mlflow.set_tags({
            TAG[0]: TAG[1],
            "source_runs": ",".join(f"{name}:{art.run_id}" for name, art in artifacts.items()),
            "data_commit": settings.dataset.commit,
        })
        mlflow.log_params({
            "pairs": ";".join(f"{a}-{b}" for a, b in cfg.pairs),
            "alpha": cfg.alpha,
            "n_bootstrap": cfg.n_bootstrap,
            "ci_level": cfg.ci_level,
            "seed": settings.models.seed,
            "n": n,
        })
        mlflow.log_metrics(metrics)
        with TemporaryDirectory(prefix="significance-") as temporary:
            path = Path(temporary) / TABLE_ARTIFACT
            table.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
    print(f"MLflow: run {run.info.run_id} ({TAG[0]}={TAG[1]})")


if __name__ == "__main__":
    main()
