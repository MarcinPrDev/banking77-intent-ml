"""Opis kalibracji na zapisanych predykcjach testu (dodatek W5) — cienkie CLI.

Uruchomienie (raz, po zielonych testach; parametry z prerejestracji w configu):
uv run python scripts/calibration_report.py
Wypisuje tylko liczby (N, accuracy, średnia pewność, ECE; tabele przedziałów) — bez tekstów testu.
Zapis do MLflow: osobny run z tagiem `analysis=calibration` — nie `split=test`, bo to opis
zapisanych predykcji, nie ocena końcowa; parametry z configu, metryki per model, tabele CSV
i wykres jako artefakty; wykres także do `evaluation.calibration.figure_path` (README).
Analiza raz: istniejący run z tym tagiem = odmowa.

Sam wykres z tabel CSV zapisanego runu (bez obliczeń i bez nowego runu):
uv run python scripts/calibration_report.py --redraw <run_id>
Logika: src/banking77_intent_ml/evaluation/calibration.py. Kod CLI: asystent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.evaluation.calibration import (
    ANALYSIS_TAG,
    BINS_ARTIFACT,
    SUMMARY_ARTIFACT,
    calibration_inputs,
    calibration_summary,
    plot_reliability,
    run_tables,
)
from banking77_intent_ml.evaluation.tracking import configure_mlflow, count_runs_with_tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Opis kalibracji na zapisanych predykcjach testu")
    parser.add_argument(
        "--redraw", metavar="RUN_ID",
        help="tylko wykres z tabel CSV zapisanego runu analysis=calibration",
    )
    args = parser.parse_args()
    settings = load_settings()
    configure_mlflow(settings)
    cfg = settings.evaluation.calibration
    if args.redraw:
        summary, bins = run_tables(settings, args.redraw)
        print(f"Wykres z runu {args.redraw}: {plot_reliability(summary, bins, cfg.figure_path)}")
        return
    if count_runs_with_tag(settings, *ANALYSIS_TAG):
        raise SystemExit(
            f"[STOP] run z tagiem {ANALYSIS_TAG[0]}={ANALYSIS_TAG[1]} już istnieje — analiza raz"
        )

    inputs = calibration_inputs(settings)
    summary, bins = calibration_summary(inputs, settings)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary.to_string(index=False))
        print()
        print(bins.to_string(index=False))

    figure = plot_reliability(summary, bins, cfg.figure_path)
    metrics = {}
    for row in summary.itertuples(index=False):
        metrics.update({
            f"{row.model}_ece": row.ece,
            f"{row.model}_accuracy": row.accuracy,
            f"{row.model}_mean_confidence": row.mean_confidence,
        })
    with mlflow.start_run(run_name="calibration") as run:
        mlflow.set_tags({
            ANALYSIS_TAG[0]: ANALYSIS_TAG[1],
            "source_runs": ",".join(f"{name}:{item.run_id}" for name, item in inputs.items()),
            "data_commit": settings.dataset.commit,
        })
        mlflow.log_params({
            "models": ";".join(cfg.models),
            "n_bins": cfg.n_bins,
            "n": int(summary["n"].iloc[0]),
        })
        mlflow.log_metrics(metrics)
        with TemporaryDirectory(prefix="calibration-") as temporary:
            for name, table in ((SUMMARY_ARTIFACT, summary), (BINS_ARTIFACT, bins)):
                path = Path(temporary) / name
                table.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(figure))
    print(f"Wykres: {figure}")
    print(f"MLflow: run {run.info.run_id} ({ANALYSIS_TAG[0]}={ANALYSIS_TAG[1]})")


if __name__ == "__main__":
    main()
