"""Opis kalibracji na zapisanych predykcjach testu (dodatek W5, 31.10).

Prerejestracja przed obliczeniami (config `evaluation.calibration`, decyzje autora 31.10.2026):
modele z `models` na teście 3 080; pewność = zapisany `score` wybranej klasy (top-label),
trafienie = `y_pred == y_true`; `n_bins` przedziałów równej szerokości na [0, 1], domkniętych
z prawej (a, b], pierwszy [0, 1/n_bins]; ECE = suma po niepustych przedziałach
n_b / N · |acc_b − pewność_b|. Opis bez dopasowania kalibracji, strojenia i wyboru modelu;
wynik nie zmienia zamrożonych liczb ani decyzji z ADR-0003.

Wczytywanie predykcji z MLflow i wykres: asystent. Obliczenia (`reliability_table`,
`expected_calibration_error`, `calibration_summary`): autor.
Obliczenia korzystają wyłącznie z id, etykiet i score zapisanych predykcji — bez tekstów.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.evaluation.basal_eval import FAMILY as BASAL_FAMILY
from banking77_intent_ml.evaluation.basal_eval import TEST_PREDICTIONS_ARTIFACT as BASAL_PREDICTIONS
from banking77_intent_ml.evaluation.error_analysis import test_artifacts
from banking77_intent_ml.evaluation.final import PREDICTION_COLUMNS
from banking77_intent_ml.evaluation.significance import correct_by_model
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.factory import STRATEGIES

BASAL_COLUMNS = {"true_id": "y_true", "predicted_id": "y_pred"}  # predictions_basal_test.csv
ANALYSIS_TAG = ("analysis", "calibration")  # run analizy (nie `split=test`)
SUMMARY_ARTIFACT = "calibration_summary.csv"
BINS_ARTIFACT = "calibration_bins.csv"
DISPLAY_NAMES = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF-IDF + LR",
    "minilm_knn": "MiniLM + kNN",
    "basal_4_5b": "Basal 4,5B",
    "basal_omni_ultra": "Basal ultra",
}


@dataclass(frozen=True)
class CalibrationInput:
    run_id: str  # run `split=test`, z którego pochodzą predykcje
    predictions: pd.DataFrame  # row_id, y_true, y_pred, score — bez tekstów


def calibration_inputs(settings: Settings) -> dict[str, CalibrationInput]:
    """Zapisane predykcje testu dla `evaluation.calibration.models`, w kolejności z configu.

    Ścieżki z `STRATEGIES` — przez `error_analysis.test_artifacts` (jeden run `split=test` na
    ścieżkę); kandydaci Basala — run `split=test`, `model_family=basal` z tagiem `candidate`
    (dokładnie jeden), kolumny `true_id`/`predicted_id` przemianowane na `y_true`/`y_pred`.
    Nieznana nazwa albo liczba runów kandydata różna od 1 → `ValueError` przed pobraniem
    artefaktów.
    """
    names = settings.evaluation.calibration.models
    known = {name for cfg in (settings.basal, settings.basal_ultra) if cfg for name in cfg.models}
    unknown = sorted(set(names) - set(STRATEGIES) - known)
    if unknown:
        raise ValueError(f"Nieznane modele w evaluation.calibration.models: {', '.join(unknown)}")
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=f"tags.split = 'test' and tags.model_family = '{BASAL_FAMILY}'",
    )
    candidates = runs.get("tags.candidate", pd.Series(index=runs.index, dtype="string"))
    for name in names:
        count = int((candidates == name).sum())
        if name not in STRATEGIES and count != 1:
            raise ValueError(f"{name}: {count} runów split=test — oczekiwano jednego")
    artifacts = test_artifacts(settings) if set(names) & set(STRATEGIES) else {}
    inputs = {}
    for name in names:
        if name in STRATEGIES:
            run_id, frame = artifacts[name].run_id, artifacts[name].predictions
        else:
            run_id = str(runs.loc[candidates == name, "run_id"].iloc[0])
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=BASAL_PREDICTIONS
            )
            frame = pd.read_csv(path).rename(columns=BASAL_COLUMNS)
        inputs[name] = CalibrationInput(run_id, frame[list(PREDICTION_COLUMNS)].copy())
    return inputs


def reliability_table(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Wiersz na przedział (wszystkie `n_bins`, także puste), w kolejności rosnącej.

    Przedziały równej szerokości na [0, 1], domknięte z prawej: (a, b]; pierwszy [0, 1/n_bins].
    Kolumny: `lower`, `upper`, `n`, `confidence` (średnia pewność), `accuracy` (odsetek trafień),
    `gap` (accuracy − confidence); w pustym przedziale `n` = 0, a średnie i `gap` = NaN.
    `ValueError`: dane inne niż wektory tej samej niezerowej długości, wartości nieskończone
    lub NaN, pewność spoza [0, 1], trafienia inne niż 0/1,
    `n_bins` inne niż dodatnia liczba całkowita.
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins musi być dodatnią liczbą całkowitą")
    if n_bins < 1:
        raise ValueError("n_bins musi być dodatnią liczbą całkowitą")
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.ndim != 1 or confidence.shape != correct.shape or confidence.size == 0:
        raise ValueError("Wymagane niepuste wektory tej samej długości")
    if not np.isfinite(confidence).all() or not np.isfinite(correct).all():
        raise ValueError("Pewność i trafienia muszą być skończone, bez NaN")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Pewność musi należeć do [0, 1]")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("Trafienia muszą mieć wartości 0 albo 1")
    edges = np.linspace(0, 1, n_bins + 1)
    # Szukanie w górnych granicach zachowuje (a, b] oraz włącza 0 do pierwszego przedziału.
    indices = np.searchsorted(edges[1:], confidence, side="left")
    rows = []
    for i in range(n_bins):
        selected = indices == i
        n = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if n else np.nan
        accuracy = float(correct[selected].mean()) if n else np.nan
        rows.append({
            "lower": edges[i], "upper": edges[i + 1], "n": n,
            "confidence": mean_confidence, "accuracy": accuracy,
            "gap": accuracy - mean_confidence,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE z tabeli `reliability_table`: suma po niepustych przedziałach n_b / N · |gap_b|.

    N = suma `n`; puste przedziały (`n` = 0) nie wnoszą nic. `ValueError`, gdy N = 0.
    """
    n = table["n"].sum()
    if n == 0:
        raise ValueError("ECE wymaga co najmniej jednej predykcji (N > 0)")
    nonempty = table.loc[table["n"] > 0]
    return float(np.sum(nonempty["n"].to_numpy() * np.abs(nonempty["gap"].to_numpy())) / n)


def calibration_summary(
    inputs: dict[str, CalibrationInput], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(podsumowanie, przedziały) dla modeli w kolejności `inputs`; `n_bins` z configu.

    Podsumowanie — wiersz na model: `model`, `n`, `accuracy`, `mean_confidence`, `ece`.
    Przedziały — `reliability_table` każdego modelu z kolumną `model` na początku.
    Każdy model musi mieć `settings.dataset.test.n_rows` wierszy, a wszystkie modele — te same
    `row_id` i `y_true` (jak w `significance.correct_by_model`); inaczej `ValueError`.
    """
    correct = correct_by_model({name: item.predictions for name, item in inputs.items()})
    n = len(correct)
    expected_n = settings.dataset.test.n_rows
    if n != expected_n:
        raise ValueError(f"Predykcje: {n} wierszy, pełny test według configu: {expected_n}")
    rows = []
    tables = []
    for name, item in inputs.items():
        confidence = item.predictions.set_index("row_id").loc[correct.index, "score"].to_numpy()
        table = reliability_table(
            confidence, correct[name].to_numpy(), settings.evaluation.calibration.n_bins
        )
        rows.append({
            "model": name, "n": n, "accuracy": float(correct[name].mean()),
            "mean_confidence": float(confidence.mean()),
            "ece": expected_calibration_error(table),
        })
        table.insert(0, "model", name)
        tables.append(table)
    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True)


def run_tables(settings: Settings, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(podsumowanie, przedziały) zapisane w runie `analysis=calibration` — do ponownego wykresu.

    Tylko odczyt artefaktów CSV, bez obliczeń na predykcjach; run bez tego tagu → `ValueError`.
    """
    configure_mlflow(settings)
    if mlflow.get_run(run_id).data.tags.get(ANALYSIS_TAG[0]) != ANALYSIS_TAG[1]:
        raise ValueError(f"Run {run_id} nie ma tagu {ANALYSIS_TAG[0]}={ANALYSIS_TAG[1]}")
    summary, bins = (
        pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name))
        for name in (SUMMARY_ARTIFACT, BINS_ARTIFACT)
    )
    return summary, bins


def plot_reliability(summary: pd.DataFrame, bins: pd.DataFrame, path: Path) -> Path:
    """Wykres niezawodności: kolumna na model, w kolejności `summary`; zapis PNG do `path`.

    Góra: accuracy w niepustych przedziałach (słupki na szerokość przedziału), średnia pewność
    w przedziale (czerwona kreska — nad słupkiem: pewność zawyżona, poniżej jego szczytu:
    zaniżona) i przekątna idealnej kalibracji; w tytule ECE i N. Dół: liczba predykcji
    w przedziałach (skala log). Tylko liczby z tabel — bez tekstów testu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary["model"].tolist()
    fig, axes = plt.subplots(
        2, len(models), figsize=(4 * len(models), 6.5), squeeze=False,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    for column, model in enumerate(models):
        row = summary.loc[summary["model"] == model].iloc[0]
        table = bins.loc[(bins["model"] == model) & (bins["n"] > 0)]
        width = table["upper"] - table["lower"]
        top, bottom = axes[0, column], axes[1, column]
        top.bar(table["lower"], table["accuracy"], width=width, align="edge",
                color="tab:blue", alpha=0.75, edgecolor="white", label="accuracy w przedziale")
        top.hlines(table["confidence"], table["lower"], table["upper"], colors="tab:red",
                   linewidth=2.5, label="średnia pewność w przedziale")
        top.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                 label="idealna kalibracja")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.set_title(f"{DISPLAY_NAMES.get(model, model)}\nECE {row['ece']:.4f}, N {int(row['n'])}")
        bottom.bar(table["lower"], table["n"], width=width, align="edge", color="tab:gray")
        bottom.set_yscale("log")
        bottom.set_xlabel("pewność (score wybranej klasy)")
        if column == 0:
            top.set_ylabel("accuracy w przedziale")
            top.legend(loc="upper left", fontsize=8)
            bottom.set_ylabel("liczba predykcji")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
