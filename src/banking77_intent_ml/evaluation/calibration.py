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
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.factory import STRATEGIES

BASAL_COLUMNS = {"true_id": "y_true", "predicted_id": "y_pred"}  # predictions_basal_test.csv
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
    `ValueError`: różne długości, puste dane, NaN, pewność spoza [0, 1], `n_bins` < 1.
    """
    raise NotImplementedError("TODO-Marcin")


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE z tabeli `reliability_table`: suma po niepustych przedziałach n_b / N · |gap_b|.

    N = suma `n`; puste przedziały (`n` = 0) nie wnoszą nic. `ValueError`, gdy N = 0.
    """
    raise NotImplementedError("TODO-Marcin")


def calibration_summary(
    inputs: dict[str, CalibrationInput], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(podsumowanie, przedziały) dla modeli w kolejności `inputs`; `n_bins` z configu.

    Podsumowanie — wiersz na model: `model`, `n`, `accuracy`, `mean_confidence`, `ece`.
    Przedziały — `reliability_table` każdego modelu z kolumną `model` na początku.
    Każdy model musi mieć `settings.dataset.test.n_rows` wierszy, a wszystkie modele — te same
    `row_id` i `y_true` (jak w `significance.correct_by_model`); inaczej `ValueError`.
    """
    raise NotImplementedError("TODO-Marcin")


def plot_reliability(summary: pd.DataFrame, bins: pd.DataFrame, path: Path) -> Path:
    """Wykres niezawodności: kolumna na model, w kolejności `summary`; zapis PNG do `path`.

    Góra: accuracy w niepustych przedziałach (słupki na szerokość przedziału), luka do średniej
    pewności (kreskowanie) i przekątna idealnej kalibracji; w tytule ECE i N. Dół: liczba
    predykcji w przedziałach (skala log). Tylko liczby z tabel — bez tekstów testu.
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
                color="tab:blue", edgecolor="white", label="accuracy")
        top.bar(table["lower"], table["confidence"] - table["accuracy"], width=width,
                bottom=table["accuracy"], align="edge", color="tab:red", alpha=0.25,
                hatch="//", edgecolor="tab:red", label="luka do pewności")
        top.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
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
