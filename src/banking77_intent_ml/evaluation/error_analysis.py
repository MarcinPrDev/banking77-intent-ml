"""Dane do analizy błędów (W3 §1.3) — tylko wczytywanie; analizę i wnioski pisze autor.

Decyzje autora (14.10.2026): analiza w `notebooks/02_analiza_bledow.ipynb`, wnioski w README
(sekcja „Analiza błędów”). Ten moduł dostarcza:
- z runów `split=test` (jeden na ścieżkę): macierz pomyłek 77×77, tabelę per klasa i predykcje
  per przykład (`row_id`, `y_true`, `y_pred`, `score`) — agregaty i id, bez tekstów: zdań
  testowych nie czytamy także po ocenie (W3.md §5);
- z modeli W2 (`paths.models_dir`, trening na 8 504): predykcje na walidacji Z TEKSTAMI — tu
  przeglądamy przykłady błędów (finalne modele widziały walidację w treningu);
- pomoc techniczną: nazwy klas obok id i wykres macierzy 77×77.
Etykiet nie zmieniamy.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.artifacts
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal, load_train_val
from banking77_intent_ml.data.labels import LabelMap
from banking77_intent_ml.evaluation.final import (
    CONFUSION_ARTIFACT,
    PER_CLASS_ARTIFACT,
    PREDICTIONS_ARTIFACT,
)
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.factory import STRATEGIES, load_predictor


@dataclass(frozen=True)
class TestArtifacts:
    run_id: str
    confusion: pd.DataFrame  # wiersze: prawdziwa klasa, kolumny: przewidziana (nazwy, kolejność id)
    per_class: pd.DataFrame  # indeks: nazwa klasy; id, precision, recall, f1, support
    predictions: pd.DataFrame  # row_id, y_true, y_pred, score — bez tekstów


def _table(name: str, run_id: str) -> pd.DataFrame:
    """Tabela z `log_table`; pierwsza kolumna = indeks zapisany przez `reset_index()`."""
    table = mlflow.load_table(name, run_ids=[run_id])
    return table.set_index(table.columns[0])


def test_artifacts(settings: Settings) -> dict[str, TestArtifacts]:
    """Artefakty dla `STRATEGIES`; dokładnie jeden run `split=test` na każdą ścieżkę.

    Pozostałe rodziny (np. LLM na podzbiorze) są pomijane. Braki i duplikaty runów
    powodują `ValueError` przed pobraniem jakiegokolwiek artefaktu.
    """
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment], filter_string="tags.split = 'test'"
    )
    families = runs.get("tags.model_family", pd.Series(index=runs.index, dtype="string"))
    runs = runs.loc[families.isin(STRATEGIES)]
    counts = families.value_counts().reindex(sorted(STRATEGIES), fill_value=0)
    for name, count in counts.items():
        if count != 1:
            raise ValueError(f"{name}: {count} runów split=test — oczekiwano jednego")
    artifacts = {}
    for name, group in runs.groupby("tags.model_family"):
        run_id = group["run_id"].iloc[0]
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=PREDICTIONS_ARTIFACT
        )
        artifacts[name] = TestArtifacts(
            run_id=run_id,
            confusion=_table(CONFUSION_ARTIFACT, run_id),
            per_class=_table(PER_CLASS_ARTIFACT, run_id),
            predictions=pd.read_csv(path),
        )
    return artifacts


def val_predictions_w2(
    settings: Settings, data: TrainVal | None = None
) -> dict[str, pd.DataFrame]:
    """Predykcje modeli W2 na walidacji: `row_id`, `text`, `y_true`, `y_pred`, `score`."""
    data = data if data is not None else load_train_val(settings)
    texts = data.x_val.tolist()
    predictions = {}
    for name in sorted(STRATEGIES):
        scores = load_predictor(name, settings).predict_proba(texts)
        predictions[name] = pd.DataFrame({
            "row_id": data.x_val.index,
            "text": texts,
            "y_true": data.y_val.to_numpy(),
            "y_pred": scores.argmax(axis=1),
            "score": scores.max(axis=1),
        })
    return predictions


def add_label_names(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Kopia ramki z kolumnami `true_name` i `pred_name` (mapa etykiet projektu)."""
    labels = LabelMap.from_settings(settings)
    return frame.assign(
        true_name=frame["y_true"].map(labels.name), pred_name=frame["y_pred"].map(labels.name)
    )


def plot_confusion(confusion: pd.DataFrame, *, normalize: bool = True, title: str = "", ax=None):
    """Macierz 77×77 jako mapa ciepła; `normalize` = udział w wierszu (recall per klasa)."""
    import matplotlib.pyplot as plt

    values = confusion.to_numpy(dtype=float)
    if normalize:
        values = values / values.sum(axis=1, keepdims=True).clip(min=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(values, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(confusion.index)), confusion.index, fontsize=5)
    ax.set_xlabel("przewidziana")
    ax.set_ylabel("prawdziwa")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, fraction=0.03)
    return ax
