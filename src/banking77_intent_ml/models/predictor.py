"""Wspólny interfejs predyktora intencji (wzorzec Strategy) — W2, pkt 5.

Decyzje autora (13.10.2026): pełny kontrakt jak w README → Architektura — `fit`, `predict`,
`predict_proba`, `save`, `load`; `predict` zwraca id klas z `configs/label_map.json` (int), a nazwy
intencji mapuje `data.labels.LabelMap` (API). Predyktor budowany z zamrożonej konfiguracji
`models.<nazwa>.frozen` przez `models.factory`. API i ewaluacja (W3) widzą tylko ten interfejs,
więc podmiana modelu to inna nazwa strategii w configu, nie zmiana kodu (OCP, DIP).

Kontrakt każdej implementacji (`tests/test_predictor.py`, ten sam test dla każdej strategii):
- `fit(texts, labels)` uczy na podanych danych (część treningowa) i zwraca `self`; przyjmuje
  listy i `pd.Series` z indeksem id wierszy (jak w `TrainVal`);
- `predict(texts)` -> `np.ndarray` (n,) int, id z mapy; równe `predict_proba(texts).argmax(1)`;
- `predict_proba(texts)` -> `np.ndarray` (n, n_classes) float; kolumna = id klasy z mapy
  `dataset.label_map_path`. To `score`, nie skalibrowane prawdopodobieństwo (kNN: głosy
  sąsiadów, DistilBERT: softmax);
- `save(directory)` zapisuje komplet artefaktu w nowym katalogu i go zwraca; istniejący
  katalog -> `FileExistsError` (nic nie nadpisuje);
- `load(directory, settings)` odtwarza predyktor dający identyczne predykcje.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from banking77_intent_ml.config import Settings


class IntentClassifier(ABC):
    """Strategia klasyfikacji intencji; konkretne ścieżki w `models.classifiers`."""

    name: ClassVar[str]  # klucz strategii = nazwa sekcji w `models` configu

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> IntentClassifier:
        """Trening zamrożonej konfiguracji na podanych danych; zwraca `self`."""

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Id klas (int) dla każdego tekstu."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score dla każdej klasy (n, n_classes); kolumna = id z mapy etykiet."""

    @abstractmethod
    def save(self, directory: Path) -> Path:
        """Komplet artefaktu w `directory` (nie nadpisuje istniejącego); zwraca ścieżkę."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path, settings: Settings) -> IntentClassifier:
        """Predyktor odtworzony z artefaktu zapisanego przez `save`."""
