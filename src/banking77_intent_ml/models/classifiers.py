"""Adaptery trzech ścieżek W2 do wspólnego interfejsu `IntentClassifier` (Strategy).

Zasady dla implementacji (pilnuje ich `tests/test_predictor.py`):
- hiperparametry wyłącznie z zamrożonej konfiguracji `models.<name>.frozen` (i bazy z configu);
- torch, transformers i sentence-transformers importuj wewnątrz metod — `models.factory`
  importuje ten moduł, a CI ma tylko extra `dev`;
- ciężkie modele ładuj przez funkcje infrastruktury wołane przez moduł (`embeddings.encode_texts`,
  nie `from ... import encode_texts`) — test kontraktu podmienia je atrapami, więc działa bez
  pobierania i bez GPU. MiniLM: `features.embeddings.encode_texts`; DistilBERT:
  `models.distilbert._load_tokenizer`, `_load_model`, `training_arguments`;
- artefakt = katalog (`paths.models_dir/<name>` przy eksporcie); istniejący -> `FileExistsError`.

TODO(Marcin, W2 pkt 5): logika trzech adapterów — kontrakt w `models/predictor.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from banking77_intent_ml.config import Settings
from banking77_intent_ml.models.predictor import IntentClassifier


class TfidfLrClassifier(IntentClassifier):
    """TF-IDF + LR, zamrożona próba HPO (`models.tfidf_lr.frozen`).

    TODO(Marcin): fit — pipeline z `build_tfidf_lr` na konfiguracji `frozen`; save/load —
    `persistence` (joblib + metadane z kontrolą mapy etykiet i wersji sklearn) w katalogu.
    """

    name = "tfidf_lr"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> TfidfLrClassifier:
        raise NotImplementedError("W2 pkt 5: TfidfLrClassifier.fit — pisze autor")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: TfidfLrClassifier.predict — pisze autor")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: TfidfLrClassifier.predict_proba — pisze autor")

    def save(self, directory: Path) -> Path:
        raise NotImplementedError("W2 pkt 5: TfidfLrClassifier.save — pisze autor")

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> TfidfLrClassifier:
        raise NotImplementedError("W2 pkt 5: TfidfLrClassifier.load — pisze autor")


class MiniLmKnnClassifier(IntentClassifier):
    """Zamrożone embeddingi MiniLM + kNN (`models.minilm_knn.frozen`).

    TODO(Marcin): fit — `embeddings.encode_texts` + `build_knn` z `frozen`; predict — embedding
    zapytania w środku (latencja mierzona uczciwie, z ekstrakcją); save/load — indeks kNN to
    estymator sklearn, więc `persistence`; model i rewizja MiniLM w metadanych.
    """

    name = "minilm_knn"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> MiniLmKnnClassifier:
        raise NotImplementedError("W2 pkt 5: MiniLmKnnClassifier.fit — pisze autor")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: MiniLmKnnClassifier.predict — pisze autor")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: MiniLmKnnClassifier.predict_proba — pisze autor")

    def save(self, directory: Path) -> Path:
        raise NotImplementedError("W2 pkt 5: MiniLmKnnClassifier.save — pisze autor")

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> MiniLmKnnClassifier:
        raise NotImplementedError("W2 pkt 5: MiniLmKnnClassifier.load — pisze autor")


class DistilBertClassifier(IntentClassifier):
    """DistilBERT, zamrożony wariant i epoka (`models.distilbert.frozen`).

    TODO(Marcin): fit — Trainer z LR z `frozen`, harmonogram na `epochs` z configu i zatrzymanie
    po `frozen.epoch` (tak powstał zamrożony checkpoint); predict/predict_proba — logity
    i softmax w trybie `eval()`; save/load — `save_pretrained` / `from_pretrained` +
    `metadata.json`, format jak `models/distilbert/` z zamrożenia 13.10 (ten artefakt się wczyta).
    """

    name = "distilbert"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> DistilBertClassifier:
        raise NotImplementedError("W2 pkt 5: DistilBertClassifier.fit — pisze autor")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: DistilBertClassifier.predict — pisze autor")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("W2 pkt 5: DistilBertClassifier.predict_proba — pisze autor")

    def save(self, directory: Path) -> Path:
        raise NotImplementedError("W2 pkt 5: DistilBertClassifier.save — pisze autor")

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> DistilBertClassifier:
        raise NotImplementedError("W2 pkt 5: DistilBertClassifier.load — pisze autor")
