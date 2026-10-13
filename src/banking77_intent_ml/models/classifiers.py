"""Adaptery trzech ścieżek W2 do wspólnego interfejsu `IntentClassifier` (Strategy).

Zasady dla implementacji (pilnuje ich `tests/test_predictor.py`):
- hiperparametry wyłącznie z zamrożonej konfiguracji `models.<name>.frozen` (i bazy z configu);
- torch, transformers i sentence-transformers importuj wewnątrz metod — `models.factory`
  importuje ten moduł, a CI ma tylko extra `dev`;
- builder z `baselines` importuj dopiero w `fit` — zależy od MLflow, zbędnego przy predykcji;
- ciężkie modele ładuj przez funkcje infrastruktury wołane przez moduł (`embeddings.encode_texts`,
  nie `from ... import encode_texts`) — test kontraktu podmienia je atrapami, więc działa bez
  pobierania i bez GPU. MiniLM: `features.embeddings.encode_texts`; DistilBERT:
  `models.distilbert._load_tokenizer`, `_load_model`, `training_arguments`;
- artefakt = katalog (`paths.models_dir/<name>` przy eksporcie); istniejący -> `FileExistsError`.

Kontrakt w `models/predictor.py`; adaptery MiniLM i DistilBERT pozostają do uzupełnienia.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.exceptions import NotFittedError

from banking77_intent_ml.config import Settings
from banking77_intent_ml.models.persistence import (
    MODEL_SUFFIX,
    build_metadata,
    load_model,
    save_model,
)
from banking77_intent_ml.models.predictor import IntentClassifier

_SKLEARN_ARTIFACT = f"model{MODEL_SUFFIX}"  # TF-IDF+LR: plik modelu w katalogu artefaktu


class TfidfLrClassifier(IntentClassifier):
    """TF-IDF + LR, zamrożona próba HPO (`models.tfidf_lr.frozen`).

    Pipeline z `build_tfidf_lr` na konfiguracji `frozen`; artefakt to `model.joblib`
    i `model.json` z kontrolą mapy etykiet i wersji sklearn. `run_id=None`, ponieważ
    adapter dopasowuje model do podanych danych bez tworzenia runu MLflow.
    """

    name = "tfidf_lr"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> TfidfLrClassifier:
        from banking77_intent_ml.models.baselines import build_tfidf_lr

        base = self.settings.models.tfidf_lr
        if base.frozen is None:
            raise ValueError("models.tfidf_lr.frozen jest puste — najpierw zamrożenie konfiguracji")
        config = replace(
            base, tfidf=dict(base.frozen["tfidf"]), logreg=dict(base.frozen["logreg"])
        )
        model = build_tfidf_lr(config, seed=self.settings.models.seed)
        model.fit(texts, labels)
        self._model = model
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise NotFittedError("TfidfLrClassifier: najpierw wywołaj fit() albo load().")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._model.predict(texts), dtype=int)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        probabilities = self._model.predict_proba(texts)
        scores = np.zeros((probabilities.shape[0], self.settings.dataset.n_classes), dtype=float)
        # sklearn zwraca kolumny dla classes_; brak klasy w treningu nie może przesunąć id.
        scores[:, self._model.classes_] = probabilities
        return scores

    def save(self, directory: Path) -> Path:
        self._require_fitted()
        model = self._model
        metadata = build_metadata(model_name=self.name, run_id=None, settings=self.settings)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            save_model(model, directory / _SKLEARN_ARTIFACT, metadata=metadata)
        except BaseException:
            # Usuwamy tylko własny pusty katalog; sprzątanie nie maskuje błędu zapisu.
            with suppress(OSError):
                directory.rmdir()
            raise
        return directory

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> TfidfLrClassifier:
        classifier = cls(settings)
        classifier._model = load_model(directory / _SKLEARN_ARTIFACT, settings=settings)
        return classifier


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
