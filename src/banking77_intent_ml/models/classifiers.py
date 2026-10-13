"""Adaptery trzech ścieżek W2 do wspólnego interfejsu `IntentClassifier` (Strategy).

Zasady dla implementacji (pilnuje ich `tests/test_predictor.py`):
- hiperparametry wyłącznie z zamrożonej konfiguracji `models.<name>.frozen` (i bazy z configu);
- torch, transformers i sentence-transformers importuj wewnątrz metod — `models.factory`
  importuje ten moduł, a CI ma tylko extra `dev`;
- buildery z `baselines` i `knn` importuj dopiero w `fit` — zależą od MLflow;
- ciężkie modele ładuj przez funkcje infrastruktury wołane przez moduł (`embeddings.encode_texts`,
  nie `from ... import encode_texts`) — test kontraktu podmienia je atrapami, więc działa bez
  pobierania i bez GPU. MiniLM: `features.embeddings.encode_texts`; DistilBERT:
  `models.distilbert._load_tokenizer`, `_load_model`, `training_arguments`;
- artefakt = katalog (`paths.models_dir/<name>` przy eksporcie); istniejący -> `FileExistsError`.

Kontrakt wszystkich adapterów w `models/predictor.py`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.exceptions import NotFittedError

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.labels import LabelMap
from banking77_intent_ml.models.persistence import (
    METADATA_SUFFIX,
    MODEL_SUFFIX,
    ModelArtifactError,
    build_metadata,
    label_map_sha256,
    load_model,
    save_model,
)
from banking77_intent_ml.models.predictor import IntentClassifier

_SKLEARN_ARTIFACT = f"model{MODEL_SUFFIX}"  # TF-IDF+LR i MiniLM+kNN: plik estymatora
_DISTILBERT_METADATA = "metadata.json"


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

    Każda predykcja obejmuje kodowanie tekstów przez `embeddings.encode_texts`.
    Artefakt to indeks kNN (`model.joblib`) i metadane (`model.json`, `run_id=None`).
    Wagi enkodera są ładowane przez `embeddings` z przypiętej rewizji; identyfikator
    i rewizja w metadanych muszą zgadzać się z configiem przy odczycie indeksu.
    `load` rozgrzewa enkoder przed zwróceniem predyktora, więc błąd ładowania wag
    przerywa start API, a pierwsze żądanie korzysta już z modelu w pamięci.
    """

    name = "minilm_knn"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> MiniLmKnnClassifier:
        cfg = self.settings.models.minilm_knn
        if cfg.frozen is None:
            raise ValueError(
                "models.minilm_knn.frozen jest puste — najpierw zamrożenie konfiguracji"
            )

        from banking77_intent_ml.features import embeddings
        from banking77_intent_ml.models.knn import build_knn

        model = build_knn(cfg.frozen["n_neighbors"], cfg.frozen["weights"], self.settings)
        model.fit(embeddings.encode_texts(texts, self.settings), labels)
        self._model = model
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise NotFittedError("MiniLmKnnClassifier: najpierw wywołaj fit() albo load().")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        from banking77_intent_ml.features import embeddings

        probabilities = self._model.predict_proba(embeddings.encode_texts(texts, self.settings))
        scores = np.zeros((probabilities.shape[0], self.settings.dataset.n_classes), dtype=float)
        scores[:, self._model.classes_] = probabilities
        return scores

    def save(self, directory: Path) -> Path:
        self._require_fitted()
        cfg = self.settings.models.minilm_knn
        metadata = build_metadata(model_name=self.name, run_id=None, settings=self.settings)
        metadata.update(model_id=cfg.model_id, revision=cfg.revision)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            save_model(self._model, directory / _SKLEARN_ARTIFACT, metadata=metadata)
        except BaseException:
            with suppress(OSError):
                directory.rmdir()
            raise
        return directory

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> MiniLmKnnClassifier:
        model_path = directory / _SKLEARN_ARTIFACT
        metadata_path = model_path.with_suffix(METADATA_SUFFIX)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"nie można odczytać metadanych: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise ModelArtifactError(f"plik metadanych musi zawierać obiekt JSON: {metadata_path}")
        cfg = settings.models.minilm_knn
        for field in ("model_id", "revision"):
            if metadata.get(field) != getattr(cfg, field):
                raise ModelArtifactError(f"niezgodne {field} enkodera MiniLM z artefaktem")
        classifier = cls(settings)
        classifier._model = load_model(model_path, settings=settings)
        from banking77_intent_ml.features import embeddings

        # Syntetyczny tekst: inicjalizacja enkodera należy do load/startu API.
        embeddings.encode_texts(["warmup"], settings)
        return classifier


class DistilBertClassifier(IntentClassifier):
    """DistilBERT, zamrożony wariant i epoka (`models.distilbert.frozen`).

    Scheduler obejmuje `epochs` z configu; callback zatrzymuje fit po `frozen.epoch`.
    Adapter uczy wyłącznie na podanych danych, bez ewaluacji i wyboru checkpointu.
    Artefakt HF zawiera wagi, tokenizer i `metadata.json`; odczyt lokalny na CPU sprawdza
    hash i kolejność klas. Starszy artefakt bez `max_len` używa limitu z configu.
    """

    name = "distilbert"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> DistilBertClassifier:
        cfg = self.settings.models.distilbert
        if cfg.frozen is None:
            raise ValueError(
                "models.distilbert.frozen jest puste — najpierw zamrożenie konfiguracji"
            )
        epoch = cfg.frozen["epoch"]
        if not isinstance(epoch, int) or not 1 <= epoch <= cfg.epochs:
            raise ValueError("models.distilbert.frozen.epoch musi być w zakresie 1..epochs")

        import pandas as pd
        import torch
        import transformers
        from transformers import DataCollatorWithPadding, Trainer, TrainerCallback, set_seed

        from banking77_intent_ml.models import distilbert

        class StopAtFrozenEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                if state.epoch >= epoch:
                    control.should_training_stop = True
                return control

        set_seed(self.settings.models.seed)
        label_map = dict(LabelMap.from_settings(self.settings).name_to_id)
        tokenizer = distilbert._load_tokenizer(self.settings)
        with TemporaryDirectory(prefix="distilbert-fit-") as temporary:
            args = replace(
                distilbert.training_arguments(cfg.frozen, self.settings, Path(temporary)),
                eval_strategy="no", save_strategy="no",
            )
            trainer = Trainer(
                model=distilbert._load_model(self.settings, label_map),
                args=args,
                data_collator=DataCollatorWithPadding(tokenizer),
                train_dataset=distilbert._features(
                    pd.Series(list(texts)), pd.Series(list(labels)), tokenizer, cfg.max_len,
                ),
                processing_class=tokenizer,
                callbacks=[StopAtFrozenEpoch()],
            )
            trainer.train()
        self._model = trainer.model.eval()
        self._tokenizer = tokenizer
        self._max_len = cfg.max_len
        self._metadata = build_metadata(model_name=self.name, run_id=None, settings=self.settings)
        self._metadata.update(
            model_id=cfg.model_id, revision=cfg.revision, max_len=cfg.max_len,
            lr=cfg.frozen["lr"], epoch=epoch, epochs=cfg.epochs, seed=self.settings.models.seed,
            torch_version=torch.__version__, transformers_version=transformers.__version__,
        )
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "_model"):
            raise NotFittedError("DistilBertClassifier: najpierw wywołaj fit() albo load().")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        import torch

        texts = list(texts)
        self._model.eval()
        scores = []
        batch_size = self.settings.models.distilbert.batch_size
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self._tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=self._max_len, return_token_type_ids=False, return_tensors="pt",
                ).to(self._model.device)
                logits = self._model(**inputs).logits
                scores.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        if not scores:
            return np.empty((0, self._model.config.num_labels), dtype=np.float32)
        return np.concatenate(scores)

    def save(self, directory: Path) -> Path:
        self._require_fitted()
        metadata = {**self._metadata, "max_len": self._max_len}
        text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            # HF zapisuje wiele plików; przerwany zapis sprząta katalog tymczasowy.
            with TemporaryDirectory(prefix=".saving-", dir=directory) as temporary:
                staging = Path(temporary)
                self._model.save_pretrained(staging)
                self._tokenizer.save_pretrained(staging)
                (staging / _DISTILBERT_METADATA).write_text(text, encoding="utf-8", newline="\n")
                for path in staging.iterdir():
                    path.rename(directory / path.name)
        except BaseException:
            with suppress(OSError):
                directory.rmdir()
            raise
        return directory

    @classmethod
    def load(cls, directory: Path, settings: Settings) -> DistilBertClassifier:
        metadata_path = directory / _DISTILBERT_METADATA
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label_hash = label_map_sha256(settings.dataset.label_map_path)
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(
                f"nie można odczytać metadanych lub mapy: {metadata_path}"
            ) from exc
        if not isinstance(metadata, dict) or metadata.get("label_map_sha256") != label_hash:
            raise ModelArtifactError("hash mapy etykiet niezgodny z artefaktem DistilBERT")

        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        label_map = dict(LabelMap.from_settings(settings).name_to_id)
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        id_to_name = {class_id: name for name, class_id in label_map.items()}
        if config.id2label != id_to_name or config.label2id != label_map:
            raise ModelArtifactError("mapa etykiet w config.json niezgodna z mapą projektu")
        classifier = cls(settings)
        classifier._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        classifier._model = AutoModelForSequenceClassification.from_pretrained(
            directory, config=config, local_files_only=True,
        ).to("cpu").eval()
        classifier._max_len = metadata.get("max_len", settings.models.distilbert.max_len)
        classifier._metadata = metadata
        return classifier
