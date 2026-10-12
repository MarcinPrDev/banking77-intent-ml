"""Testy interfejsu predyktora (models/predictor.py, factory.py, classifiers.py) — W2, pkt 5.

Fabryka: nazwa strategii -> klasa, odmowa bez zamrożonej konfiguracji, domyślny katalog
artefaktu. Test kontraktu (README → Architektura, LSP) — ten sam dla każdej strategii: trening
na danych zabawkowych (mapa 3 klas w tmp_path), `predict` == argmax `predict_proba`, zapis ->
odczyt daje identyczne predykcje, drugi zapis nic nie nadpisuje. Ciężkie modele podmienione
(bez pobierania, CPU): MiniLM — `embeddings.encode_texts` (HashingVectorizer), DistilBERT —
malutki model jak w `test_distilbert.py` (w CI, bez torch, pomijany).

Adapter, który rzuca jeszcze NotImplementedError szkieletu, jest pomijany — jego test włącza
się sam, gdy autor go napisze (`_skip_while_skeleton` do usunięcia razem z ostatnim szkieletem).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.features import embeddings
from banking77_intent_ml.models.classifiers import TfidfLrClassifier
from banking77_intent_ml.models.factory import STRATEGIES, build_predictor, load_predictor

_LABEL_MAP = {"card_arrival": 0, "exchange_rate": 1, "top_up_failed": 2}
_TRAIN_TEXTS = [
    "where is my card", "my card has not arrived", "card still not here",
    "what is the exchange rate", "rate for euro", "exchange rate fee",
    "top up failed", "my top up did not work", "top up declined",
]
_TRAIN_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2]
_QUERIES = ["card not here yet", "euro rate today", "top up not working"]
_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


# --- Fabryka -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_factory_builds_strategy_from_frozen_config(name):
    settings = load_settings()

    predictor = build_predictor(name, settings)

    assert type(predictor) is STRATEGIES[name]
    assert predictor.name == name
    assert predictor.settings is settings


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="nieznana strategia"):
        build_predictor("nie_ma_takiej", load_settings())


def test_factory_requires_frozen_config():
    settings = load_settings()
    minilm = replace(settings.models.minilm_knn, frozen=None)
    settings = replace(settings, models=replace(settings.models, minilm_knn=minilm))

    with pytest.raises(ValueError, match="frozen"):
        build_predictor("minilm_knn", settings)


def test_load_predictor_defaults_to_models_dir(monkeypatch):
    settings = load_settings()
    calls = []

    def fake_load(cls, directory, _settings):
        calls.append((cls, directory))
        return "predyktor"

    monkeypatch.setattr(TfidfLrClassifier, "load", classmethod(fake_load))

    assert load_predictor("tfidf_lr", settings) == "predyktor"
    assert calls == [(TfidfLrClassifier, settings.paths.models_dir / "tfidf_lr")]


# --- Test kontraktu ------------------------------------------------------------------------------


@pytest.fixture
def toy_settings(tmp_path, tmp_mlflow_settings):
    """Mapa 3 klas i katalogi w tmp_path; zamrożone konfiguracje z configu projektu."""
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(json.dumps(_LABEL_MAP), encoding="utf-8")
    base = tmp_mlflow_settings
    return replace(
        base,
        dataset=replace(base.dataset, label_map_path=label_map_path, n_classes=len(_LABEL_MAP)),
        paths=replace(base.paths, models_dir=tmp_path / "models", cache_dir=tmp_path / "cache"),
    )


def _fake_minilm(monkeypatch) -> None:
    """Embeddingi z HashingVectorizer: norma L2 jak moduł Normalize MiniLM, float32 jak model."""
    vectorizer = HashingVectorizer(n_features=64, alternate_sign=False)
    monkeypatch.setattr(
        embeddings, "encode_texts",
        lambda texts, settings: vectorizer.transform(list(texts)).toarray().astype(np.float32),
    )


def _tiny_distilbert(monkeypatch, settings):
    """Malutki tokenizer i model na CPU (jak `test_distilbert.py`); 2 epoki, zamrożona 1."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers import (
        BertTokenizerFast,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )

    from banking77_intent_ml.models import distilbert

    words = sorted({word for text in [*_TRAIN_TEXTS, *_QUERIES] for word in text.split()})
    # transformers 5: słownik przez `vocab` (dict); `vocab_file` w konstruktorze jest pomijany
    vocab = {token: index for index, token in enumerate([*_SPECIAL_TOKENS, *words])}

    def load_model(settings, label_map):
        config = DistilBertConfig(
            vocab_size=len(vocab), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
            max_position_embeddings=32, num_labels=len(label_map),
            id2label={index: name for name, index in label_map.items()}, label2id=dict(label_map),
        )
        return DistilBertForSequenceClassification(config)

    make_args = distilbert.training_arguments
    monkeypatch.setattr(
        distilbert, "_load_tokenizer", lambda settings: BertTokenizerFast(vocab=vocab)
    )
    monkeypatch.setattr(distilbert, "_load_model", load_model)
    monkeypatch.setattr(
        distilbert, "training_arguments",
        lambda variant, settings, output_dir: replace(
            make_args(variant, settings, output_dir), disable_tqdm=True
        ),
    )
    monkeypatch.setattr(distilbert.torch.cuda, "is_available", lambda: False)
    cfg = replace(
        settings.models.distilbert, max_len=16, epochs=2, batch_size=4,
        frozen={"lr": 1e-3, "epoch": 1},
    )
    return replace(settings, models=replace(settings.models, distilbert=cfg))


@contextmanager
def _skip_while_skeleton(name: str):
    try:
        yield
    except NotImplementedError as exc:
        if "W2 pkt 5" not in str(exc):
            raise
        pytest.skip(f"{name}: adapter jeszcze nie napisany — {exc}")


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_honours_contract(name, toy_settings, monkeypatch):
    settings = toy_settings
    if name == "minilm_knn":
        _fake_minilm(monkeypatch)
    elif name == "distilbert":
        settings = _tiny_distilbert(monkeypatch, settings)
    index = range(100, 100 + 2 * len(_TRAIN_TEXTS), 2)  # id wierszy jak w TrainVal, nie 0..n-1
    x_train = pd.Series(_TRAIN_TEXTS, index=index)
    y_train = pd.Series(_TRAIN_LABELS, index=index)

    with _skip_while_skeleton(name):
        predictor = build_predictor(name, settings)
        assert predictor.fit(x_train, y_train) is predictor

        predicted = predictor.predict(_QUERIES)
        scores = predictor.predict_proba(_QUERIES)
        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == (len(_QUERIES),)
        assert np.issubdtype(predicted.dtype, np.integer)
        assert set(predicted.tolist()) <= set(_LABEL_MAP.values())
        assert scores.shape == (len(_QUERIES), len(_LABEL_MAP))
        assert np.issubdtype(scores.dtype, np.floating)
        np.testing.assert_array_equal(predicted, scores.argmax(axis=1))
        assert predictor.predict(_QUERIES[:1]).shape == (1,)  # batch=1 jak w API

        directory = settings.paths.models_dir / name
        assert predictor.save(directory) == directory
        with pytest.raises(FileExistsError):
            predictor.save(directory)

        loaded = load_predictor(name, settings)  # domyślnie paths.models_dir/<name>
        assert type(loaded) is type(predictor)
        np.testing.assert_array_equal(loaded.predict(_QUERIES), predicted)
        np.testing.assert_allclose(loaded.predict_proba(_QUERIES), scores, rtol=0, atol=1e-6)
