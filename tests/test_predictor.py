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
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pickle import PicklingError

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import HashingVectorizer

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.features import embeddings
from banking77_intent_ml.models.classifiers import MiniLmKnnClassifier, TfidfLrClassifier
from banking77_intent_ml.models.factory import STRATEGIES, build_predictor, load_predictor
from banking77_intent_ml.models.persistence import ModelArtifactError, load_model

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


def test_factory_import_does_not_load_training_or_encoder_libraries(tmp_path):
    # Osobny proces: inne testy mogą już mieć MLflow w sys.modules.
    code = """
import sys
import banking77_intent_ml.models.factory

for name in ("mlflow", "torch", "transformers", "sentence_transformers"):
    assert name not in sys.modules, f"import fabryki predyktorów ładuje {name}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


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


@pytest.fixture(params=["tfidf_lr", "minilm_knn"])
def sklearn_classifier(request, monkeypatch):
    if request.param == "minilm_knn":
        _fake_minilm(monkeypatch)
    return STRATEGIES[request.param]


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
        if name != "distilbert" or "W2 pkt 5" not in str(exc):
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


def test_tfidf_adapter_uses_frozen_params_and_seed_without_mutation(toy_settings):
    settings = replace(toy_settings, models=replace(toy_settings.models, seed=17))
    original = deepcopy(settings.models.tfidf_lr)
    predictor = TfidfLrClassifier(settings)

    assert predictor.fit(_TRAIN_TEXTS, _TRAIN_LABELS) is predictor
    directory = predictor.save(settings.paths.models_dir / "frozen")
    model_path, = directory.glob("*.joblib")
    model = load_model(model_path, settings=settings)

    for step in ("tfidf", "logreg"):
        for key, value in original.frozen[step].items():
            assert model.named_steps[step].get_params()[key] == value
    assert model.named_steps["logreg"].random_state == 17
    assert settings.models.tfidf_lr == original
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "tfidf_lr"
    assert metadata["run_id"] is None  # fit adaptera nie tworzy runu MLflow


def test_sklearn_scores_keep_label_ids_when_training_omits_classes(tmp_path, sklearn_classifier):
    settings = load_settings()
    n_classes = settings.dataset.n_classes
    class_ids = [1, n_classes - 1]
    labels = [class_ids[label] for label in _TRAIN_LABELS[:6]]
    predictor = sklearn_classifier(settings).fit(_TRAIN_TEXTS[:6], labels)

    scores = predictor.predict_proba(_QUERIES)

    assert scores.shape == (len(_QUERIES), n_classes)
    absent = [class_id for class_id in range(n_classes) if class_id not in class_ids]
    np.testing.assert_array_equal(scores[:, absent], 0)
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    np.testing.assert_array_equal(predictor.predict(_QUERIES), scores.argmax(axis=1))
    assert set(predictor.predict(_QUERIES)) <= set(class_ids)

    directory = predictor.save(tmp_path / "sparse-classes")
    # Odczyt korzysta z artefaktu, nawet gdy nie ma konfiguracji do ponownego fitu.
    without_frozen = replace(getattr(settings.models, predictor.name), frozen=None)
    settings = replace(
        settings, models=replace(settings.models, **{predictor.name: without_frozen})
    )
    loaded = sklearn_classifier.load(directory, settings)
    assert loaded is not predictor
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), scores)
    np.testing.assert_array_equal(loaded.predict(_QUERIES), predictor.predict(_QUERIES))


def test_sklearn_save_rejects_existing_empty_directory(toy_settings, sklearn_classifier):
    predictor = sklearn_classifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = toy_settings.paths.models_dir / "empty"
    directory.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        predictor.save(directory)

    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("error_type", [OSError, PicklingError])
def test_sklearn_failed_save_allows_retry(
    toy_settings, monkeypatch, error_type, sklearn_classifier,
):
    predictor = sklearn_classifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = toy_settings.paths.models_dir / "retry"
    error = error_type("przerwany zapis modelu")

    def broken_dump(model, file, **kwargs):
        file.write(b"niekompletny model")
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(joblib, "dump", broken_dump)
        with pytest.raises(error_type, match="przerwany zapis") as caught:
            predictor.save(directory)

    assert caught.value is error
    assert not directory.exists()
    assert predictor.save(directory) == directory
    loaded = sklearn_classifier.load(directory, toy_settings)
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), predictor.predict_proba(_QUERIES))


def test_sklearn_failed_save_preserves_other_files_and_original_error(
    toy_settings, monkeypatch, sklearn_classifier,
):
    predictor = sklearn_classifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = toy_settings.paths.models_dir / "nonempty"
    other_file = directory / "other.txt"
    error = OSError("przerwany zapis modelu")

    def broken_dump(model, file, **kwargs):
        other_file.write_text("zachowaj", encoding="utf-8")
        raise error

    monkeypatch.setattr(joblib, "dump", broken_dump)
    with pytest.raises(OSError, match="przerwany zapis") as caught:
        predictor.save(directory)

    assert caught.value is error
    assert list(directory.iterdir()) == [other_file]
    assert other_file.read_text(encoding="utf-8") == "zachowaj"


@pytest.mark.parametrize("method", ["predict", "predict_proba", "save"])
def test_sklearn_unfitted_reports_required_fit_or_load(toy_settings, method, sklearn_classifier):
    predictor = sklearn_classifier(toy_settings)
    directory = toy_settings.paths.models_dir / "unfitted"
    argument = directory if method == "save" else _QUERIES

    with pytest.raises(NotFittedError, match=r"fit\(\).*load\(\)"):
        getattr(predictor, method)(argument)

    assert not directory.exists()


@pytest.mark.parametrize("field", ["label_map_sha256", "sklearn_version"])
def test_sklearn_load_checks_artifact_metadata(toy_settings, field, sklearn_classifier):
    predictor = sklearn_classifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = predictor.save(toy_settings.paths.models_dir / "invalid-metadata")
    metadata_path, = directory.glob("*.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = "niezgodne"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelArtifactError):
        sklearn_classifier.load(directory, toy_settings)


@pytest.mark.parametrize(
    "n_neighbors, weights, metric", [(1, "uniform", "cosine"), (3, "distance", "euclidean")],
)
def test_minilm_uses_frozen_params_and_encodes_each_request(
    toy_settings, monkeypatch, n_neighbors, weights, metric,
):
    cfg = replace(
        toy_settings.models.minilm_knn, metric=metric, k_values=(7,), weights=("other",),
        frozen={"n_neighbors": n_neighbors, "weights": weights},
    )
    settings = replace(toy_settings, models=replace(toy_settings.models, minilm_knn=cfg))
    original = deepcopy(cfg)
    vectorizer = HashingVectorizer(n_features=64, alternate_sign=False)
    calls = []

    def encode(texts, received_settings):
        calls.append((list(texts), received_settings))
        return vectorizer.transform(list(texts)).toarray().astype(np.float32)

    monkeypatch.setattr(embeddings, "encode_texts", encode)
    predictor = MiniLmKnnClassifier(settings)
    assert predictor.fit(_TRAIN_TEXTS, _TRAIN_LABELS) is predictor
    queries = [*_QUERIES, "card arrived late"]
    scores = predictor.predict_proba(queries)
    predicted = predictor.predict(queries)
    assert [texts for texts, _ in calls] == [_TRAIN_TEXTS, queries, queries]
    assert all(received is settings for _, received in calls)
    assert cfg == original

    directory = predictor.save(settings.paths.models_dir / "minilm")
    model_path, = directory.glob("*.joblib")
    model = load_model(model_path, settings=settings)
    assert model.n_neighbors == n_neighbors
    assert model.weights == weights
    assert model.metric == metric
    expected = model.predict_proba(vectorizer.transform(queries).toarray().astype(np.float32))
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(predicted, expected.argmax(axis=1))
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "minilm_knn"
    assert metadata["run_id"] is None
    assert metadata["model_id"] == cfg.model_id
    assert metadata["revision"] == cfg.revision


def test_minilm_fit_requires_frozen_before_encoding(toy_settings, monkeypatch):
    cfg = replace(toy_settings.models.minilm_knn, frozen=None)
    settings = replace(toy_settings, models=replace(toy_settings.models, minilm_knn=cfg))

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Nie koduj tekstów bez zamrożonej konfiguracji")

    monkeypatch.setattr(embeddings, "encode_texts", unexpected_encode)
    with pytest.raises(ValueError, match="frozen"):
        MiniLmKnnClassifier(settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)


def test_minilm_load_warms_encoder_before_returning(toy_settings, monkeypatch):
    _fake_minilm(monkeypatch)
    predictor = MiniLmKnnClassifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    expected = predictor.predict_proba(_QUERIES)
    directory = predictor.save(toy_settings.paths.models_dir / "warmup")
    encode = embeddings.encode_texts
    calls = []

    def record_encode(texts, settings):
        calls.append((list(texts), settings))
        return encode(texts, settings)

    monkeypatch.setattr(embeddings, "encode_texts", record_encode)
    loaded = MiniLmKnnClassifier.load(directory, toy_settings)

    # Enkoder gotowy już po load; nie trzeba pierwszego żądania /predict.
    assert len(calls) == 1
    warmup_texts, settings = calls[0]
    assert len(warmup_texts) == 1 and warmup_texts[0].strip()
    assert settings is toy_settings
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), expected)


def test_minilm_load_propagates_encoder_failure(toy_settings, monkeypatch):
    _fake_minilm(monkeypatch)
    predictor = MiniLmKnnClassifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = predictor.save(toy_settings.paths.models_dir / "failed-warmup")
    error = OSError("brak wag enkodera")

    def broken_encode(*args, **kwargs):
        raise error

    monkeypatch.setattr(embeddings, "encode_texts", broken_encode)
    with pytest.raises(OSError, match="brak wag enkodera") as caught:
        MiniLmKnnClassifier.load(directory, toy_settings)
    assert caught.value is error


@pytest.mark.parametrize("field", ["model_id", "revision"])
@pytest.mark.parametrize("mismatch", ["settings", "missing_metadata"])
def test_minilm_load_rejects_incompatible_encoder_before_unpickling(
    toy_settings, monkeypatch, field, mismatch,
):
    _fake_minilm(monkeypatch)
    predictor = MiniLmKnnClassifier(toy_settings).fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    directory = predictor.save(toy_settings.paths.models_dir / "encoder")
    settings = toy_settings
    if mismatch == "settings":
        cfg = replace(settings.models.minilm_knn, **{field: "inny-enkoder"})
        settings = replace(settings, models=replace(settings.models, minilm_knn=cfg))
    else:
        metadata_path, = directory.glob("*.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata[field]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Nie wczytuj indeksu kNN z niezgodnym enkoderem")

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Nie rozgrzewaj enkodera przed walidacją metadanych")

    monkeypatch.setattr(joblib, "load", unexpected_load)
    monkeypatch.setattr(embeddings, "encode_texts", unexpected_encode)
    with pytest.raises(ModelArtifactError, match=field):
        MiniLmKnnClassifier.load(directory, settings)
