"""Testy zapisu i odczytu modelu (models/persistence.py) — N2, decyzje autora 9.10.2026.

Dwa poziomy dowodu „nowy proces daje te same predykcje” (etykiety i prawdopodobieństwa co do
bitu): sztuczne dane — działa też w CI; 20 pierwszych zdań z zamrożonej walidacji — tylko
lokalnie, bo CI nie ma danych.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.models.baselines import build_tfidf_lr
from banking77_intent_ml.models.persistence import (
    ModelArtifactError,
    build_metadata,
    load_model,
    save_model,
)

_SETTINGS = load_settings()
HAS_DATA = (_SETTINGS.paths.raw_dir / _SETTINGS.dataset.train.filename).exists()


@pytest.fixture
def toy_texts() -> tuple[pd.Series, pd.Series]:
    """Sztuczne teksty i id klas do treningu małego pipeline'u (3 klasy)."""
    texts = pd.Series([
        "top up failed", "top up pending", "card not arrived", "card arrived late",
        "exchange rate fee", "bad exchange rate",
    ])
    labels = pd.Series([0, 0, 1, 1, 2, 2])
    return texts, labels


def _assert_predictions_in_new_process(model, path, texts, tmp_path):
    texts_path = tmp_path / "texts.json"
    texts_path.write_text(json.dumps(list(texts)), encoding="utf-8")
    labels_path = tmp_path / "labels.npy"
    proba_path = tmp_path / "proba.npy"
    code = """
import json
import sys
from pathlib import Path

import numpy as np

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.models.persistence import load_model

model = load_model(Path(sys.argv[1]), settings=load_settings())
texts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
np.save(sys.argv[3], model.predict(texts))
np.save(sys.argv[4], model.predict_proba(texts))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(texts_path), str(labels_path), str(proba_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert np.array_equal(model.predict(texts), np.load(labels_path, allow_pickle=False))
    assert np.array_equal(model.predict_proba(texts), np.load(proba_path, allow_pickle=False))


def _unexpected_load(*args, **kwargs):
    pytest.fail("joblib.load nie powinien ruszyć przed sprawdzeniem metadanych")


def test_toy_model_gives_identical_predictions_in_new_process(toy_texts, tmp_path):
    """Zapis -> wczytanie w nowym procesie (subprocess) -> te same etykiety i prawdopodobieństwa."""
    texts, labels = toy_texts
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(texts, labels)
    metadata = build_metadata(model_name="tfidf_lr", run_id="pytest", settings=_SETTINGS)
    manifest = json.loads(_SETTINGS.split.manifest_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == "tfidf_lr"
    assert metadata["run_id"] == "pytest"
    assert metadata["sklearn_version"] == sklearn.__version__
    assert metadata["numpy_version"] == np.__version__
    assert len(metadata["label_map_sha256"]) == 64
    assert metadata["data_commit"] == _SETTINGS.dataset.commit
    assert metadata["sha256_val_ids"] == manifest["sha256_val_ids"]

    path = tmp_path / "models" / "toy.joblib"
    assert save_model(model, path, metadata=metadata) == path
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) == metadata
    _assert_predictions_in_new_process(
        model, path, ["top up declined", "card delivery", "exchange rate today"], tmp_path
    )

    saved_model = path.read_bytes()
    saved_metadata = path.with_suffix(".json").read_bytes()
    with pytest.raises(FileExistsError):
        save_model(model, path, metadata={**metadata, "run_id": "inny-run"})
    assert path.read_bytes() == saved_model
    assert path.with_suffix(".json").read_bytes() == saved_metadata


def test_failed_save_removes_partial_model(toy_texts, tmp_path, monkeypatch):
    """Nieudany dump nie zostawia pliku blokującego ponowny zapis."""
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(*toy_texts)
    metadata = build_metadata(model_name="tfidf_lr", run_id=None, settings=_SETTINGS)
    path = tmp_path / "toy.joblib"
    error = OSError("przerwany zapis modelu")

    def broken_dump(model, file, **kwargs):
        file.write(b"niekompletny model")
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(joblib, "dump", broken_dump)
        with pytest.raises(OSError, match="przerwany zapis") as caught:
            save_model(model, path, metadata=metadata)

    assert caught.value is error
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    assert save_model(model, path, metadata=metadata) == path
    loaded = load_model(path, settings=_SETTINGS)
    assert np.array_equal(loaded.predict(toy_texts[0]), model.predict(toy_texts[0]))


@pytest.mark.skipif(not HAS_DATA, reason="brak danych w dane/ (CI) — test tylko lokalny")
def test_validation_sentences_identical_after_reload_in_new_process(tmp_path):
    """TF-IDF+LR z configu, fit na części treningowej; 20 pierwszych zdań z walidacji: te same
    predykcje po wczytaniu w nowym procesie."""
    data = load_train_val(_SETTINGS)
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(data.x_train, data.y_train)
    metadata = build_metadata(model_name="tfidf_lr", run_id=None, settings=_SETTINGS)
    path = save_model(model, tmp_path / "tfidf_lr.joblib", metadata=metadata)
    _assert_predictions_in_new_process(model, path, data.x_val.iloc[:20], tmp_path)


def test_load_refuses_missing_metadata(toy_texts, tmp_path, monkeypatch):
    """Sam plik .joblib bez metadanych -> ModelArtifactError."""
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(*toy_texts)
    path = tmp_path / "toy.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(joblib, "load", _unexpected_load)

    with pytest.raises(ModelArtifactError, match="metadanych"):
        load_model(path, settings=_SETTINGS)


def test_load_refuses_different_label_map(toy_texts, tmp_path, monkeypatch):
    """Metadane z innym hashem mapy etykiet -> ModelArtifactError."""
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(*toy_texts)
    metadata = build_metadata(model_name="tfidf_lr", run_id=None, settings=_SETTINGS)
    path = save_model(model, tmp_path / "toy.joblib", metadata=metadata)

    label_map = json.loads(_SETTINGS.dataset.label_map_path.read_text(encoding="utf-8"))
    map_path = tmp_path / "label_map.json"
    # Te same przypisania, inna kolejność kluczy i formatowanie — hash ma pozostać zgodny.
    map_path.write_text(json.dumps(dict(reversed(list(label_map.items())))), encoding="utf-8")
    settings = replace(_SETTINGS, dataset=replace(_SETTINGS.dataset, label_map_path=map_path))
    assert build_metadata(model_name="tfidf_lr", run_id=None, settings=settings) == metadata
    loaded = load_model(path, settings=settings)
    assert np.array_equal(loaded.predict(toy_texts[0]), model.predict(toy_texts[0]))

    monkeypatch.setattr(joblib, "load", _unexpected_load)
    first, second = list(label_map)[:2]
    label_map[first], label_map[second] = label_map[second], label_map[first]
    map_path.write_text(json.dumps(label_map), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="mapy etykiet"):
        load_model(path, settings=settings)

    metadata["label_map_sha256"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="mapy etykiet"):
        load_model(path, settings=_SETTINGS)


def test_load_refuses_different_sklearn_version(toy_texts, tmp_path, monkeypatch):
    """Metadane z inną wersją sklearn -> ModelArtifactError."""
    model = build_tfidf_lr(_SETTINGS.models.tfidf_lr, seed=_SETTINGS.models.seed)
    model.fit(*toy_texts)
    metadata = build_metadata(model_name="tfidf_lr", run_id=None, settings=_SETTINGS)
    path = save_model(model, tmp_path / "toy.joblib", metadata=metadata)
    metadata["sklearn_version"] = "0.0.0"
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(joblib, "load", _unexpected_load)

    with pytest.raises(ModelArtifactError, match="sklearn"):
        load_model(path, settings=_SETTINGS)
