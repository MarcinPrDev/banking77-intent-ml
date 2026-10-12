"""Testy cache embeddingów MiniLM (features/embeddings.py) — bez pobierania modelu.

Koder zastąpiony sztucznym: test sprawdza logikę cache (liczone raz, potem odczyt; inna
rewizja albo inny split = liczone od nowa), nie sam model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.features import embeddings


@pytest.fixture
def toy_data() -> TrainVal:
    return TrainVal(
        x_train=pd.Series(["top up failed", "card arrived", "exchange rate"], index=[3, 5, 8]),
        y_train=pd.Series([0, 1, 2], index=[3, 5, 8]),
        x_val=pd.Series(["top up declined", "card delivery"], index=[4, 9]),
        y_val=pd.Series([0, 1], index=[4, 9]),
        label_map={"a": 0, "b": 1, "c": 2},
    )


@pytest.fixture
def cache_settings(tmp_path):
    base = load_settings()
    return replace(base, paths=replace(base.paths, cache_dir=tmp_path / "cache"))


def _fake_encoder(calls: list):
    def encode_splits(data, settings):
        calls.append(settings.models.minilm_knn.revision)
        rng = np.random.default_rng(len(calls))
        train = rng.normal(size=(len(data.x_train), 4))
        val = rng.normal(size=(len(data.x_val), 4))
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        val /= np.linalg.norm(val, axis=1, keepdims=True)
        return train, val, {"device": "cpu", "load_seconds": 0.0, "encode_train_seconds": 0.1}

    return encode_splits


def test_embeddings_computed_once_then_read_from_cache(toy_data, cache_settings, monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "encode_splits", _fake_encoder(calls))

    first = embeddings.load_or_compute(toy_data, cache_settings)
    second = embeddings.load_or_compute(toy_data, cache_settings)

    assert len(calls) == 1
    assert not first.from_cache and second.from_cache
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.val, second.val)
    assert second.meta["revision"] == cache_settings.models.minilm_knn.revision
    assert second.meta["n_train"] == 3 and second.meta["n_val"] == 2
    assert second.meta["norm_min"] == pytest.approx(1.0)


def test_cache_recomputed_after_revision_change(toy_data, cache_settings, monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "encode_splits", _fake_encoder(calls))
    embeddings.load_or_compute(toy_data, cache_settings)

    minilm = replace(cache_settings.models.minilm_knn, revision="0" * 40)
    other = replace(cache_settings, models=replace(cache_settings.models, minilm_knn=minilm))
    result = embeddings.load_or_compute(toy_data, other)

    assert len(calls) == 2
    assert not result.from_cache and result.meta["revision"] == "0" * 40


def test_cache_recomputed_after_split_change(toy_data, cache_settings, monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "encode_splits", _fake_encoder(calls))
    embeddings.load_or_compute(toy_data, cache_settings)

    moved = replace(toy_data, x_val=toy_data.x_val.set_axis([4, 10]))
    result = embeddings.load_or_compute(moved, cache_settings)

    assert len(calls) == 2 and not result.from_cache
