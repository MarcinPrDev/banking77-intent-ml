"""Testy wczytywania train/val z zamrożonego splitu (data/datasets.py) — sztuczne pliki, CI-safe.

Pilnują, że indeksy są sprawdzane hashem z manifestu (zmieniony plik = błąd), a etykiety to id
z mapy etykiet; indeks serii = pozycja wiersza w pliku.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.data.splits import SplitError, ids_sha256

LABEL_MAP = {"a": 0, "b": 1, "c": 2}
ROWS = [("t0", "a"), ("t1", "a"), ("t2", "b"), ("t3", "b"), ("t4", "c"), ("t5", "c")]
TRAIN_IDS, VAL_IDS = [0, 2, 4, 5], [1, 3]


def _settings(tmp_path, val_ids_on_disk=VAL_IDS):
    raw_dir, splits_dir = tmp_path / "raw", tmp_path / "splits"
    raw_dir.mkdir()
    splits_dir.mkdir()
    lines = ["text,category"] + [f"{text},{category}" for text, category in ROWS]
    (raw_dir / "train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (splits_dir / "train_ids.json").write_text(json.dumps(TRAIN_IDS), encoding="utf-8")
    (splits_dir / "val_ids.json").write_text(json.dumps(val_ids_on_disk), encoding="utf-8")
    manifest = {"sha256_train_ids": ids_sha256(TRAIN_IDS), "sha256_val_ids": ids_sha256(VAL_IDS)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "label_map.json").write_text(json.dumps(LABEL_MAP), encoding="utf-8")

    base = load_settings()
    dataset = replace(
        base.dataset,
        n_classes=3,
        train=replace(base.dataset.train, n_rows=len(ROWS)),
        label_map_path=tmp_path / "label_map.json",
    )
    return replace(
        base,
        dataset=dataset,
        paths=replace(base.paths, raw_dir=raw_dir, splits_dir=splits_dir),
        split=replace(base.split, manifest_path=tmp_path / "manifest.json"),
    )


def test_train_val_follow_frozen_ids_and_label_map(tmp_path):
    data = load_train_val(_settings(tmp_path))
    assert data.x_train.index.tolist() == TRAIN_IDS
    assert data.x_val.index.tolist() == VAL_IDS
    assert data.y_train.tolist() == [0, 1, 2, 2]  # a, b, c, c jako id z mapy
    assert data.y_val.tolist() == [0, 1]
    assert data.x_val.tolist() == ["t1", "t3"]


def test_changed_ids_file_is_rejected(tmp_path):
    with pytest.raises(SplitError, match="hash niezgodny"):
        load_train_val(_settings(tmp_path, val_ids_on_disk=[1, 4]))
