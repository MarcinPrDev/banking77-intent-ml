"""Testy loadera BANKING77 na małych, sztucznych CSV (bez prawdziwych danych — działają w CI).

Pilnują decyzji z `data/loader.py`: (1) brak = tylko pusty napis, a "None" / "NA" to wiadomości;
(2) indeks = pozycja w pliku, kolejność bez zmian; komunikaty błędów podają pozycje, nigdy treść.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.loader import (
    DataValidationError,
    _read_and_validate,
    load_test,
    load_train,
)

HEADER = "text,category\n"


def _csv(tmp_path: Path, body: str, name: str = "data.csv", header: str = HEADER) -> Path:
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def test_na_like_strings_are_messages_not_missing(tmp_path):
    path = _csv(tmp_path, "None,a\nNA,b\nnull,a\nN/A,b\n")
    df = _read_and_validate(path, expected_rows=4, n_classes=2)
    assert df["text"].tolist() == ["None", "NA", "null", "N/A"]


def test_empty_or_whitespace_text_raises_with_positions_not_content(tmp_path):
    path = _csv(tmp_path, "TRESC-WIADOMOSCI,a\n,b\n   ,a\n")
    with pytest.raises(DataValidationError) as exc:
        _read_and_validate(path, expected_rows=3, n_classes=2)
    message = str(exc.value)
    assert "'text'" in message and "[1, 2]" in message
    assert "TRESC-WIADOMOSCI" not in message  # błąd mówi gdzie, nie co


def test_empty_category_raises(tmp_path):
    path = _csv(tmp_path, "ok,a\nok,\n")
    with pytest.raises(DataValidationError, match="'category'"):
        _read_and_validate(path, expected_rows=2, n_classes=1)


@pytest.mark.parametrize(
    ("header", "body", "rows", "n_classes", "fragment"),
    [
        ("tekst,kategoria\n", "x,a\n", 1, 1, "kolumny"),
        (HEADER, "x,a\n", 2, 1, "wierszy"),
        (HEADER, "x,a\ny,a\n", 2, 2, "kategorii"),
    ],
    ids=["zly-naglowek", "zla-liczba-wierszy", "zla-liczba-kategorii"],
)
def test_structure_mismatch_raises(tmp_path, header, body, rows, n_classes, fragment):
    path = _csv(tmp_path, body, header=header)
    with pytest.raises(DataValidationError, match=fragment):
        _read_and_validate(path, expected_rows=rows, n_classes=n_classes)


def test_missing_file_points_to_fetch_data(tmp_path):
    with pytest.raises(DataValidationError, match="fetch_data") as exc:
        _read_and_validate(tmp_path / "brak.csv", expected_rows=1, n_classes=1)
    assert isinstance(exc.value, ValueError)  # kod łapiący ValueError złapie też ten błąd


def test_index_is_position_in_file_and_order_is_kept(tmp_path):
    path = _csv(tmp_path, "c,a\na,b\nb,a\n")
    df = _read_and_validate(path, expected_rows=3, n_classes=2)
    assert df.index.equals(pd.RangeIndex(3))
    assert df["text"].tolist() == ["c", "a", "b"]  # kolejność z pliku, bez sortowania


def test_load_train_and_load_test_read_their_own_files(tmp_path):
    base = load_settings()
    settings = replace(
        base,
        paths=replace(base.paths, raw_dir=tmp_path),
        dataset=replace(
            base.dataset,
            n_classes=2,
            train=replace(base.dataset.train, n_rows=2),
            test=replace(base.dataset.test, n_rows=3),
        ),
    )
    _csv(tmp_path, "t1,a\nt2,b\n", name=settings.dataset.train.filename)
    _csv(tmp_path, "x1,a\nx2,b\nx3,a\n", name=settings.dataset.test.filename)
    assert load_train(settings)["text"].tolist() == ["t1", "t2"]
    assert load_test(settings)["text"].tolist() == ["x1", "x2", "x3"]
