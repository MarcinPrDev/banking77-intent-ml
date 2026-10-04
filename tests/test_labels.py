"""Testy mapy etykiet (data/labels.py) — bez prawdziwych danych, działają w CI.

Pilnują decyzji autora z 4.10: id wg sorted() nazw; istniejąca mapa nigdy nie jest nadpisywana.
"""

from __future__ import annotations

import pytest

from banking77_intent_ml.data.labels import (
    LabelMapError,
    build_label_map,
    load_label_map,
    write_or_verify_label_map,
)


def test_ids_follow_sorted_unique_names():
    assert build_label_map(["card_arrival", "age_limit", "card_arrival", "atm_support"]) == {
        "age_limit": 0,
        "atm_support": 1,
        "card_arrival": 2,
    }


def test_write_then_verify_same_map(tmp_path):
    path = tmp_path / "label_map.json"
    label_map = build_label_map(["b", "a"])
    assert write_or_verify_label_map(label_map, path) is True  # pierwszy zapis
    assert write_or_verify_label_map(label_map, path) is False  # ponownie: zgodna, bez zmian
    assert load_label_map(path) == {"a": 0, "b": 1}


def test_different_map_is_never_overwritten(tmp_path):
    path = tmp_path / "label_map.json"
    write_or_verify_label_map(build_label_map(["a", "b"]), path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(LabelMapError, match="nie nadpisuję"):
        write_or_verify_label_map(build_label_map(["a", "c"]), path)
    assert path.read_text(encoding="utf-8") == before
