"""Testy wspólnej ewaluacji (evaluation/metrics.py) na sztucznych danych — CI-safe.

Pilnują decyzji autora z 6.10: wejście = id klas z mapy; macro-F1 po WSZYSTKICH klasach z mapy
(klasa bez predykcji i bez przykładu nie znika ze średniej); wiersze macierzy = prawdziwa klasa.
"""

from __future__ import annotations

import pytest

from banking77_intent_ml.evaluation.metrics import evaluate

LABEL_MAP = {"a": 0, "b": 1, "c": 2}


def test_perfect_predictions():
    result = evaluate([0, 1, 2, 2], [0, 1, 2, 2], LABEL_MAP)
    assert result.accuracy == 1.0 and result.macro_f1 == 1.0
    assert int(result.confusion.to_numpy().trace()) == 4
    assert list(result.per_class.index) == ["a", "b", "c"]  # kolejność id z mapy
    assert result.metrics() == {"accuracy": 1.0, "macro_f1": 1.0}


def test_class_without_predictions_and_support_still_counts_in_macro_f1():
    result = evaluate([0, 0, 1], [0, 0, 1], LABEL_MAP)  # klasy "c" nie ma nigdzie
    assert result.accuracy == 1.0
    assert result.macro_f1 == pytest.approx(2 / 3)  # (1 + 1 + 0) / 3, a nie (1 + 1) / 2
    assert result.per_class.loc["c", "support"] == 0


def test_single_class_predictions_like_most_frequent_dummy():
    result = evaluate([0, 1, 2], [0, 0, 0], LABEL_MAP)
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.macro_f1 == pytest.approx(0.5 / 3)  # F1(a) = 0,5; F1(b) = F1(c) = 0


def test_confusion_rows_are_true_class_columns_predicted():
    result = evaluate([0, 1], [1, 1], LABEL_MAP)
    assert result.confusion.loc["a", "b"] == 1
    assert result.confusion.loc["b", "b"] == 1
    assert result.confusion.loc["b", "a"] == 0


def test_values_outside_label_map_are_rejected():
    with pytest.raises(ValueError, match="spoza mapy"):
        evaluate([0, 1], [0, 7], LABEL_MAP)
    with pytest.raises(ValueError, match="spoza mapy"):
        evaluate(["a", "b"], ["a", "b"], LABEL_MAP)  # nazwy zamiast id


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="tej samej długości"):
        evaluate([0, 1, 2], [0, 1], LABEL_MAP)
