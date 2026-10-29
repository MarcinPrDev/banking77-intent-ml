"""Testy testu istotności (evaluation/significance.py) na danych syntetycznych — bez MLflow i testu.

Wyrównanie przykładów, dokładny McNemar, korekta Holma i parowany bootstrap.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.evaluation import significance


def _predictions(y_true, y_pred, row_ids=None) -> pd.DataFrame:
    """Ramka jak `predictions_test.csv`: `row_id, y_true, y_pred, score`."""
    n = len(y_true)
    return pd.DataFrame({
        "row_id": np.arange(n) if row_ids is None else row_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "score": np.full(n, 0.5),
    })


def test_identical_predictions_have_no_discordant_pairs():
    """Te same predykcje dwóch ścieżek: b = c = 0, p = 1,0."""
    frame = _predictions([0, 1, 2, 3], [0, 0, 2, 2], [30, 10, 40, 20])
    original = frame.copy(deep=True)
    correct = significance.correct_by_model({"a": frame, "b": frame.iloc[::-1]})
    expected = pd.DataFrame(
        {"a": [False, False, True, True], "b": [False, False, True, True]},
        index=pd.Index([10, 20, 30, 40], name="row_id"),
    )
    pd.testing.assert_frame_equal(correct, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert significance.mcnemar_exact(correct.a.to_numpy(), correct.b.to_numpy()) == (0, 0, 1.0)


def test_mcnemar_matches_binomtest_for_known_counts():
    """Zbuduj predykcje o znanych b i c; p równe `scipy.stats.binomtest(min(b, c), b + c, 0.5)`."""
    # 7 poprawnych tylko w A, 1 tylko w B, 2 wspólne trafienia, 2 wspólne błędy.
    correct = significance.correct_by_model({
        "a": _predictions([0] * 12, [0] * 7 + [1] + [0, 0, 1, 1]),
        "b": _predictions([0] * 12, [1] * 7 + [0] + [0, 0, 1, 1]),
    })
    b, c, p = significance.mcnemar_exact(correct.a.to_numpy(), correct.b.to_numpy())
    assert (b, c) == (7, 1)
    assert p == pytest.approx(0.0703125)
    assert p == binomtest(1, 8, 0.5, alternative="two-sided").pvalue
    assert significance.mcnemar_exact(correct.b.to_numpy(), correct.a.to_numpy()) == (1, 7, p)


def test_holm_adjusts_known_p_values():
    """[0,01; 0,04; 0,03] -> [0,03; 0,06; 0,06] (kolejność wejścia zachowana)."""
    values = [0.01, 0.04, 0.03]
    assert significance.holm(values) == pytest.approx([0.03, 0.06, 0.06])
    assert values == [0.01, 0.04, 0.03]
    assert significance.holm([0.8, 0.9, 0.7]) == [1.0, 1.0, 1.0]
    assert significance.holm([0.02, 0.02, 0.0]) == pytest.approx([0.04, 0.04, 0.0])
    assert significance.holm([0.2]) == [0.2]
    assert significance.holm([]) == []


def test_bootstrap_is_reproducible_with_same_seed():
    """Dwa wywołania z tym samym seedem dają identyczne tablice różnic."""
    correct = pd.DataFrame({
        "a": [True, False, True, True],
        "b": [False, True, True, False],
        "c": [False, False, True, False],
    })
    pairs = [("a", "b"), ("b", "c"), ("a", "c"), ("b", "a")]
    first = significance.paired_bootstrap_diffs(correct, pairs, n_bootstrap=103, seed=7, chunk=11)
    second = significance.paired_bootstrap_diffs(correct, pairs, n_bootstrap=103, seed=7, chunk=11)
    whole = significance.paired_bootstrap_diffs(correct, pairs, n_bootstrap=103, seed=7, chunk=103)
    other_seed = significance.paired_bootstrap_diffs(
        correct, pairs, n_bootstrap=103, seed=8, chunk=11
    )
    assert set(first) == set(pairs)
    for pair in pairs:
        assert first[pair].shape == (103,)
        np.testing.assert_array_equal(first[pair], second[pair])
        np.testing.assert_array_equal(first[pair], whole[pair])
    assert not np.array_equal(first[("a", "b")], other_seed[("a", "b")])
    assert np.unique(first[("a", "b")]).size > 1
    assert np.all(np.isin(first[("a", "b")], np.arange(-4, 5) / 4))
    np.testing.assert_array_equal(first[("a", "b")], -first[("b", "a")])
    # Addytywność każdej próbki wymaga tych samych wierszy we wszystkich parach.
    np.testing.assert_array_equal(
        first[("a", "c")], first[("a", "b")] + first[("b", "c")]
    )


def test_bootstrap_ci_is_zero_for_identical_models():
    """Identyczne ścieżki: wszystkie różnice = 0, przedział [0, 0]."""
    correct = pd.DataFrame({"a": [True, False, True], "b": [True, False, True]})
    diffs = significance.paired_bootstrap_diffs(
        correct, [("a", "b")], n_bootstrap=101, seed=42, chunk=9
    )[("a", "b")]
    np.testing.assert_array_equal(diffs, np.zeros(101))
    np.testing.assert_array_equal(np.quantile(diffs, [0.025, 0.975]), [0, 0])


def test_misaligned_row_ids_or_labels_raise():
    """Różne zbiory `row_id` albo różne `y_true` dla tego samego `row_id` -> `ValueError`."""
    base = _predictions([0, 1, 2], [0, 0, 2])
    invalid = [
        base.assign(row_id=[0, 1, 3]),
        base.assign(y_true=[0, 2, 2]),
        base.assign(row_id=[0, 0, 2]),
        base.assign(row_id=[0, None, 2]),
        base.assign(y_pred=[0, None, 2]),
        base.iloc[:0],
    ]
    for frame in invalid:
        with pytest.raises(ValueError):
            significance.correct_by_model({"a": base, "b": frame})
    with pytest.raises(ValueError):
        significance.correct_by_model({})


@pytest.mark.parametrize("alpha, expected_reject", [(0.015625, True), (0.01, False)])
def test_significance_table_uses_config_and_percentage_points(alpha, expected_reject):
    """Wszystkie trafienia kontra same błędy: dokładne p, Holm, znak i skala CI."""
    settings = load_settings()
    cfg = replace(
        settings.evaluation.significance,
        pairs=(("a", "b"), ("b", "a")),
        n_bootstrap=23,
        ci_level=0.8,
        alpha=alpha,
    )
    settings = replace(
        settings,
        evaluation=replace(settings.evaluation, significance=cfg),
        dataset=replace(settings.dataset, test=replace(settings.dataset.test, n_rows=8)),
    )
    table = significance.significance_table({
        "a": _predictions([0] * 8, [0] * 8),
        "b": _predictions([0] * 8, [1] * 8).iloc[::-1],
    }, settings)
    expected = pd.DataFrame({
        "a": ["a", "b"], "b": ["b", "a"], "n": [8, 8],
        "acc_a": [1.0, 0.0], "acc_b": [0.0, 1.0], "diff_pp": [100.0, -100.0],
        "b_only": [8, 0], "c_only": [0, 8], "p": [0.0078125, 0.0078125],
        "p_holm": [0.015625, 0.015625], "reject": [expected_reject] * 2,
        "ci_low_pp": [100.0, -100.0], "ci_high_pp": [100.0, -100.0],
    })
    pd.testing.assert_frame_equal(table, expected)


@pytest.mark.parametrize("actual_n", [3, 5])
def test_significance_table_rejects_wrong_sample_size(actual_n):
    """Zgodne ramki wszystkich modeli nie wystarczą: N musi odpowiadać configowi."""
    settings = load_settings()
    settings = replace(
        settings,
        dataset=replace(settings.dataset, test=replace(settings.dataset.test, n_rows=4)),
    )
    names = {name for pair in settings.evaluation.significance.pairs for name in pair}
    predictions = {name: _predictions([0] * actual_n, [0] * actual_n) for name in names}
    with pytest.raises(ValueError, match=rf"{actual_n}.*4"):
        significance.significance_table(predictions, settings)


@pytest.mark.parametrize("pair", [("unknown_model", "a"), ("a", "unknown_model")])
def test_unknown_model_in_pair_raises_value_error(pair):
    """Nieznana ścieżka po dowolnej stronie pary daje błąd z jej nazwą."""
    correct = pd.DataFrame({"a": [True, False]})
    with pytest.raises(ValueError, match="unknown_model"):
        significance.paired_bootstrap_diffs(correct, [pair], n_bootstrap=5, seed=42)
    settings = load_settings()
    settings = replace(
        settings,
        dataset=replace(settings.dataset, test=replace(settings.dataset.test, n_rows=2)),
        evaluation=replace(
            settings.evaluation,
            significance=replace(settings.evaluation.significance, pairs=(pair,)),
        ),
    )
    with pytest.raises(ValueError, match="unknown_model"):
        significance.significance_table({"a": _predictions([0, 0], [0, 1])}, settings)
