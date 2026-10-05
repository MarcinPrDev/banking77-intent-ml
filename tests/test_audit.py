"""Testy audytu integralności testu i podzbiorów S2 (data/audit.py) — sztuczne dane, CI-safe.

Pilnują decyzji autora z 5.10: czysty podzbiór = bez odpowiednika w train po normalizacji,
próbka pod LLM = per_class na klasę z czystego podzbioru, deterministyczna dla seeda.
"""

from __future__ import annotations

import pandas as pd
import pytest

from banking77_intent_ml.data.audit import AuditError, audit_test_integrity, make_eval_subsets

LABEL_MAP = {"a": 0, "b": 1, "c": 2}


def _test_frame() -> pd.DataFrame:
    rows = [(f"query {cls} {i}", cls) for cls in LABEL_MAP for i in range(8)]
    rows[0] = ("Card arrived?", "a")  # odpowiednik w train po normalizacji, ta sama etykieta
    rows[8] = ("Top-up pending!", "b")  # odpowiednik w train, ale tam z etykietą "c"
    return pd.DataFrame(rows, columns=["text", "category"])


def _train_frame() -> pd.DataFrame:
    rows = [("card ARRIVED", "a"), ("top up pending", "c")]
    rows += [(f"train only {i}", "a") for i in range(5)]
    return pd.DataFrame(rows, columns=["text", "category"])


def test_audit_passes_for_balanced_test_matching_label_map():
    audit = audit_test_integrity(_test_frame(), LABEL_MAP, n_classes=3)
    assert audit == {
        "n_rows": 24,
        "n_classes": 3,
        "rows_per_class": 8,
        "names_match_label_map": True,
    }


def test_audit_rejects_unbalanced_test():
    test = _test_frame()
    test.loc[1, "category"] = "b"  # a: 7, b: 9
    with pytest.raises(AuditError, match="niezrównoważony"):
        audit_test_integrity(test, LABEL_MAP, n_classes=3)


def test_audit_rejects_category_names_outside_label_map():
    with pytest.raises(AuditError, match="mapa etykiet"):
        audit_test_integrity(_test_frame(), {"a": 0, "b": 1, "d": 2}, n_classes=3)


def test_clean_subset_drops_normalized_overlap_and_counts_label_mismatch():
    subsets = make_eval_subsets(_test_frame(), _train_frame(), per_class=2, random_state=42)
    assert subsets.n_overlap == 2
    assert subsets.n_overlap_label_mismatch == 1  # "Top-up pending!" ma w train etykietę "c"
    assert 0 not in subsets.clean_ids and 8 not in subsets.clean_ids
    assert len(subsets.clean_ids) == 22


def test_llm_sample_is_per_class_from_clean_subset_and_deterministic():
    test = _test_frame()
    first = make_eval_subsets(test, _train_frame(), per_class=2, random_state=42)
    assert set(first.llm_ids) <= set(first.clean_ids)
    assert test.loc[first.llm_ids, "category"].value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}
    assert first == make_eval_subsets(test, _train_frame(), per_class=2, random_state=42)
    other = make_eval_subsets(test, _train_frame(), per_class=2, random_state=7)
    assert other.llm_ids != first.llm_ids and other.clean_ids == first.clean_ids


def test_too_few_clean_texts_in_a_class_raises():
    with pytest.raises(AuditError, match="za mało czystych"):
        make_eval_subsets(_test_frame(), _train_frame(), per_class=8, random_state=42)
