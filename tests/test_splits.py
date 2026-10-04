"""Testy zamrożonego podziału (data/splits.py) na sztucznych danych — bez CSV, działają w CI.

Pilnują decyzji autora z 4.10: duplikaty po normalizacji razem, grupa konfliktowa w train,
stratyfikacja, determinizm (seed), przepis hasha i zamrożenie (zapis albo weryfikacja, nigdy
nadpisanie).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.splits import (
    SplitError,
    build_manifest,
    ids_sha256,
    make_splits,
    normalize_text,
    write_or_verify_split,
)

CLASSES = ("a_cls", "b_cls", "c_cls")
DUPLICATES = ["Card arrived?", "card ARRIVED", "Card, arrived!"]  # jeden klucz: "card arrived"
CONFLICT = [("Top-up pending!", "x_pending"), ("top up pending", "y_reverted")]  # wiersze 123–124


def _frame() -> pd.DataFrame:
    rows = [(f"message {cls} number {i}", cls) for cls in CLASSES for i in range(40)]
    rows += [(text, "a_cls") for text in DUPLICATES]  # wiersze 120–122
    rows += CONFLICT
    return pd.DataFrame(rows, columns=["text", "category"])


def _split(df: pd.DataFrame, seed: int = 42):
    return make_splits(df, val_size=0.15, random_state=seed)


def test_normalize_text_follows_config_rule():
    assert normalize_text("Hello,  WORLD!!") == "hello world"
    assert normalize_text("ｈｅｌｌｏ") == "hello"  # NFKC: pełna szerokość
    assert normalize_text("Straße") == "strasse"  # casefold, nie lower()
    assert normalize_text("Top-up pending!") == "top up pending"


def test_duplicates_stay_together_and_conflict_group_goes_to_train():
    df = _frame()
    result = _split(df)
    train, val = set(result.train_ids), set(result.val_ids)
    assert train.isdisjoint(val) and train | val == set(range(len(df)))
    duplicate_rows = {120, 121, 122}
    assert duplicate_rows <= train or duplicate_rows <= val
    assert result.conflict_groups == [
        {"rows": [123, 124], "labels": ["x_pending", "y_reverted"], "assigned_to": "train"}
    ]
    assert {123, 124} <= train
    assert result.n_groups == 122  # 120 unikalnych + grupa duplikatów + grupa konfliktowa


def test_split_is_stratified_and_close_to_val_size():
    df = _frame()
    result = _split(df)
    assert abs(len(result.val_ids) / len(df) - 0.15) < 0.03
    val_counts = df.loc[result.val_ids, "category"].value_counts()
    assert set(val_counts.index) == set(CLASSES)  # każda klasa w walidacji
    assert val_counts.min() >= 5  # ~40 * 0,15 = 6 na klasę


def test_split_is_deterministic_for_seed():
    df = _frame()
    assert _split(df, seed=42) == _split(df, seed=42)
    assert _split(df, seed=42).val_ids != _split(df, seed=7).val_ids


def test_empty_group_key_raises():
    df = pd.DataFrame({"text": ["???", "ok"], "category": ["a", "b"]})
    with pytest.raises(SplitError, match="pusty klucz"):
        make_splits(df, val_size=0.5, random_state=0)


def test_index_must_be_position_in_file():
    with pytest.raises(SplitError, match="indeks"):
        _split(_frame().iloc[::-1])


def test_ids_sha256_recipe_is_order_independent():
    expected = hashlib.sha256(b"[1,2,3]").hexdigest()  # HASH_RECIPE: zwarty JSON posortowanej listy
    assert ids_sha256([3, 1, 2]) == ids_sha256([1, 2, 3]) == expected


def test_write_then_verify_never_overwrites(tmp_path):
    df, settings = _frame(), load_settings()
    splits_dir, manifest_path = tmp_path / "splits", tmp_path / "split_manifest.json"
    kwargs = {"splits_dir": splits_dir, "manifest_path": manifest_path}
    result = _split(df)
    write_or_verify_split(result, build_manifest(result, df, settings), **kwargs)

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_ids = json.loads((splits_dir / "val_ids.json").read_text(encoding="utf-8"))
    assert saved["sha256_val_ids"] == ids_sha256(val_ids)
    assert saved["seed"] == 42 and saved["n_train"] + saved["n_val"] == len(df)

    report = write_or_verify_split(result, build_manifest(result, df, settings), **kwargs)
    assert all("zgodny" in line for line in report)  # ponowne uruchomienie = weryfikacja

    other = _split(df, seed=7)
    before = (splits_dir / "val_ids.json").read_text(encoding="utf-8")
    with pytest.raises(SplitError, match="nie nadpisuję"):
        write_or_verify_split(other, build_manifest(other, df, settings), **kwargs)
    assert (splits_dir / "val_ids.json").read_text(encoding="utf-8") == before


def test_verify_keeps_later_fields_and_only_reports_versions(tmp_path):
    df, settings = _frame(), load_settings()
    manifest_path = tmp_path / "split_manifest.json"
    kwargs = {"splits_dir": tmp_path / "splits", "manifest_path": manifest_path}
    result = _split(df)
    write_or_verify_split(result, build_manifest(result, df, settings), **kwargs)

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["test_subsets"] = {"n_clean": 3055}  # pole dopisane później (S2)
    saved["versions"]["pandas"] = "inna-wersja"
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")

    report = write_or_verify_split(result, build_manifest(result, df, settings), **kwargs)
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after["test_subsets"] == {"n_clean": 3055}
    assert any("versions" in line for line in report)
