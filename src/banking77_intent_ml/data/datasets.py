"""Teksty i etykiety z ZAMROŻONEGO splitu (BANKING77) — jedno źródło danych dla wszystkich modeli.

Indeksy z `dane/splits/` są przed użyciem sprawdzane hashem z `configs/split_manifest.json`:
zmieniony plik z indeksami = błąd, a nie cicha zmiana splitu. Etykiety jako id
z `configs/label_map.json` (decyzja autora 6.10: modele i `evaluate()` pracują na id).
Indeks serii = pozycja wiersza w train.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from banking77_intent_ml.config import Settings, load_settings
from banking77_intent_ml.data.labels import LabelMapError, load_label_map
from banking77_intent_ml.data.loader import load_train
from banking77_intent_ml.data.splits import TRAIN_IDS_FILE, VAL_IDS_FILE, SplitError, ids_sha256


@dataclass(frozen=True)
class TrainVal:
    x_train: pd.Series  # teksty części treningowej
    y_train: pd.Series  # id klas
    x_val: pd.Series
    y_val: pd.Series
    label_map: dict[str, int]


def _frozen_ids(settings: Settings) -> tuple[list[int], list[int]]:
    """Pozycje train/val z `dane/splits/`, sprawdzone hashem z manifestu."""
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    ids = []
    for name, key in ((TRAIN_IDS_FILE, "sha256_train_ids"), (VAL_IDS_FILE, "sha256_val_ids")):
        positions = json.loads((settings.paths.splits_dir / name).read_text(encoding="utf-8"))
        if ids_sha256(positions) != manifest[key]:
            raise SplitError(f"{name}: hash niezgodny z manifestem ({key}) — split zmieniony?")
        ids.append(positions)
    return ids[0], ids[1]


def full_train(data: TrainVal) -> tuple[pd.Series, pd.Series]:
    """Train + dawna walidacja (10 003) w kolejności wierszy train.csv — refit przed testem (W3).

    Decyzja autora 14.10.2026 (ADR-0002): refit zamrożonych konfiguracji na całym train; dawna
    walidacja przestaje służyć ocenie i strojeniu. Kolejność = pozycja w train.csv, niezależna
    od podziału.
    """
    x = pd.concat([data.x_train, data.x_val]).sort_index()
    y = pd.concat([data.y_train, data.y_val]).sort_index()
    return x, y


def load_train_val(settings: Settings | None = None) -> TrainVal:
    """Train → fit, val → ocena: teksty i id klas wg zamrożonego splitu i mapy etykiet."""
    s = settings if settings is not None else load_settings()
    train_ids, val_ids = _frozen_ids(s)
    train = load_train(s)
    label_map = load_label_map(s.dataset.label_map_path)
    labels = train["category"].map(label_map)
    if labels.isna().any():
        unknown = sorted(set(train.loc[labels.isna(), "category"]))
        raise LabelMapError(f"kategorie spoza mapy etykiet: {unknown[:10]}")
    labels = labels.astype(int)
    return TrainVal(
        x_train=train["text"].loc[train_ids],
        y_train=labels.loc[train_ids],
        x_val=train["text"].loc[val_ids],
        y_val=labels.loc[val_ids],
        label_map=label_map,
    )
