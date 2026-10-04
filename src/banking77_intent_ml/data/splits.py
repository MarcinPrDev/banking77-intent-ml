"""Zamrożony podział train -> train/val z grupowaniem duplikatów (BANKING77, W1 S1).

Kod: asystent (Claude) na prośbę autora, 4.10.2026; decyzje projektowe — autor (4.10.2026):
1. Grupa = wiersze o tym samym kluczu po normalizacji z configu (`split.dedup_normalization`):
   Unicode NFKC -> casefold -> tokeny \\w+ połączone pojedynczą spacją. Cała grupa trafia do
   jednej części — wariant tego samego zdania nie leży po obu stronach granicy train/val.
2. Droga: jeden reprezentant na grupę (wiersz o najniższej pozycji) -> sklearn
   `train_test_split(stratify=etykieta, test_size=split.val_size,
   random_state=split.random_state)` -> rozwinięcie grup z powrotem do wierszy.
3. Grupy konfliktowe (jeden klucz, różne etykiety) trafiają do train jawną regułą przed
   losowaniem; nic nie jest usuwane (`n_excluded = 0`).
4. Pliki z indeksami (`dane/splits/*.json`, same pozycje wierszy) są w gicie.

Zamrożenie: `write_or_verify_split` nigdy nie nadpisuje — przy kolejnym uruchomieniu sprawdza,
że wynik jest identyczny z zapisanym (kontrola odtwarzalności), a różnicę zgłasza błędem.
Indeksy = pozycje wierszy w `train.csv` pod przypiętym commitem (indeks z `load_train`).
Pliki splitu i manifest nie zawierają tekstów — tylko pozycje, etykiety i liczby.
Uruchomienie: `uv run python scripts/make_splits.py`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from banking77_intent_ml.config import Settings

TRAIN_IDS_FILE = "train_ids.json"
VAL_IDS_FILE = "val_ids.json"
HASH_RECIPE = "sha256(UTF-8 json.dumps(sorted(ids), separators=(',', ':')))"
METHOD = (
    "reprezentant grupy = wiersz o najniższej pozycji; sklearn train_test_split("
    "stratify=etykieta, test_size=val_size, random_state=seed) na reprezentantach; "
    "grupy rozwinięte do wierszy"
)
CONFLICT_RULE = "grupy z różnymi etykietami -> train przed losowaniem (decyzja autora 4.10.2026)"
# Pola manifestu opisujące środowisko, nie wynik: różnica przy weryfikacji = informacja, nie błąd.
_INFORMATIONAL_KEYS = frozenset({"versions"})
_TOKEN = re.compile(r"\w+")


class SplitError(ValueError):
    """Podział niezgodny z kontraktem albo z zamrożonymi plikami."""


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[int]  # rosnące pozycje wierszy w train.csv
    val_ids: list[int]
    val_size: float  # parametry faktycznie użyte do losowania (trafiają do manifestu)
    random_state: int
    n_groups: int
    conflict_groups: list[dict]  # [{"rows": [...], "labels": [...], "assigned_to": "train"}]


def normalize_text(text: str) -> str:
    """Klucz grupy wg `split.dedup_normalization`: NFKC -> casefold -> tokeny \\w+ ze spacją."""
    return " ".join(_TOKEN.findall(unicodedata.normalize("NFKC", text).casefold()))


def ids_sha256(ids: Iterable[int]) -> str:
    """Hash listy pozycji wg `HASH_RECIPE` — kolejność na wejściu nie ma znaczenia."""
    payload = json.dumps(sorted(int(i) for i in ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_splits(train: pd.DataFrame, *, val_size: float, random_state: int) -> SplitResult:
    """Grupowany, stratyfikowany podział wierszy train na train/val (decyzje 1–3 wyżej)."""
    if not train.index.equals(pd.RangeIndex(len(train))):
        raise SplitError("indeks musi być pozycją wiersza w pliku (0..n-1) — użyj load_train()")
    keys = train["text"].map(normalize_text)
    if keys.eq("").any():
        positions = train.index[keys.eq("")].tolist()[:10]
        raise SplitError(f"teksty bez tokenów \\w+ (pusty klucz grupy), pozycje: {positions}")
    labels = train["category"]

    labels_per_key = labels.groupby(keys).nunique()
    conflict_keys = set(labels_per_key.index[labels_per_key > 1])
    in_conflict = keys.isin(conflict_keys)

    representatives = (
        pd.DataFrame({"key": keys, "label": labels})[~in_conflict]
        .drop_duplicates(subset="key", keep="first")  # pierwszy = najniższa pozycja w pliku
    )
    _, val_keys = train_test_split(
        representatives["key"],
        test_size=val_size,
        random_state=random_state,
        stratify=representatives["label"],
    )
    is_val = keys.isin(set(val_keys))

    # Bezpieczniki: z konstrukcji zawsze przechodzą — pilnują przyszłych zmian w kodzie wyżej.
    if set(keys[is_val]) & set(keys[~is_val]):
        raise SplitError("grupa duplikatów po obu stronach podziału")
    if in_conflict[is_val].any():
        raise SplitError("grupa konfliktowa w walidacji — reguła: zawsze train")

    conflict_groups = [
        {
            "rows": train.index[keys.eq(key)].tolist(),
            "labels": sorted(labels[keys.eq(key)].unique().tolist()),
            "assigned_to": "train",
        }
        for key in sorted(conflict_keys, key=lambda k: train.index[keys.eq(k)].min())
    ]
    return SplitResult(
        train_ids=train.index[~is_val].tolist(),
        val_ids=train.index[is_val].tolist(),
        val_size=val_size,
        random_state=random_state,
        n_groups=int(keys.nunique()),
        conflict_groups=conflict_groups,
    )


def _class_counts(labels: pd.Series) -> dict[str, int]:
    return {str(name): int(count) for name, count in labels.value_counts().sort_index().items()}


def build_manifest(result: SplitResult, train: pd.DataFrame, settings: Settings) -> dict:
    """Pola manifestu: klucze wymagane przez strażnika + audyt (bez tekstów i ścieżek maszyny)."""
    labels = train["category"]
    return {
        "seed": result.random_state,
        "val_size": result.val_size,
        "n_train": len(result.train_ids),
        "n_val": len(result.val_ids),
        "n_excluded": 0,
        "sha256_val_ids": ids_sha256(result.val_ids),
        "sha256_train_ids": ids_sha256(result.train_ids),
        "hash_recipe": HASH_RECIPE,
        "files": {
            "dir": "paths.splits_dir (config)",
            "train_ids": TRAIN_IDS_FILE,
            "val_ids": VAL_IDS_FILE,
        },
        "method": METHOD,
        "dedup_normalization": settings.split.dedup_normalization,
        "n_groups": result.n_groups,
        "n_duplicate_rows": len(train) - result.n_groups,
        "conflict_rule": CONFLICT_RULE,
        "conflict_groups": result.conflict_groups,
        "class_counts": {
            "train": _class_counts(labels.loc[result.train_ids]),
            "val": _class_counts(labels.loc[result.val_ids]),
        },
        "source": {
            "commit": settings.dataset.commit,
            "train_sha256": settings.dataset.train.sha256,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def _dump(obj: object, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(obj, separators=(",", ":")) + "\n"
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_or_verify_split(
    result: SplitResult, manifest: dict, *, splits_dir: Path, manifest_path: Path
) -> list[str]:
    """Zapisuje brakujące pliki splitu i pola manifestu; istniejące tylko weryfikuje.

    Faza 1 sprawdza wszystko, co już istnieje: różnica w indeksach albo w polach manifestu
    (poza informacyjnymi, np. `versions`) -> `SplitError`, zanim cokolwiek zostanie zapisane.
    Faza 2 zapisuje tylko brakujące pliki i pola — pola dopisane później (np. podzbiory
    testowe z S2) zostają nietknięte. Zwraca linie raportu.
    """
    report: list[str] = []
    to_write: dict[Path, str] = {}

    for name, ids in ((TRAIN_IDS_FILE, result.train_ids), (VAL_IDS_FILE, result.val_ids)):
        path = splits_dir / name
        if not path.exists():
            to_write[path] = _dump(ids, compact=True)
        elif _read_json(path) != ids:
            raise SplitError(f"{path}: zamrożone indeksy różnią się od wyniku — nie nadpisuję")
        else:
            report.append(f"{name}: zgodny z zamrożonym plikiem")

    existing = _read_json(manifest_path) if manifest_path.exists() else {}
    changed = [
        key
        for key, value in manifest.items()
        if key in existing and key not in _INFORMATIONAL_KEYS and existing[key] != value
    ]
    if changed:
        raise SplitError(f"{manifest_path}: pola różne od zamrożonych {changed} — nie nadpisuję")
    for key in sorted(_INFORMATIONAL_KEYS & existing.keys()):
        if existing[key] != manifest.get(key):
            report.append(f"{manifest_path.name}: '{key}' inne niż przy zamrożeniu (informacyjnie)")
    missing = {key: value for key, value in manifest.items() if key not in existing}
    if missing:
        to_write[manifest_path] = _dump({**existing, **missing})
    else:
        report.append(f"{manifest_path.name}: zgodny z zamrożonym plikiem")

    for path, text in to_write.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        report.append(f"{path.name}: zapisano")
    return report
