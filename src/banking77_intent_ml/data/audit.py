"""Audyt integralności testu i podzbiory testowe ustalone z góry (BANKING77, W1 S2).

Kod: asystent (Claude) na prośbę autora, 5.10.2026; decyzje projektowe — autor (5.10.2026):
1. Czysty podzbiór = wiersze testu, których klucz po normalizacji z configu (ta sama funkcja
   `normalize_text` co w `splits.py`) NIE występuje w całym train.csv (10 003 wiersze, także
   część walidacyjna). Podzbiór jest jeden dla wszystkich modeli i zostaje czysty także przy
   refitie na 10 003. Liczność musi się zgadzać z `evaluation.n_clean_test_expected`.
2. Wiadomości pod LLM = `llm.per_class` na klasę z CZYSTEGO podzbioru (żadna nie ma odpowiednika
   w train — równe szanse LLM zero-shot i modeli dostrajanych); losowanie per klasa:
   `groupby(category).sample(n=per_class, random_state=split.random_state)`.
3. Klucze podzbiorów w manifeście są obowiązkowe (strażnik `tests/test_contract.py`).

Audyt integralności (jedyne dozwolone w W1 użycie testu): liczba wierszy, `n_rows / n_classes`
na klasę (test jest zrównoważony), zbiór nazw kategorii == mapa etykiet, pokrycie z train
i zgodność etykiet pokrytych tekstów. Kod przetwarza teksty testu, ale niczego z nich nie
wypisuje ani nie zapisuje — pliki i manifest zawierają tylko pozycje, liczby i nazwy kategorii.
Pliki zapisuje `write_or_verify_ids` (nigdy nie nadpisuje — kolejne uruchomienie weryfikuje).
Uruchomienie: `uv run python scripts/make_test_subsets.py [--dry-run]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.labels import load_label_map
from banking77_intent_ml.data.loader import load_test, load_train
from banking77_intent_ml.data.splits import ids_sha256, normalize_text, write_or_verify_ids

CLEAN_RULE = (
    "test bez odpowiednika w całym train.csv po normalizacji z configu "
    "(decyzja autora 5.10.2026)"
)
LLM_RULE = (
    "llm.per_class na klasę z czystego podzbioru; pandas groupby(category).sample("
    "n=per_class, random_state=split.random_state) (decyzja autora 5.10.2026)"
)


class AuditError(ValueError):
    """Test niezgodny z oczekiwaniami protokołu — stop przed jakimkolwiek treningiem."""


@dataclass(frozen=True)
class EvalSubsets:
    clean_ids: list[int]  # rosnące pozycje wierszy w test.csv
    llm_ids: list[int]
    per_class: int  # parametry faktycznie użyte do losowania (trafiają do manifestu)
    random_state: int
    n_overlap: int  # teksty testu z odpowiednikiem w train po normalizacji
    n_overlap_label_mismatch: int  # ...których etykiety nie ma wśród etykiet tego klucza w train


def _require_positional_index(frame: pd.DataFrame, name: str) -> None:
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise AuditError(f"indeks {name} musi być pozycją wiersza w pliku — użyj loadera")


def audit_test_integrity(
    test: pd.DataFrame, label_map: dict[str, int], *, n_classes: int
) -> dict:
    """Nazwy kategorii testu == mapa etykiet i test zrównoważony (n_rows / n_classes na klasę)."""
    names_test, names_map = set(test["category"]), set(label_map)
    if names_test != names_map:
        raise AuditError(
            f"nazwy kategorii testu != mapa etykiet: tylko w teście "
            f"{sorted(names_test - names_map)}, tylko w mapie {sorted(names_map - names_test)}"
        )
    if len(test) % n_classes:
        raise AuditError(f"{len(test)} wierszy testu nie dzieli się po równo na {n_classes} klas")
    per_class = len(test) // n_classes
    counts = test["category"].value_counts()
    uneven = counts[counts != per_class]
    if not uneven.empty:
        raise AuditError(
            f"test niezrównoważony — oczekiwano {per_class} na klasę, inne: {uneven.to_dict()}"
        )
    return {
        "n_rows": len(test),
        "n_classes": n_classes,
        "rows_per_class": per_class,
        "names_match_label_map": True,
    }


def make_eval_subsets(
    test: pd.DataFrame, train: pd.DataFrame, *, per_class: int, random_state: int
) -> EvalSubsets:
    """Czysty podzbiór (decyzja 1) i próbka pod LLM z czystego podzbioru (decyzja 2)."""
    _require_positional_index(train, "train")
    _require_positional_index(test, "test")
    train_keys = train["text"].map(normalize_text)
    test_keys = test["text"].map(normalize_text)
    overlap = test_keys.isin(set(train_keys))

    train_labels_by_key = train.groupby(train_keys)["category"].agg(lambda labels: set(labels))
    mismatch = sum(
        label not in train_labels_by_key[key]
        for key, label in zip(test_keys[overlap], test["category"][overlap], strict=True)
    )

    clean = test[~overlap]
    clean_counts = clean["category"].value_counts().reindex(sorted(set(test["category"])))
    too_small = clean_counts[clean_counts.fillna(0) < per_class]
    if not too_small.empty:
        raise AuditError(
            f"za mało czystych tekstów na {per_class} na klasę: {too_small.fillna(0).to_dict()}"
        )
    llm = clean.groupby("category").sample(n=per_class, random_state=random_state)
    return EvalSubsets(
        clean_ids=clean.index.tolist(),
        llm_ids=sorted(llm.index.tolist()),
        per_class=per_class,
        random_state=random_state,
        n_overlap=int(overlap.sum()),
        n_overlap_label_mismatch=int(mismatch),
    )


def build_subsets_manifest(subsets: EvalSubsets, audit: dict, settings: Settings) -> dict:
    """Pola manifestu S2: klucze wymagane przez strażnika + opis reguł i wynik audytu."""
    return {
        "n_test_clean": len(subsets.clean_ids),
        "sha256_test_clean_ids": ids_sha256(subsets.clean_ids),
        "n_llm_eval": len(subsets.llm_ids),
        "sha256_llm_eval_ids": ids_sha256(subsets.llm_ids),
        "test_subsets": {
            "files": {
                "dir": "paths.splits_dir (config)",
                "test_clean": settings.evaluation.clean_test_subset_path.name,
                "llm_eval": settings.llm.eval_ids_path.name,
            },
            "clean_rule": CLEAN_RULE,
            "llm_rule": LLM_RULE,
            "llm_per_class": subsets.per_class,
            "llm_seed": subsets.random_state,
            "n_overlap_with_train": subsets.n_overlap,
            "n_overlap_label_mismatch": subsets.n_overlap_label_mismatch,
        },
        "test_audit": audit,
    }


def run_test_audit(
    settings: Settings, *, dry_run: bool = False
) -> tuple[EvalSubsets, dict, list[str]]:
    """S2 od początku do końca: wczytanie, audyt, podzbiory, zapis albo weryfikacja plików.

    `dry_run=True` liczy i sprawdza wszystko, ale niczego nie zapisuje.
    """
    if not settings.split.manifest_path.exists():
        raise AuditError("brak split_manifest.json — najpierw scripts/make_splits.py")
    if not settings.dataset.label_map_path.exists():
        raise AuditError("brak label_map.json — najpierw scripts/make_label_map.py")
    llm, n_classes = settings.llm, settings.dataset.n_classes
    if llm.n_eval != llm.per_class * n_classes:
        raise AuditError(f"llm.n_eval={llm.n_eval} != {llm.per_class} na klasę × {n_classes}")

    train, test = load_train(settings), load_test(settings)
    audit = audit_test_integrity(
        test, load_label_map(settings.dataset.label_map_path), n_classes=n_classes
    )
    subsets = make_eval_subsets(
        test, train, per_class=llm.per_class, random_state=settings.split.random_state
    )
    expected = settings.evaluation.n_clean_test_expected
    if len(subsets.clean_ids) != expected:
        raise AuditError(
            f"czysty podzbiór ma {len(subsets.clean_ids)} wierszy, config oczekuje {expected} "
            "— stop: zbadaj przed jakimkolwiek treningiem"
        )

    fields = build_subsets_manifest(subsets, audit, settings)
    if dry_run:
        return subsets, fields, ["dry run: niczego nie zapisano"]
    id_files = {
        settings.evaluation.clean_test_subset_path: subsets.clean_ids,
        settings.llm.eval_ids_path: subsets.llm_ids,
    }
    report = write_or_verify_ids(id_files, fields, manifest_path=settings.split.manifest_path)
    return subsets, fields, report
