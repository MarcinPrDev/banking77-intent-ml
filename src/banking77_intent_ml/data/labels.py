"""Mapa etykiet BANKING77: nazwa kategorii <-> id (0..n_classes-1), jedna dla modeli, API i testów.

Kod: asystent (Claude) na prośbę autora, 4.10.2026; decyzja projektowa — autor (4.10.2026):
id = pozycja nazwy na liście `sorted()` unikalnych nazw kategorii z train (porządek kodów
Unicode, nie locale). Mapę da się odtworzyć z samych nazw; strażnik (`tests/test_contract.py`)
pilnuje kolejności, bijekcji 0..n_classes-1 i liczby klas w `configs/label_map.json`.
Uruchomienie: `uv run python scripts/make_label_map.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


class LabelMapError(ValueError):
    """Mapa etykiet niezgodna z danymi albo z zapisanym plikiem."""


def build_label_map(categories: Iterable[str]) -> dict[str, int]:
    """{nazwa: id} z unikalnych nazw kategorii; id wg `sorted()` (decyzja 4.10)."""
    return {name: idx for idx, name in enumerate(sorted(set(categories)))}


def load_label_map(path: Path) -> dict[str, int]:
    """Czyta `{nazwa: id}` z pliku JSON zapisanego przez `write_or_verify_label_map`."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LabelMapError(f"{path}: oczekiwano obiektu JSON {{nazwa: id}}")
    return {str(name): int(idx) for name, idx in raw.items()}


def write_or_verify_label_map(label_map: dict[str, int], path: Path) -> bool:
    """Zapisuje mapę, gdy pliku nie ma; gdy jest — sprawdza, że jest identyczna.

    Zwraca True (plik zapisany) albo False (istniał i się zgadza). Inna treść ->
    `LabelMapError`, plik zostaje bez zmian: od tych id zależą modele, API i macierze pomyłek.
    """
    if path.exists():
        if load_label_map(path) != label_map:
            raise LabelMapError(
                f"{path} różni się od mapy zbudowanej z train — nie nadpisuję "
                "(zmiana id rozjechałaby modele, API i macierze pomyłek)"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(label_map, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return True
