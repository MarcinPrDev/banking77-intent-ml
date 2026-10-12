"""Mapa etykiet BANKING77: nazwa kategorii <-> id (0..n_classes-1), jedna dla modeli, API i testów.

Kod: asystent (Claude) na prośbę autora, 4.10.2026; decyzja projektowa — autor (4.10.2026):
id = pozycja nazwy na liście `sorted()` unikalnych nazw kategorii z train (porządek kodów
Unicode, nie locale). Mapę da się odtworzyć z samych nazw; strażnik (`tests/test_contract.py`)
pilnuje kolejności, bijekcji 0..n_classes-1 i liczby klas w `configs/label_map.json`.
Uruchomienie: `uv run python scripts/make_label_map.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banking77_intent_ml.config import Settings


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


@dataclass(frozen=True)
class LabelMap:
    """Jedna mapa nazwa <-> id dla modeli i API (README → Architektura), z `dataset.label_map_path`.

    Predyktory zwracają id (decyzja autora 13.10.2026); nazwy intencji mapuje ta klasa.
    """

    name_to_id: Mapping[str, int]

    @classmethod
    def from_settings(cls, settings: Settings) -> LabelMap:
        return cls(load_label_map(settings.dataset.label_map_path))

    @cached_property
    def _id_to_name(self) -> dict[int, str]:
        return {class_id: name for name, class_id in self.name_to_id.items()}

    def id(self, name: str) -> int:
        return self.name_to_id[name]

    def name(self, class_id: int) -> str:
        return self._id_to_name[int(class_id)]

    def __len__(self) -> int:
        return len(self.name_to_id)
