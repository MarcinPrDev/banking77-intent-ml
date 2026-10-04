"""Mapa etykiet z train -> configs/label_map.json (cienkie CLI: tworzy albo sprawdza istniejącą).

Uruchomienie: uv run python scripts/make_label_map.py
Logika: src/banking77_intent_ml/data/labels.py (id wg sorted() nazw — decyzja autora 4.10.2026).
Kod CLI: asystent (Claude) na prośbę autora.
"""

from __future__ import annotations

import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.labels import (
    LabelMapError,
    build_label_map,
    write_or_verify_label_map,
)
from banking77_intent_ml.data.loader import DataValidationError, load_train


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    path = settings.dataset.label_map_path
    try:
        label_map = build_label_map(load_train(settings)["category"])
        if len(label_map) != settings.dataset.n_classes:
            raise LabelMapError(
                f"{len(label_map)} kategorii w train, config: {settings.dataset.n_classes}"
            )
        created = write_or_verify_label_map(label_map, path)
    except (DataValidationError, LabelMapError) as exc:
        print(f"[FAIL] MAPA ETYKIET: {exc}")
        sys.exit(1)

    state = "zapisano" if created else "zgodna z istniejącym plikiem (nic nie zmieniono)"
    first, last = next(iter(label_map)), next(reversed(label_map))
    print(f"[OK] {path.name}: {len(label_map)} kategorii — {state}")
    print(f"     id 0 = {first!r} ... id {len(label_map) - 1} = {last!r}")


if __name__ == "__main__":
    main()
