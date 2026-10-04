"""Zamrożony podział train/val -> dane/splits/*.json + configs/split_manifest.json (cienkie CLI).

Uruchomienie: uv run python scripts/make_splits.py
Pierwsze uruchomienie zapisuje pliki; każde kolejne sprawdza, że wynik jest identyczny
(kontrola odtwarzalności) i niczego nie nadpisuje. Logika: src/banking77_intent_ml/data/splits.py.
Kod CLI: asystent (Claude) na prośbę autora; decyzje podziału — autor (4.10.2026).
"""

from __future__ import annotations

import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.loader import DataValidationError, load_train
from banking77_intent_ml.data.splits import (
    SplitError,
    build_manifest,
    make_splits,
    write_or_verify_split,
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    settings = load_settings()
    try:
        train = load_train(settings)
        result = make_splits(
            train, val_size=settings.split.val_size, random_state=settings.split.random_state
        )
        manifest = build_manifest(result, train, settings)
        report = write_or_verify_split(
            result,
            manifest,
            splits_dir=settings.paths.splits_dir,
            manifest_path=settings.split.manifest_path,
        )
    except (DataValidationError, SplitError) as exc:
        print(f"[FAIL] MAKE SPLITS: {exc}")
        sys.exit(1)

    n_train, n_val = manifest["n_train"], manifest["n_val"]
    print(f"[OK] wiersze: {len(train)} | grupy: {manifest['n_groups']} "
          f"| nadmiarowe duplikaty: {manifest['n_duplicate_rows']}")
    print(f"     train: {n_train} | val: {n_val} ({n_val / (n_train + n_val):.2%}) | n_excluded: 0")
    for group in manifest["conflict_groups"]:
        rows, labels = group["rows"], group["labels"]
        print(f"     grupa konfliktowa -> train: wiersze {rows}, etykiety {labels}")
    print(f"     sha256_val_ids: {manifest['sha256_val_ids']}")
    for line in report:
        print(f"     {line}")


if __name__ == "__main__":
    main()
