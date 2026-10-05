"""Podzbiory testowe ustalone z góry (W1 S2) -> dane/splits/ + split_manifest.json (cienkie CLI).

Uruchomienie: uv run python scripts/make_test_subsets.py [--dry-run]
Pierwsze uruchomienie zapisuje pliki i dopisuje pola manifestu; kolejne tylko weryfikuje (nic nie
nadpisuje). Wypisuje wyłącznie liczby i hashe — nigdy treści testu.
Logika: src/banking77_intent_ml/data/audit.py. Kod CLI: asystent (Claude) na prośbę autora;
decyzje — autor (5.10.2026).
"""

from __future__ import annotations

import argparse
import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.audit import AuditError, run_test_audit
from banking77_intent_ml.data.labels import LabelMapError
from banking77_intent_ml.data.loader import DataValidationError
from banking77_intent_ml.data.splits import SplitError


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Podzbiory testowe BANKING77 (W1 S2).")
    parser.add_argument("--dry-run", action="store_true", help="policz i sprawdź, nic nie zapisuj")
    args = parser.parse_args()
    try:
        _, fields, report = run_test_audit(load_settings(), dry_run=args.dry_run)
    except (AuditError, DataValidationError, LabelMapError, SplitError) as exc:
        print(f"[FAIL] PODZBIORY TESTOWE: {exc}")
        sys.exit(1)

    audit, info = fields["test_audit"], fields["test_subsets"]
    print(f"[OK] test: {audit['n_rows']} wierszy, {audit['rows_per_class']} na klasę, "
          f"{audit['n_classes']} kategorii == mapa etykiet")
    print(f"     pokrycie z train: {info['n_overlap_with_train']} "
          f"(z inną etykietą niż w train: {info['n_overlap_label_mismatch']})")
    print(f"     czysty podzbiór: {fields['n_test_clean']} | "
          f"sha256 {fields['sha256_test_clean_ids']}")
    print(f"     pod LLM: {fields['n_llm_eval']} ({info['llm_per_class']} na klasę, "
          f"seed {info['llm_seed']}) | sha256 {fields['sha256_llm_eval_ids']}")
    for line in report:
        print(f"     {line}")


if __name__ == "__main__":
    main()
