"""Smoke działającego API (W3 N3; W4: po deployu do kind) — cienkie CLI.

Uruchomienie:
  uv run python scripts/smoke_api.py --url http://localhost:8080   # sprawdza usługę
  uv run python scripts/smoke_api.py --record                      # nagrywa oczekiwane wyniki
Tryb sprawdzania czeka na `/health` (do `--timeout` s), porównuje wersję z `deploy.model_version`,
wysyła zdania kontrolne z `deploy.smoke_cases` i sprawdza 422 dla pustego tekstu; kod wyjścia 1
przy dowolnej niezgodności. `--record` liczy oczekiwane wyniki ocenionym artefaktem
(`deploy.artifact_dir`, lokalnie, CPU) — po zmianie modelu albo wersji.
Logika: src/banking77_intent_ml/api/smoke.py.
"""

from __future__ import annotations

import argparse
import json
import sys

from banking77_intent_ml.api.smoke import (
    http_request,
    record_cases,
    run_smoke,
    wait_until_healthy,
)
from banking77_intent_ml.config import load_settings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Smoke API BANKING77")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--timeout", type=float, default=120.0, help="czekanie na /health [s]")
    parser.add_argument("--record", action="store_true", help="nagraj oczekiwane wyniki")
    args = parser.parse_args()
    settings = load_settings()
    path = settings.deploy.smoke_cases

    if args.record:
        expected = record_cases(settings)
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
        for case in expected["cases"]:
            print(f"  {case['intent']:32s} score {case['score']:.3f} margines "
                  f"{case['margin']:.3f} | {case['text']}")
        print(f"Zapisano {len(expected['cases'])} przypadków "
              f"({expected['model_version']}) -> {path}")
        return

    expected = json.loads(path.read_text(encoding="utf-8"))
    request = http_request(args.url)
    try:
        wait_until_healthy(request, args.timeout)
    except TimeoutError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    failures = run_smoke(request, expected, settings)
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        sys.exit(1)
    print(f"[OK] {args.url}: /health, {len(expected['cases'])} zdań kontrolnych i 422 — "
          f"wersja {settings.deploy.model_version}")


if __name__ == "__main__":
    main()
