# Zmiana nazwy repo i pakietu — wykonana 2.10.2026

`us-accidents-ml` / `us_accidents_ml` → **`banking77-intent-ml` / `banking77_intent_ml`**.

**Powód:** po zmianie tematu projektu (ADR-0001) stara nazwa opisywała poprzedni zbiór danych.
**Moment:** przed pierwszą linią kodu BANKING77 — czyste drzewo, zielone CI, brak eksperymentów,
które trzeba by przenosić.

## Co zostało zmienione

| Element | Przed | Po |
|---|---|---|
| Repo na GitHubie | `MarcinPrDev/us-accidents-ml` | `MarcinPrDev/banking77-intent-ml` (stare adresy przekierowują) |
| Pakiet Pythona | `src/us_accidents_ml` | `src/banking77_intent_ml` |
| `pyproject.toml` → `name` | `us-accidents-ml` | `banking77-intent-ml` (+ `uv.lock`) |
| CI (`ci.yaml`) | `--cov=us_accidents_ml` | `--cov=banking77_intent_ml` |
| Folder lokalny autora | `P:\Repos\us-accidents-ml` | `P:\Repos\banking77-intent-ml` |

Poza gitem (lokalny setup pracy z asystentem AI): ścieżka `$root` i komunikaty hooka w
`.claude/settings.local.json`, junction pamięci Claude Code dla nowej ścieżki folderu, playbook,
skille, hub w Obsidianie.

## Czego zmiana nie dotyka

- Historia gita: commity do `60e628c` włącznie używają starej nazwy pakietu — to jest historia,
  nie błąd. ADR-0001 opisuje temat, nie nazwę repo.
- Siostrzane repo `us-accidents-etl` (Spark ETL, CI/CD) — zostaje pod swoją nazwą.
- Zmienna środowiskowa `BANKING77_CONFIG` i klucze `configs/config.yaml`.
- Dane (`dane/`, poza gitem) — przeniosły się razem z folderem.

## Dla starych klonów

```bash
git remote set-url origin https://github.com/MarcinPrDev/banking77-intent-ml.git
git pull
uv sync --extra dev --extra nlp --extra gpu --extra serving   # pakiet ma nową nazwę
```

## Pułapki, które wyszły przy zmianie (na przyszłość)

1. **`.venv` zapamiętuje ścieżkę bezwzględną** (launchery w `.venv/Scripts`, instalacja edytowalna
   pakietu). Po zmianie nazwy folderu środowisko trzeba odtworzyć:
   `uv sync --reinstall --extra dev --extra nlp --extra gpu --extra serving`. To pełna reinstalacja —
   z torchem cu128 trwa kilka minut.
2. **`uv sync` jest „exact"** — extras pominięte w komendzie są usuwane ze środowiska. Lokalnie
   zawsze pełna lista extras; goły `uv sync --extra dev` tylko w CI.
3. **Folder zablokowany przez edytor**: zmiana nazwy folderu wymaga zamknięcia VS Code i terminali
   z katalogiem roboczym w repo („Device or resource busy").
4. **Narzędzia z wpisaną ścieżką**: hook Claude Code (`$root`), katalog projektu Claude Code
   (osobny per ścieżka — pamięć wymaga nowego junction), skrypty pomocnicze poza repo.
5. **Katalog starego pakietu**: przenosić cały katalog (razem z `__pycache__`), a stary
   `*.egg-info` usunąć — inaczej w `src/` zostaje pusta wydmuszka o starej nazwie.
6. **Kolejność**: najpierw commit porządkowy i zielone CI, potem zmiana nazwy pakietu + `uv lock`
   + testy, zmiana nazwy na GitHubie (`gh repo rename` sam aktualizuje `origin`), na końcu folder
   i reinstalacja środowiska.
