# CLAUDE.md — us-accidents-ml

Projekt zaliczeniowy kursu AI & ML Engineering (SAGES/Kodołamacz). **Plan-of-record: README.md**
(protokół ewaluacji, kontrakt predykcji, harmonogram W1–W8 z datami, core vs stretch).

## Zasady współpracy (ważne!)

- **Kod i decyzje pisze Marcin. Claude = pair-programmer/reviewer**: code review, debugowanie,
  tłumaczenie, naprowadzanie, odpytywanie. Nie pisz za autora części merytorycznych
  (EDA, feature engineering, modele, wnioski, ADR-y) — szkielety/infra/dokumentacja OK.
- Przed wysyłką do feedbacku (W8): „test obrony" — przepytaj autora z każdej decyzji projektu.
- Commity z asystą: stopka `Co-Authored-By`.

## Twarde reguły techniczne

- **Kontrakt predykcji**: cechy z `configs/config.yaml → forbidden_features`
  (`Distance(mi)`, `End_Time`+czas trwania, `End_Lat/Lng`) NIGDY nie wchodzą do modelu.
- Wszystkie ścieżki (tabular / tekst pełny i cenzurowany / LLM / fuzja) oceniane na
  **zamrożonym splicie** (`make_splits`, seed w configu); metryka główna **macro-F1**;
  LLM na stratyfikowanej podpróbce 1–2 tys. z test setu.
- Środowisko: **uv** (`uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`).
- GPU: RTX 5080 (Blackwell/sm_120) → PyTorch instalować z kół **cu128+** (jawny index-url).
- Dane: `dane/US_Accidents_March23.csv` (Kaggle, poza gitem). `Severity` = wpływ na ruch, nie obrażenia.

## Szerszy kontekst (lokalne ścieżki)

- Hub projektu (checklisty, powtórki per etap, draft DM):
  `P:\Obsidian\ObsidianSync\Prywatne\Kurs AI ML\02 Praca zaliczeniowa.md`
- Mapa wszystkich materiałów kursu: `...\Kurs AI ML\00 MAPA materiałów i źródeł.md`
- Archiwum materiałów per zjazd: `P:\ML-AI-Szkolenie\` (Z1–Z14, Repos, Datasets)
- Siostrzany ETL (Spark, CI/CD, Dataproc): https://github.com/MarcinPrDev/us-accidents-etl

Termin: 27.08.2026 (elastyczny — DM do M. Rybińskiego), wysyłka ≤ 24.08.
