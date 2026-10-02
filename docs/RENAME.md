# Zmiana nazwy repo i pakietu — checklista (stan 30.09.2026, decyzja NIEPODJĘTA)

Repo `us-accidents-ml` i pakiet `us_accidents_ml` noszą nazwę poprzedniego tematu (US Accidents).
Po pivocie na BANKING77 (30.09.2026) nazwa jest historyczna. **Czy i kiedy zmieniać — decyzja
Marcina.** Do tego czasu wszystkie dokumenty używają obecnych ścieżek.

**Kiedy najbezpieczniej:** na początku slotu, przed dużą porcją kodu (np. początek W2), albo
dopiero przy upublicznieniu repo (W5) — nigdy w środku weekendu z eksperymentami, bo rename
psuje jednocześnie hook, junction pamięci i importy.

**Propozycje nazwy** (do wyboru, nie rekomendacja): `banking77-intent-ml`, `intent-nlp-banking77`,
`banking77-classifier`. Pakiet Python = nazwa z podkreśleniami (`banking77_intent_ml`).
Nowa nazwa poniżej jako `<NOWA>` / `<nowa_pkg>`.

## Checklista (kolejność ma znaczenie)

Przed startem: `git status` czysty, wszystkie testy zielone, pamięć sesji zapisana (`/koniec-etapu`).

**Pułapka nr 1 — `.venv` zapamiętuje ścieżkę bezwzględną.** Instalacja edytowalna pakietu
(`.venv/Lib/site-packages/__editable__.us_accidents_ml-0.1.0.pth` zawiera `P:\Repos\us-accidents-ml\src`),
`pyvenv.cfg` i launchery w `.venv/Scripts` mają wpisaną starą ścieżkę. Po zmianie nazwy folderu import
pakietu przestaje działać, dopóki nie powtórzysz `uv sync`. **`uv sync` trzeba powtórzyć po KAŻDEJ
zmianie ścieżki folderu** (krok 2), nie tylko po zmianie nazwy pakietu (krok 1). Komenda środowiska
lokalnego zawsze z pełną listą extras (`uv sync` jest „exact" — pomija = usuwa):
`uv sync --extra dev --extra nlp --extra gpu` (od W2 także `--extra serving`).

1. **Kod i konfiguracja (jeden commit „Rename: us-accidents-ml -> <NOWA>")**
   - [ ] `pyproject.toml`: `name = "<NOWA>"`; `[tool.setuptools.packages.find]` bez zmian (szuka w `src/`)
   - [ ] `git mv src/us_accidents_ml src/<nowa_pkg>`
   - [ ] zamiana importów: `grep -rn "us_accidents_ml" src tests scripts notebooks .github` → `<nowa_pkg>`
     (m.in. `tests/test_smoke.py`, `.github/workflows/ci.yaml` → `--cov=<nowa_pkg>`)
   - [ ] `src/<nowa_pkg>/__init__.py`: docstring pakietu
   - [ ] `.gitignore`: komentarz o `/models/` wspomina `src/us_accidents_ml/models/` — zaktualizować
   - [ ] `README.md` (nagłówek, notka o nazwie), `CLAUDE.md` (tytuł), ten plik (status → wykonano)
   - [ ] `uv lock` (nazwa projektu w lockfile) → `uv sync --extra dev --extra nlp --extra gpu` →
     `uv run pytest` → `uv run ruff check .` — to sprawdza rename PAKIETU, folder jest jeszcze stary
   - [ ] usunąć stary `src/us_accidents_ml.egg-info/` (i tak w .gitignore)

2. **Folder lokalny i GitHub**
   - [ ] zamknąć wszystkie sesje Claude Code i IDE w `P:\Repos\us-accidents-ml`
   - [ ] GitHub → Settings → Rename repository → `<NOWA>` (GitHub przekierowuje stary URL, ale remote
     i tak zaktualizować)
   - [ ] lokalnie: `Rename-Item P:\Repos\us-accidents-ml P:\Repos\<NOWA>`
   - [ ] **ponownie `uv sync --extra dev --extra nlp --extra gpu`** w nowym folderze (instalacja
     edytowalna i `.venv` mają wpisaną starą ścieżkę bezwzględną — pułapka nr 1). Przy problemach:
     `Remove-Item -Recurse -Force .venv` → `uv sync ...` od zera → potem `uv run pytest`
   - [ ] `git remote set-url origin https://github.com/MarcinPrDev/<NOWA>.git`; `git fetch` jako test
   - [ ] w `README.md` sekcja *Dlaczego bez Sparka* linkuje `us-accidents-etl` — to INNE repo, nie zmieniać

3. **Setup współpracy z AI (poza gitem — łatwo zapomnieć)**
   - [ ] `.claude/settings.local.json` → hook PostToolUse: `$root="P:/Repos/us-accidents-ml"` →
     `P:/Repos/<NOWA>` (inaczej strażnik protokołu przestaje się uruchamiać, bez błędu)
   - [ ] `.claude/settings.local.json` → **komunikaty hooka** nadal z US Accidents: `statusMessage`
     „Strażnik kontraktu predykcji (…)" → „Strażnik protokołu ewaluacji (pytest tests/test_contract.py)";
     tekst błędu „KONTRAKT PREDYKCJI ZLAMANY (…)" → „PROTOKÓŁ EWALUACJI ZŁAMANY (tests/test_contract.py)".
     Tylko tekst — logika hooka i `$root` bez zmian. **Niezależnie od rename** — najlepiej już w W1
     (po czerwonym hooku Marcin szukałby `forbidden_features`). Plik poza gitem; 7.07 klasyfikator
     uprawnień odmówił asystentowi edycji tego pliku — robi Marcin (albo asystent, jeśli uprawnienia pozwolą)
   - [ ] `.claude/settings.local.json` → `additionalDirectories` bez zmian (Obsidian, archiwum)
   - [ ] **Junction pamięci**: katalog auto-pamięci Claude Code jest per cwd. Obecnie
     `C:\Users\mproc\.claude\projects\P--Repos-us-accidents-ml\memory` → junction do
     `C:\Users\mproc\.claude\projects\C--Users-mproc-OneDrive-Pulpit-AI-ML\memory`.
     Po zmianie ścieżki nowy cwd dostanie NOWY katalog projektu (`P--Repos-<NOWA>`) bez pamięci.
     Kroki: uruchomić raz `claude` w nowym folderze (utworzy katalog projektu) → zamknąć →
     usunąć pusty `...\P--Repos-<NOWA>\memory` → `cmd /c mklink /J "C:\Users\mproc\.claude\projects\P--Repos-<NOWA>\memory" "C:\Users\mproc\.claude\projects\C--Users-mproc-OneDrive-Pulpit-AI-ML\memory"`
     → sprawdzić `dir` (widoczny `MEMORY.md`). Stary junction `P--Repos-us-accidents-ml` można
     zostawić (martwy) albo usunąć `rmdir` (rmdir na junction NIE usuwa celu).
   - [ ] `CLAUDE.local.md` i `.claude/playbook/*.md`: `grep -rn "us-accidents-ml" .claude CLAUDE.local.md`
   - [ ] pamięć (`MEMORY.md` + pliki tematyczne): zaktualizować ścieżkę repo we wpisach

4. **Poza repo**
   - [ ] Obsidian: `02 Praca zaliczeniowa.md` (hub), `00 MAPA…`, `CLAUDE.md` w `Pulpit\AI ML`
     → `grep -rn "us-accidents-ml" "P:\Obsidian\ObsidianSync\Prywatne\Kurs AI ML"`
   - [ ] Google Calendar: opisy wydarzeń W1–W5 z linkiem/nazwą repo
   - [ ] CV / dossier kompetencji, jeśli już wpisano nazwę repo

5. **Weryfikacja końcowa**
   - [ ] nowa sesja Claude Code w `P:\Repos\<NOWA>`: `/start-etapu Wn` widzi pamięć i playbook
   - [ ] edycja dowolnego pliku w `src/` odpala hook (komunikat „Strażnik protokołu…" w statusie)
   - [ ] `uv run pytest` zielony (jeśli `ImportError`/`ModuleNotFoundError` → nie powtórzono `uv sync`
     po zmianie ścieżki, pułapka nr 1), CI na GitHubie zielone po pushu
   - [ ] `git log --oneline -3` — historia zachowana (rename nie tworzy nowego repo)

## Czego rename NIE dotyka

- Historia gita, ADR-y (odwołują się do tematu, nie nazwy), `docs/adr/0001…` (tam nazwa
  historyczna jest właśnie historią).
- Siostrzane repo `us-accidents-etl` — zostaje pod swoją nazwą.
- Dane w `dane/` (poza gitem) — przenoszą się razem z folderem.
