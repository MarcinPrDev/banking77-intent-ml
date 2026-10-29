# CLAUDE.md — banking77-intent-ml (temat: BANKING77, od 30.09.2026)

Projekt zaliczeniowy kursu AI & ML Engineering (SAGES/Kodołamacz). **Plan-of-record: README.md**
(dane i protokół ewaluacji, modele i budżety, plan W1–W5 z datami, minimum vs stretch, wdrożenie).
Hierarchia dokumentów przy konflikcie: `README.md` > `CLAUDE.md` > `.claude/playbook/00-zasady.md`
> reszta playbooka; **liczby zawsze z `configs/config.yaml`**.
Repo i pakiet przemianowane 2.10.2026 (wcześniej `us-accidents-ml` / `us_accidents_ml`) — zapis: `docs/RENAME.md`.

## Zasady współpracy (ważne!)

- **Kod i decyzje merytoryczne pisze Marcin. Claude = pair-programmer/reviewer**: code review,
  debugowanie, tłumaczenie, naprowadzanie, odpytywanie. Nie pisz za autora części merytorycznych
  (EDA, `make_splits`, cechy, modele, ewaluacja, wnioski, ADR-y) — szkielety/infra/dokumentacja/
  konfiguracja OK. Protokół miękkiej ochrony ról (2026-07-05): ostrzeżenie → alternatywa →
  wykonanie tylko po wyraźnym potwierdzeniu.
- Rytuał sesji: `/start-etapu Wn` → praca → `/review-ml` przed review kodu ML → `/test-obrony`
  (min. raz na slot, obowiązkowo przed wysyłką) → `/koniec-etapu`.
- Commity z asystą: stopka `Co-Authored-By`.

## Twarde reguły techniczne

- **Test set jest nietykalny**: `dane/banking77/test.csv` służy TYLKO audytowi integralności
  i ocenie końcowej. Żadnego fitu (także TF-IDF/tokenizera), HPO, wyboru epoki/progu ani doboru
  promptu na teście. Kod w `src/` poza katalogami/modułami `evaluation`/`data`/`audit` nie odwołuje
  się do pliku testowego (`test.csv`, `dataset.test`, pliki pochodne 3 055/385) — pilnuje
  `tests/test_contract.py` (hook PostToolUse; odstępstwo: `# allow-testset` w tej samej linii).
- **Gdzie żyje kod**: logika treningu/HPO/ewaluacji/pomiarów w `src/banking77_intent_ml/{data,features,
  models,evaluation,api}`; w `scripts/` tylko cienkie CLI (`fetch_data`, `smoke_api`, `demo`) —
  strażnik skanuje wyłącznie `src/`, a hook patrzy tylko na `src|configs|tests`.
- **Zamrożony split walidacyjny**: indeksy w `dane/splits/` (deny na Edit/Write; nie regenerować
  „dla porządku") + manifest `configs/split_manifest.json` (git-tracked; `seed` i `val_size`
  = `split.random_state`, `split.val_size` z configu). Znormalizowane duplikaty razem w jednej
  części. Podzbiór 3 055 testów bez pokrycia i 385 id dla LLM — ustalone z góry, też w `dane/splits/`.
- **Przypięte dane i mapa etykiet**: commit `57ec275d…` + SHA-256 obu plików w
  `configs/config.yaml` (`scripts/fetch_data.py` egzekwuje); `configs/label_map.json` = jedyna
  mapa nazwa ↔ id dla wszystkich modeli i API. Nie mieszać z kopiami HF/MTEB. Wagi MiniLM
  i DistilBERT z przypiętą rewizją (`revision` w configu — zamienić `PIN-ME` przed pierwszym runem).
- **Metryki**: `accuracy` główna (porównanie z literaturą); **macro-F1 = metryka wyboru**
  checkpointu/konfiguracji na walidacji; recall/F1 per klasa, macierz pomyłek; koszt = czas
  treningu, rozmiar artefaktu, latencja CPU p50/p95 batch=1 (zimny vs rozgrzany).
  **Nie wymyślaj liczb** — każda z MLflow/wykonanego kodu albo „do zmierzenia".
- Środowisko: **uv**. Lokalnie JEDNA komenda (od 22.10.2026 z extras stretchów llm i ui):
  `uv sync --extra dev --extra nlp --extra gpu --extra serving --extra llm --extra ui`;
  goły `uv sync --extra dev` wyłącznie w CI. **`uv sync` jest
  „exact" — extras pominięte w komendzie są usuwane ze środowiska** (torch/transformers znikają);
  po każdym sync na GPU sprawdź `torch.cuda.is_available()`. `uv run` jest bezpieczny (inexact).
  Extras: `nlp` (sentence-transformers), `gpu` (torch cu128 + transformers), `llm` (groq + litellm),
  `serving` (fastapi, uvicorn, httpx), `ui` (streamlit), `cpu` (torch CPU — tylko obraz Docker,
  rozłączny z `gpu`).
- MLflow (od 6.10.2026: SQLite): `configure_mlflow(settings)` z `evaluation/tracking.py` na starcie
  każdego skryptu i notebooka — runy w `mlflow.db`, artefakty w `mlruns/`, oba jako bezwzględne
  ścieżki z configu. MLflow 3.14 odrzuca magazyn plikowy, a bez jawnego URI zapisuje do `mlflow.db`
  w cwd (run z `notebooks/` wylądowałby w drugim magazynie). UI z korzenia repo:
  `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- GPU: RTX 5080 (Blackwell/sm_120) → PyTorch tylko z kół **cu128+** (jawny index w pyproject).
  Problemy GPU mają osobny timebox i nie blokują ścieżek CPU.
- Dane poza gitem (`dane/` w .gitignore); w repo tylko kod, `fetch_data.py`, mapa etykiet,
  manifest splitu. Licencja danych CC BY 4.0 — atrybucja w README. Żadnych sekretów w repo
  (`GROQ_API_KEY` tylko w `.env`/GitHub Secrets).
- **Wykrycie istotnych PII w danych = stop i decyzja autora o rezygnacji ze zbioru.**

## Szerszy kontekst

- Hub projektu, notatki researchu (protokół i plan źródłowy, decyzje i korekty), mapa materiałów
  i archiwum kursu — notatki autora poza repo; ścieżki w `CLAUDE.local.md` (poza gitem).
- Siostrzany ETL (Spark, CI/CD, Dataproc): https://github.com/MarcinPrDev/us-accidents-etl

**Termin: 31.10.2026 (sobota) — wysyłka ≤ 31.10; 1.11 tylko poprawki po wysyłce.**
Bramka po W1: niedziela 4.10 wieczór (README → *Bramka po W1 i fallback*).
