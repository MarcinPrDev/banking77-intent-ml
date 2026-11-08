> Przeniesione z README 8.11.2026 (redakcja po feedbacku prowadzącego); treść bez zmian.

# Przebieg projektu: plan, bramka, status i księga godzin

## Plan — 5 slotów, 60 h (30.09 → 31.10.2026)

*Plan z 30.09.2026; przebieg faktyczny i księga godzin — [Status](#status).*

Podział pracy: dane/protokół/EDA **10 h** · modele/HPO **12 h** · ewaluacja i analiza błędów **6 h**
· pakiet/API/testy **8 h** · Docker/kind/CI **9 h** · raport/demo **7 h** · rezerwa **8 h**.
Godziny slotu obejmują weekend i wieczory danego tygodnia — od 5.10.2026 każdy wieczór pn–pt
(także piątek) jest roboczy (~1–1,5 h, jedno małe zadanie; rozkład dzienny w playbookach slotów
i w kalendarzu), a godziny ponad budżet slotu idą do księgi rezerwy. Nazwy slotów W1–W5 są
używane przez skille (`/start-etapu Wn`) i kalendarz.

| Slot | Termin | h | Wymagany rezultat (definicja „done") |
|---|---|---:|---|
| **W1** | pt 2.10 (wieczór) + **3–4.10** | 12 | dane pobrane i zweryfikowane; krótka EDA + audyt jakości (duplikaty, konflikty, długości, PII); mapa etykiet; **zamrożona walidacja** (grupowany, stratyfikowany split + manifest); TF-IDF baseline w MLflow; predykcje przeżywają zapis/odczyt; krótki test uruchomienia DistilBERT na GPU |
| **W2** | **10–11.10** | 14 | trzy ścieżki (TF-IDF+LR, MiniLM+kNN, DistilBERT) z ograniczonym HPO na walidacji; pierwsze pomiary CPU (latencja, rozmiar); szkic interfejsu predyktora + Factory; **FastAPI z baseline'em** (`/predict`, `/health`, wersja modelu) |
| **W3** | **17–18.10** | 14 | **zamrożenie eksperymentów**; ocena końcowa na teście + podzbiór 3 055; analiza błędów; wybrany artefakt wersjonowany; testy jednostkowe (mapa etykiet, walidacja wejścia, zapis→odczyt, kontrakt API); Dockerfile (obraz CPU). **Bez nowych rodzin modeli** |
| **W4** | nd **18.10** + wieczory **19–22.10** (23–25.10 bez pracy) | 14 | **kind + automatyczny workflow** (build → load → Deployment/Service → test HTTP) w GitHub Actions; próba odtworzenia projektu z instrukcji na czystym klonie; wykorzystanie rezerwy; stretch po rdzeniu, do 22.10 |
| **W5** | **31.10 (sobota)** | 6 | raport (README z wynikami i wnioskami), demo, kontrola reprodukowalności, **oddanie**. Bez nowych funkcji |
| | **Razem** | **60** | **Wdrożenie działa przed ostatnim dniem; 1.11 tylko poprawki po wysyłce** |

**Księga godzin:** 60 h = 52 h pracy + 8 h rezerwy. Rezerwa leży w W4 (ogony W1–W3, próba
odtworzenia, debug workflow). Wieczory 26–30.10 (test obrony, gitleaks, wnioski — do 4,5 h) są
**poza** 60 h: pokrywa je niewykorzystana rezerwa albo świadome nadgodziny. Wieczory
30.09–1.10 pochłonęła decyzja o zmianie tematu (research + weryfikacje) — poza budżetem W1.
Stan rezerwy: wiersz w *Status*, aktualizowany na koniec każdego slotu.

**Dokumentacja przyrostowo:** koniec slotu = aktualizacja *Status* w README + wiersz w tabeli
wyników (accuracy, macro-F1, koszt) + 1–2 zdania „co nie zadziałało". ADR-y (`docs/adr/`, MADR)
pisane w momencie decyzji.

## Minimum vs stretch

*Stan na 29.10: minimum zrealizowane; stretch — seedy, LLM zero-shot, UI i GKE — zrobione w W4
(szczegóły w [Status](#status)).*

**MINIMUM (zalicza rubrykę):** EDA z audytem jakości · przypięte dane + zamrożony protokół ·
Dummy / TF-IDF+LR / MiniLM+kNN / DistilBERT z ograniczonym HPO w MLflow · porównanie jakości
i kosztu · analiza błędów · pakiet OOP (Strategy/Factory/pipeline'y) + pytest · FastAPI ·
Docker (CPU) · **kind z automatycznym deployem i testem HTTP w CI** · README z wnioskami
i instrukcją odtworzenia.

**STRETCH — w tej kolejności, dopiero po domknięciu minimum:**
1. dodatkowe seedy zamrożonej konfiguracji DistilBERT, mierzone **na walidacji** (rozrzut
   zamiast jednej liczby; test nietknięty — decyzja autora 18.10);
2. **LLM zero-shot (wariant A)** — Groq przez LiteLLM na 385 wiadomościach, protokół jak wyżej;
3. prosty UI (Streamlit) nad API;
4. GCP: GKE tym samym obrazem (klaster żyje godziny), Artifact Registry.

Rezerwa 8 h **nie jest** z góry budżetem na stretch.

**Plan stretchu (decyzja autora 18.10):** wszystkie cztery punkty w tej kolejności, każdy po
rdzeniu danego wieczoru W4, z granicą w czwartek 22.10 (W5 bez nowych funkcji; pt–nd 23–25.10
bez pracy). Co nie zdąży, zostaje opisane w README jako niezrobione.

## Bramka po W1 i fallback

*Wynik: bramka niezaliczona 4.10, zielona przy powtórce 9.10; fallback LinearSVC niepotrzebny
(DistilBERT działał na GPU od W1).*

**Bramka (powtórka: piątek 9.10, pierwotnie niedziela 4.10) — kontynuujemy, jeśli:** dane i grupowany podział są
odtwarzalne (fetch → make_splits → identyczny manifest), baseline uczy się sensownie ponad Dummy
na tej samej walidacji, predykcje przeżywają zapis/odczyt, i wiadomo, gdzie pójdzie czas wdrożenia.
**Nie uzależniamy decyzji od 92%.** Jeśli po 3–4 h diagnostyki baseline nie działa → naprawić
loader / mapę etykiet / pipeline **przed** kolejnymi modelami. Problemy GPU mają osobny timebox
i nie blokują ścieżek CPU.

**Stan na 9.10:** warunki techniczne są spełnione — dane i zamrożony podział są odtwarzalne,
TF-IDF+LR osiąga 84,39% accuracy wobec 1,87% Dummy `most_frequent` na walidacji, a zapis i odczyt
modelu zachowują identyczne predykcje. **Decyzja autora (pt 9.10): kontynuujemy — W2 od 10.10.**
Czas wdrożenia: autor ocenia plan W2–W4 jako wykonalny przy większym dziennym wymiarze pracy;
godziny ponad plan slotów trafiają do księgi rezerwy (*Status*).

**Fallback:** przy problemach z transformerem trzecią metodą zostaje **TF-IDF + LinearSVC**
(mniej powtórki DL, rdzeń rubryki zachowany). Przy budżecie bliższym 50 h — od początku tnij HPO
i UI, nie wdrożenie. Poślizg o 1 slot w W1–W3 → tnij stretch/HPO i dokładaj wieczory tak, by W4
zaczął się z działającym obrazem Docker; W4–W5 są nietykalne (wdrożenie = twardy wymóg rubryki) —
jeśli na starcie W4 nie ma obrazu, wdrażamy baseline TF-IDF, nie czekamy na lepszy model.

## Status

- [x] Szkielet projektu (pakiet, config, CI, uv) — przeniesiony z poprzedniego tematu
- [x] Smoke testy: GPU (torch cu128, sm_120, `scripts/smoke_gpu.py`), klucz Groq przez LiteLLM
      (`scripts/smoke_llm.py`) — infrastruktura sprawdzona przed pivotem
- [x] **30.09:** decyzja o zmianie tematu na BANKING77 · **2.10:** nowy plan-of-record, config,
      strażnik protokołu (`tests/test_contract.py`) i `scripts/fetch_data.py` w repo
- [x] **ADR-0001 — zmiana tematu na BANKING77**: zaakceptowany 9.10 (kontekst i opcje zestawione
      z notatek researchu; sekcje *Decyzja* i *Konsekwencje* — autor)
- [x] **[ADR-0002 — protokół ewaluacji](../adr/0002-protokol-ewaluacji.md)** — zaakceptowany 14.10:
      grupowany split, grupa konfliktowa, mapa etykiet, podzbiory testowe i refit na 10 003.
- [x] **W1** (pt 2.10 wieczór + weekend 3–4.10; twarde minimum domknięte w wieczory 5–9.10):
  - [x] dane pobrane i zweryfikowane (`scripts/fetch_data.py`: SHA-256, 10 003 / 3 080 wierszy)
  - [x] loader z walidacją struktury (`data/loader.py`) + testy na sztucznych CSV
  - [x] mapa etykiet `configs/label_map.json` (77 kategorii, id wg `sorted()` nazw)
  - [x] zamrożony split train/val (`scripts/make_splits.py`): 8 504 / 1 499, 9 972 grupy
    duplikatów (31 nadmiarowych), grupa konfliktowa w train; indeksy w `dane/splits/` (w gicie)
    + `configs/split_manifest.json`; ponowne uruchomienie tylko weryfikuje („zgodny”)
  - [x] strażnik protokołu zielony na realnych plikach (zakres rozszerzony 4.10)
  - [x] podzbiory testowe 3 055 / 385 + audyt nazw kategorii testu (S2, 5.10; pliki w
    `dane/splits/` tylko do odczytu)
  - [x] wspólna `evaluate()` + Dummy ×2 (6.10) i TF-IDF+LR (7.10) w MLflow — wyniki w tabeli niżej
  - [x] zapis/odczyt modelu (`models/persistence.py`, 9.10): joblib + metadane (wersje bibliotek,
    hash mapy etykiet, id runu); identyczne predykcje w nowym procesie, udowodnione testem
  - [x] krótka EDA z wnioskami (`notebooks/01_eda.ipynb`, sekcje 1–8, 6.10); długości w tokenach
    → sob 10.10, redakcja wniosków → wt 13.10
  - [x] rewizje wag przypięte w configu (9.10; MiniLM i DistilBERT, licencje apache-2.0) + test
    uruchomienia DistilBERT na GPU (`scripts/smoke_gpu.py`, 9.10): działa — mini-epoka 10 kroków
    × batch 32 na zdaniach z train, bf16, szczyt VRAM 1,40 GB z 17,1 GB; bez metryk
  - Bramka po W1: 4.10 nie przeszła (brak baseline'u w MLflow i testu zapisu/odczytu), warunkowo
    kontynuowano; **powtórka pt 9.10: zielona — kontynuujemy, W2 od 10.10** (szczegóły:
    *Bramka po W1 i fallback*)
  - Co nie zadziałało: twarde minimum nie zmieściło się w weekendzie 3–4.10 — baseline
    w MLflow i zapis/odczyt przechodzą na wieczory 5–9.10 (pierwsze do cięcia: DistilBERT, EDA)
- [x] **2.10 — zależności pod BANKING77**: z zależności głównych usunięte `xgboost`, `duckdb`,
      `pygeohash`; dodany extra `serving` (`fastapi`, `uvicorn[standard]`, `httpx` — wymagany przez
      `TestClient`); `optuna` zostaje do decyzji przy HPO w W2 (Optuna czy siatka) — rozstrzygnięte
      w W2: Optuna (TPE) dla TF-IDF+LR, siatka dla kNN i DistilBERT
- [x] **W2** (plan 10–11.10, realizacja 12–14.10): trzy ścieżki + HPO + pomiary CPU + API
  z baseline'em — **zamknięty 14.10, bramka zielona**; ADR-0002 i decyzja o reficie zapisane 14.10 (W3)
  - [x] HPO TF-IDF+LR (12.10): Optuna TPE, 16 prób na walidacji, wybór po macro-F1 (remis → mniej
    cech); zamrożona próba #12 w `models.tfidf_lr.frozen`
  - [x] MiniLM+kNN (12.10): embeddingi z przypiętej rewizji (cache), siatka k {1, 3, 5, 10} ×
    {uniform, distance} na walidacji; zamrożone k = 1, `uniform` w `models.minilm_knn.frozen`
  - [x] DistilBERT (13.10): 5 LR {2–6}e-5 × 5 epok (budżet podniesiony z 2–3 przed wynikami),
    ocena co epokę, wybór po macro-F1 (remis → mniej epok); zamrożony wariant 4, epoka 4
    w `models.distilbert.frozen`, artefakt w `models/distilbert/`; zwycięzca na brzegu siatki LR
  - [x] Wspólny interfejs predyktora (13.10): `IntentClassifier` + fabryka z configu + trzy
    adaptery (kod autora); test kontraktu zielony dla trzech ścieżek; każdy adapter odtwarza
    zamrożony run na walidacji co do bitu; artefakty `models/tfidf_lr/` i `models/minilm_knn/`
    (`scripts/export_model.py`, trening na train) oraz `models/distilbert/` z zamrożenia
  - [x] FastAPI z baseline'em (13.10): `POST /predict` (top-1: intencja, id, `score`, wersja
    modelu), `GET /health`, 422 dla pustego lub za długiego tekstu (> 500 znaków), predyktor
    ładowany raz (lifespan); serwuje `api.model` (TF-IDF+LR); lokalnie port 8080 (8000
    zarezerwowany przez Windows)
  - [x] Pomiary kosztu CPU (14.10): latencja batch=1 p50/p95 przy 1 i 8 wątkach, zimny start,
    rozmiar artefaktu — tabela *Koszt na CPU* niżej
  - Co nie zadziałało: weekend 10–11.10 bez pracy, więc W2 przesunął się na 12–14.10.
    Czas zabrały też problemy środowiska: czerwone CI po imporcie `sentence-transformers`,
    brak `accelerate`, zmiany API w `transformers` 5 i puste `CUDA_VISIBLE_DEVICES` na Windows.
- [x] **W3** (plan 12–18.10, realizacja 14–18.10; zamknięty 18.10): zamrożenie, ocena końcowa,
  analiza błędów, artefakt, testy, Docker
  - [x] Zamrożenie (14.10): ADR-0002 (protokół z W1 + refit przed testem), tag
    `experiments-frozen` na `d809c57`
  - [x] Refit na 10 003 (14.10): runy `final_refit`, artefakty w `models/final/`; koszt
    finalnych artefaktów — badanie `cpu_cost_final`
  - [x] Ocena końcowa na teście — raz (14.10): tabela końcowa i interpretacja niżej
  - [x] Analiza błędów (15.10): notebook 02, wnioski w sekcji *Analiza błędów (W3)*
  - [x] ADR-0003 (15.10): wdrażamy DistilBERT; model w CI z GitHub Release (opcja a); sekcja
    `deploy` w configu (`distilbert-v1`, `models/final/distilbert`)
  - [x] Obraz Docker CPU (15.10): `banking77-intent-api:distilbert-v1`, **3,18 GB** (`.venv`
    1,9 GB — torch CPU 750 MB, pyarrow 153 MB, scipy 113 MB, transformers 112 MB; model 257 MiB);
    torch 2.11.0+cpu (`torch.version.cuda` = None, brak pakietów `nvidia-*`); model ładuje się
    bez sieci; gotowość 3,5 s od `docker run`; RAM 391 MiB; test przeniesienia: te same id
    w 50/50, |Δ score| ≤ 3e-07. Zależności główne (mlflow, optuna, matplotlib, seaborn,
    pyarrow) zostają w obrazie — decyzja autora 15.10, rozmiar zmierzony
  - [x] `scripts/smoke_api.py` (15.10): `/health` z wersją z configu, 14 zdań kontrolnych
    (napisanych na potrzeby testu, nie ze zbioru) z oczekiwanymi wynikami ocenionego artefaktu
    (`configs/smoke_cases.json`), 422 dla pustego tekstu; wobec kontenera: [OK]. Test
    przeniesienia na dawnej walidacji wykonany jednorazowo (te same id w 50/50)
  - [x] Testy W3 §1.5 (16.10): mapa etykiet (`test_labels`, strażnik w `test_contract`),
    walidacja wejścia API — 422 (`test_api`), zapis i odczyt dla trzech adapterów z identycznymi
    predykcjami (`test_predictor`, test kontraktu), API przez TestClient, smoke (`test_smoke_api`);
    strażnik protokołu 0,06 s. Zgodność wersji: artefakty sklearn — odczyt odmawia przy innej
    wersji sklearn (`test_persistence`); DistilBERT — obraz instaluje z tego samego `uv.lock`
    (transformers 5.13.0, torch 2.11.0), a smoke porównuje wyniki zdań kontrolnych z ocenionym
    artefaktem (decyzja autora 16.10: bez osobnej kontroli wersji w `load`)
  - [x] Rozgrzewka DistilBERT (18.10): `load` wykonuje predykcję na syntetycznym tekście przed
    gotowością API. Pomiary kontenera z 15.10 dotyczą wersji sprzed tej zmiany.
  - Co nie zadziałało: notebook wymagał poprawy typów komórek i uzupełnienia obliczeń dla
    podanych liczb. Obraz CPU pozostał duży (3,18 GB), bo zachowano zależności treningu i analizy.
- [x] **W4** (start nd 18.10, wieczory 19–22.10; pt–nd 23–25.10 bez pracy — decyzja autora 18.10;
  zamknięty 26.10): kind + workflow CI + test HTTP + odtworzenie z instrukcji
  - [x] Runner CI przypięty do `ubuntu-24.04` (18.10; `ubuntu-latest` przechodzi na 26.04 od 19.10)
  - [x] Obraz przebudowany z rozgrzewką DistilBERT (18.10): build 94 s z cache zależności;
    gotowość 3,96 s od `docker run`, pierwsze `/predict` 15,5 ms (kolejne 13,4–14,6 ms),
    RAM 382 MiB, smoke [OK]
  - [x] Manifesty `k8s/` + test lokalny na kind v0.33.0 / Kubernetes v1.37.0 (18.10): decyzje
    autora — 2 repliki, requests 500m CPU / 512Mi, limit pamięci 1Gi bez limitu CPU, Service
    NodePort; utworzenie klastra 54 s, `kind load` 54 s, rollout 4 s, smoke [OK] przez NodePort,
    ruch na obie repliki (19/16 żądań), pamięć podów 381/377 MiB, bez restartów
  - [x] Model w GitHub Release (18.10; ADR-0003, opcja a): Release `distilbert-v1`, asset
    268 789 760 B, SHA-256 w `deploy.release_sha256`; `scripts/model_release.py pack/unpack`
    (manifest: sumy plików, mapa etykiet, wersje bibliotek, runy refit `5b883a97` i ocena
    `fb0d0bde`); pobranie z GitHuba i weryfikacja: 5 plików bajt w bajt zgodnych z ocenionym
    artefaktem. Użycie w workflow — w `deploy-kind.yaml`
  - [x] Workflow `deploy-kind.yaml` (18.10): zielony za pierwszym razem na PR #1 i na `main` —
    [run 39138227351](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39138227351),
    6 min 28 s. Kroki (run na PR): zwolnienie dysku 85 s (wolne 13 → 31 GB z 72 GB, po budowie
    24 GB), model z Release + weryfikacja 4 s, budowa obrazu od zera 98 s (2,21 GB), klaster kind
    44 s, `kind load` 100 s, rollout 13 s, smoke 3 s; tag w Deployment = `deploy.model_version`;
    akcje przypięte pełnym SHA, bez sekretów (tylko `GITHUB_TOKEN`)
  - [x] Próba odtworzenia na świeżych klonach (21–22.10): **łącznie ok. 38 min, 4 potknięcia**
    (2 niespełnione warunki wstępne, blokada portów, błąd własnego skryptu spoza repo).
    **Ciepłe cache uv, Dockera, wag HF i obrazu kind**.
    Czas obejmuje obsługę i diagnozę; nie jest obietnicą czasu pierwszego uruchomienia.
  - Zaliczono 21.10 (`4504fcc`): testy 264/264, lint, weryfikację Release, budowę obrazu oraz
    A1–A6 w osobnym klonie. Ocena testowa raz w nowym MLflow; predykcje i score wszystkich
    trzech modeli identyczne z pierwotnymi runami — porównanie `predictions_test.csv` z obu
    magazynów MLflow po `row_id` (3 080 wierszy na model, zero różnic klas i score).
    Zaliczono 22.10: smoke lokalnego API,
    kontenera i kind na 18080/18081, ze świeżego klonu lokalnego commita `a6d7be5`, z własnym
    środowiskiem i ponownie pobranym Release. A1–A6 bez powtórki.
  - Warunki wstępne: Docker Desktop był wyłączony, a aktywne konto `gh` (`MartinPrG`) nie miało
    dostępu do repo. Uruchomiono Dockera i użyto konta `MarcinPrDev`; zmiana konta zajęła
    ok. 30 s od odmowy do poprawnego ponowienia komendy. To nie błędy instrukcji.
  - Co nie zadziałało: wykluczone porty hosta 8080/8081 oraz zbyt krótki timeout we własnym
    skrypcie pomocniczym spoza repo, sprawdzającym gotowość API (nie w `smoke_api.py`).
    Wykluczenie TCP 8071–8170 potwierdził `netsh interface ipv4 show excludedportrange protocol=tcp`.
    Objaw: odmowa dostępu do gniazda, mimo załadowania modelu i poprawnej budowy obrazu.
    Czas stracony na Docker, porty i skrypt pomocniczy: niezmierzony osobno. Po uruchomieniu
    Dockera, zmianie konta i portów oraz pominięciu tego skryptu właściwe smoke testy przeszły.
  - Wcześniejsza próba 18.10 (próba techniczna asystenta): testy, budowa obrazu i smoke
    kontenera oraz kind przeszły.
    Poprawiono kolejność pobrania modelu przed API, dopisano `--extra cpu` dla macOS oraz
    sekcję *Odtworzenie wyników* (A — zamrożone modele, B — ponowne HPO).
  - Odniesienie dla wdrożenia bez cache: [Deploy kind, run 39138227351](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39138227351)
    na świeżym runnerze GitHuba — 6 min 28 s, szczegóły wyżej. Ten run nie obejmuje treningów
    A1–A6. Po zmianie portu kind na 18081 workflow zielony na `7a97783`
    ([run 39849732584](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39849732584))
    i `eaed6f7` ([run 39858858539](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39858858539)).
  - [x] Stretch po rdzeniu, do czw 22.10 — wszystkie cztery:
    - [x] seedy DistilBERT na walidacji (18.10): tabela i wniosek w *Wyniki na walidacji*
    - [x] LLM zero-shot (20.10): plan Groq Developer (decyzja autora; limity 250 000 tokenów/min
      zamiast 8 000), walidacja v1 i v2, zamrożony v2, test 385 raz — tabela przy ocenie
      końcowej wraz z wnioskiem
    - [x] UI Streamlit (21.10): `src/banking77_intent_ml/ui/` — cienki klient API (formularz,
      `/health`, `/predict`, czytelne błędy 422 i braku połączenia), extra `ui` (streamlit
      1.66.1), 4 testy (Streamlit AppTest z atrapą API); sprawdzone na żywo z lokalnym API:
      wymyślona wiadomość o nieotrzymanej karcie → `card_arrival`, score 0,986
    - [x] GKE Autopilot (22.10, ręcznie, poza CI): ten sam obraz i manifesty w `europe-central2`,
      smoke `[OK]` przez port-forward, klaster i rejestr usunięte tego samego wieczoru —
      szczegóły i koszt w *Wdrożenie*
  - Co nie zadziałało: Groq zwracał błędy 429 przy limicie 8 000 tokenów/min — pomogła zmiana
    planu; przy małym limicie odpowiedzi rozumowanie zużywało go w całości, więc odpowiedź
    była pusta. W nowym projekcie GCP konto usługi węzłów nie miało żadnych ról.
- [x] **W5** (27–31.10; wysyłka przesunięta na pn 2.11 — decyzja autora 31.10): raport, demo,
  reprodukowalność, dodatki, oddanie
  - [x] Raport (27–29.10): odpowiedź na pytanie badawcze, ograniczenia, co nie zadziałało,
    tabela wymagań projektu, EDA w *Dane*, koszt GKE z rozliczeń, spis treści i korekta spójności
  - [x] Test obrony „całość” (28–29.10): 21 pytań, w tym 3 system design
  - [x] Kontrola na świeżym klonie (29.10): ok. 7 min 15 s — *Quickstart*
  - [x] gitleaks v8.30.1 czysty (`gitleaks git --log-opts="--all" --redact`; 28.10: 123 commity,
    31.10: 168 commitów we wszystkich refach); `LICENSE` MIT (28.10); repo publiczne od 31.10
    (PR #8), model z Release przez `curl` bez logowania
  - [x] Dodatki po głównym badaniu, bez zmiany modelu do wdrożenia, każdy przez PR z zielonym
    CI i Deploy kind: metryki Prometheusa (PR #2), test istotności na 3 080 (PR #3), Basal
    zero-shot 4,5B (PR #4) i ultra na H100 (PR #5), opis kalibracji (PR #6), GIF, Space na
    Hugging Face i `docs/PITCH.md` (PR #7)
  - [x] Tag `v1.0-submission` — oznacza tę wersję kodu i README (31.10)
  - [x] Wysyłka do prowadzącego (pn 2.11); feedback otrzymany 8.11
- [x] **Poprawki po feedbacku prowadzącego (8.11.2026)** — jedyna uwaga techniczna: obraz serwujący
  zawierał zależności treningowe. Zrobione: grupa `train` w `pyproject.toml`, obraz inferencji bez niej
  (3,19 → 2,49 GB lokalnie; na runnerze CI 2,21 → 1,71 GB, budowa 98 → ok. 56 s), cel `train`
  w Dockerfile, `scripts/` w kontekście budowy; 319 testów i lint zielone, smoke kontenera `[OK]`,
  `deploy-kind` zielony na PR #10. Wersja wysłana pozostaje pod tagiem `v1.0-submission`.
  - Co nie zadziałało: plan zakładał wysyłkę 31.10, ale dodatki po głównym badaniu (Basal,
    kalibracja, demo online) przesunęły ją na 2.11. Model Basal 11B FP8 odpadł technicznie
    (tryby z grafami CUDA na Windows), a Space Docker na Hugging Face wymaga konta PRO.
- [x] **2.10 — zmiana nazwy** repo, pakietu i folderu na `banking77-intent-ml` /
      `banking77_intent_ml` (`docs/RENAME.md`)
- [x] **2.10 — porządki po poprzednim temacie**: notebook EDA, `csv_to_parquet.py`, `sanity_w1.py`
      i loader US Accidents usunięte z repo (treść w historii gita do commita `bc20003`; kopie
      robocze i dane poza repo). Loader BANKING77 i nowy `notebooks/01_eda.ipynb` powstają w W1
- Rezerwa: zużyto **8 z 8 h** (stan po W1, 9.10: W1 zajął 20 h przy budżecie 12 h; kolejne
  nadwyżki = świadome nadgodziny albo cięcie wg kolejności z *Bramka po W1 i fallback*;
  aktualizacja na koniec każdego slotu)
  - W2 (12–14.10): **14 h pracy przy budżecie 14 h — 0 h nadwyżki**; rezerwa nadal zużyta 8 z 8 h.
  - W3 (14–18.10): **14 h pracy przy budżecie 14 h — 0 h nadwyżki**; rezerwa nadal zużyta 8 z 8 h.
  - W4 (18–22.10): **ok. 22 h pracy przy budżecie 14 h — ok. 8 h nadwyżki** (świadome nadgodziny;
    rezerwa wyczerpana po W1). Łącznie W1–W4 ok. 70 h wobec 54 h budżetu slotów.
  - W5 (26–31.10): **24 h pracy przy budżecie 6 h — 18 h nadwyżki** (świadome nadgodziny:
    wieczory 26–30.10 i dodatki 29–31.10), w tym ok. 21 h w sesjach z asystentem (znaczniki
    czasu sesji, przerwa powyżej 30 min kończy blok). Łącznie W1–W5 ok. 94 h wobec 60 h planu.
