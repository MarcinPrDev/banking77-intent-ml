# Klasyfikacja zgłoszeń na BANKING77 — porównanie jakości, kosztu i wdrożenia metod NLP

[![CI](https://github.com/MarcinPrDev/banking77-intent-ml/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/MarcinPrDev/banking77-intent-ml/actions/workflows/ci.yaml)
[![Deploy kind](https://github.com/MarcinPrDev/banking77-intent-ml/actions/workflows/deploy-kind.yaml/badge.svg?branch=main)](https://github.com/MarcinPrDev/banking77-intent-ml/actions/workflows/deploy-kind.yaml)

Projekt końcowy kursu **AI & ML Engineering** (SAGES/Kodołamacz, gr. 06.12.2025).
Klasyfikacja intencji pojedynczej wiadomości klienta banku do jednej z **77 kategorii**
na benchmarku [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, 2020).

> Historia projektu — zmiana tematu 30.09.2026, zmiana nazwy repo 2.10, plan 5 slotów, przebieg i księga
> godzin: [docs/HISTORIA.md](docs/HISTORIA.md) i [docs/STATUS.md](docs/STATUS.md).

> **Co system robi:** dostaje angielską wiadomość („I still haven't received my card"), zwraca
> identyfikator intencji (`card_arrival`). **Czego nie robi:** nie rozwiązuje sprawy klienta,
> nie ocenia ryzyka, nie wykrywa intencji spoza 77 klas (test nie waliduje OOD).

![Demo: UI Streamlit nad API z modelem distilbert-v1 — dwie wiadomości i zwrócone intencje](docs/figures/demo-ui.gif)

*Demo z 31.10.2026: UI Streamlit (stretch 3) nad lokalnym API z modelem `distilbert-v1`.
Zdania kontrolne z `configs/smoke_cases.json`, spoza zbioru BANKING77; szczegóły w [Demo](#demo).
Demo online z tym samym modelem:
[Hugging Face Space `MarcinPr/banking77-intent-demo`](https://huggingface.co/spaces/MarcinPr/banking77-intent-demo).*

**Pytanie badawcze:** czy dostrajany mały transformer (DistilBERT) poprawia jakość klasyfikacji
względem TF-IDF + regresji logistycznej i zamrożonych embeddingów + kNN **na tyle, żeby uzasadnić
większy koszt uczenia i predykcji na CPU**? Wynikiem projektu jest uzasadniony wybór jednego modelu
i działająca usługa — także wtedy, gdy najdroższy model nie wygra.

## W skrócie

Porównałem trzy metody rozpoznawania 77 intencji bankowych i wdrożyłem wybrany model jako API
w Dockerze, z automatycznym wdrożeniem do kind w CI.
Na teście 3 080 wiadomości DistilBERT osiągnął accuracy 93,02%, MiniLM+kNN 92,05%,
a TF-IDF+LR 90,81%.
Odpowiedź na pytanie badawcze brzmi „częściowo”: przy pierwszeństwie jakości uznaję większy koszt
DistilBERT za uzasadniony względem TF-IDF+LR (p95 rozgrzanej predykcji CPU przy jednym wątku,
batch=1, bez narzutu HTTP: 31,42 wobec 0,43 ms), ale przewaga accuracy nad kNN nie osiągnęła
istotności w teście McNemara po korekcie Holma.
DistilBERT wybrano po obejrzeniu wyników testu, więc wynik nie jest niezależnym potwierdzeniem
wyboru ani dowodem opłacalności w rzeczywistym banku.

## Spis treści

- **Skrót na 2 minuty:** [docs/PITCH.md](docs/PITCH.md)
- **Raport:** [Dane](#dane) · [Protokół ewaluacji](#protokół-ewaluacji-wspólny-dla-wszystkich-modeli)
  · [Modele i budżet strojenia](#modele-i-budżet-strojenia) · [Wyniki](#wyniki)
  ([tabela końcowa](#tabela-końcowa)) · [Analiza błędów](#analiza-błędów-w3)
  · [Odpowiedź na pytanie badawcze](#odpowiedź-na-pytanie-badawcze)
  · [Ograniczenia](#ograniczenia-i-granice-weryfikacji)
  · [Co nie zadziałało](#co-nie-zadziałało-i-co-zrobiłbym-inaczej) · [Dalsze prace](#dalsze-prace)
- **Uruchomienie:** [Quickstart](#quickstart) · [Demo](#demo) · [Odtworzenie wyników](#odtworzenie-wyników)
  · [Wdrożenie](#wdrożenie)
- **Projekt:** [Architektura](#architektura-wzorce-i-solid) · [Przepływ artefaktów modelu](#przepływ-artefaktów-modelu)
  · [Wymagania projektu końcowego](#wymagania-projektu-końcowego--gdzie-są-spełnione)
  · [Dlaczego bez Sparka i CV](#dlaczego-bez-sparka-i-cv)
- **Przebieg projektu (poza raportem):** [docs/STATUS.md](docs/STATUS.md) — plan 5 slotów, bramka po W1,
  minimum vs stretch, status i księga godzin · [docs/HISTORIA.md](docs/HISTORIA.md) — zmiana tematu, ADR-y
  · [Tooling](#tooling--współpraca-z-ai) · [Licencja](#licencja)

## Dane

| | Trening (`train.csv`) | Test (`test.csv`) |
|---|---:|---:|
| Wiersze | 10 003 | 3 080 |
| Kategorie | 77 | 77 |
| Przykłady na kategorię | 35–187 (mediana 127) | dokładnie 40 |
| Puste teksty / etykiety | 0 / 0 | 0 / 0 |
| Nadmiarowe duplikaty po normalizacji¹ | 31 | 4 |
| Grupy znormalizowane z różnymi etykietami | 1 | 1 |

¹ Normalizacja audytowa: Unicode NFKC → casefold → tokeny `\w+` połączone spacją. **25 tekstów
testowych ma odpowiednik w train** po tej normalizacji (etykiety zgodne); bez nich zostaje **3 055**.
Liczby z własnego audytu danych (30.09.2026, normalizacja jak w `split.dedup_normalization`
w configu). Kod w repo: duplikaty i konflikt w train — notebook 01 §5 i `data/splits.py`; pokrycie
testu i podzbiory 3 055 / 385 — `data/audit.py` (CLI `scripts/make_test_subsets.py`); notatka
źródłowa poza repo.

- **Źródło:** repozytorium `PolyAI-LDN/task-specific-datasets`, przypięty commit
  `57ec275d8078af65b7731c2a98be812d844a6d6b`, pliki `banking_data/train.csv`
  (SHA-256 `b06e26ac…cbc664b`) i `banking_data/test.csv` (SHA-256 `d12d6e3b…cfeeb474d`), łącznie
  1 079 034 B. Pełne sumy w `configs/config.yaml`; pobranie i weryfikacja: `scripts/fetch_data.py`.
  Nie mieszamy z kopiami z HF/MTEB (inna licencja w metadanych, inny protokół, w v2 usunięte wiersze).
- **Licencja danych:** CC BY 4.0 — atrybucja w sekcji *Licencja*. Dane poza repo; w repo kod,
  skrypt pobrania, mapa etykiet i manifest splitu.
- **Co wiemy o pochodzeniu:** karta danych nie dokumentuje sposobu pozyskania ani anonimizacji
  (przy danych osobowych wpis „N/A"). Własny skan 13 083 tekstów pięcioma wzorcami (e-mail, URL,
  ≥9 cyfr, IBAN, przedstawianie się) dał **0 trafień — to nie dowodzi braku PII** (bez NER, bez
  ręcznego przeglądu). **Wykrycie istotnych PII w trakcie EDA = podstawa rezygnacji ze zbioru.**
- **Znane problemy etykiet:** Ying & Thomas (2022) wskazują automatycznie **1 428 potencjalnych
  problemów w train (~14%)**. Nie usuwamy ich hurtowo i nie zmieniamy etykiet testu po obejrzeniu
  predykcji; analizujemy pomyłki i niejednoznaczne intencje w analizie błędów.
- **Ograniczenia zbioru:** test jest sztucznie zrównoważony (wynik nie odwzorowuje częstości
  rzeczywistych zgłoszeń); przy 40 przykładach na klasę jedna pomyłka = 2,5 p.p. recallu klasy;
  brak czasu i identyfikatorów rozmów (nie da się zweryfikować podziału chronologicznego).
- **EDA — tylko train (10 003 wiadomości):** mediana / P95 / maksimum długości:
  znaki 47 / 151,9 / 433, słowa 10 / 29 / 79, tokeny DistilBERT z tokenami specjalnymi
  13 / 37 / 98 ([notebook, §4, §6 i §9](notebooks/01_eda.ipynb)).
  `max_len=128` nie obcina żadnej wiadomości train; nie daje to gwarancji dla przyszłych zgłoszeń.
  Najkrótsze teksty mają mało kontekstu, wśród najdłuższych są opisy weryfikacji w formie
  odpowiedzi, a podobne intencje bywają niejednoznaczne i wymagają uwagi na szczegóły.

Wykresy z notebooka 01 (train, 10 003 wiadomości; notebooki w repo są bez wyników — filtr
nbstripout — więc tu zapisane wyniki komórek, bez tekstów zdań):

![Liczność 77 klas w train.csv, posortowana, z linią mediany](docs/figures/eda-class-counts-train.png)

![Histogramy długości tekstów w train: znaki i słowa, z medianą](docs/figures/eda-text-lengths-train.png)

## Protokół ewaluacji (wspólny dla wszystkich modeli)

1. **Przypięte:** commit, sumy plików, mapa nazw kategorii → id (`configs/label_map.json`).
2. **Walidacja z train:** ~15% (`split.val_size`), stratyfikowana; **znormalizowane duplikaty
   trafiają razem do jednej części**; indeksy w `dane/splits/` (w gicie — same numery wierszy,
   bez tekstów; katalog nietykalny)
   + manifest `configs/split_manifest.json` (git-tracked: seed, val_size, n_train, n_val,
   sha256 indeksów). Konflikty etykiet zbadane przed podziałem.
3. **Test tylko do dwóch rzeczy:** audytu integralności i oceny końcowej. Żadnego fitu, HPO,
   wyboru epoki, progu ani doboru promptu na teście. TF-IDF dopasowany wyłącznie na train.
4. **Refit — decyzja 14.10.2026:** trzy zamrożone konfiguracje uczone od nowa na całych
   10 003 zdaniach (train + dawna walidacja). DistilBERT: harmonogram na 5 epok, stop po 4.
   Po reficie brak niezależnej walidacji. Porównanie z *Full*: accuracy na pełnym teście 3 080,
   z uwzględnieniem różnic strojenia i jednego seeda.
5. **Metryki:** `accuracy` — główna, do porównania z literaturą; **macro-F1 — metryka wyboru
   checkpointu/konfiguracji na walidacji (ustalone z góry)**; recall i F1 per klasa; macierz
   pomyłek. Przy jednym seedzie nie wyciągamy wniosków z różnic o ułamek punktu procentowego.
6. **Kontrola wrażliwości:** każdy model raportowany także na podzbiorze **3 055** tekstów bez
   pokrycia z train (ustalonym przed treningiem). To nie nowy benchmark — nie porównywać
   z publikacjami na 3 080.
7. **Koszt:** czas treningu, rozmiar całego artefaktu, latencja CPU p50/p95 dla batch=1
   (z preprocessingiem i ekstrakcją embeddingów), **zimny start osobno od rozgrzanej predykcji**.
   Surowego wyniku kNN/softmaxu nie nazywamy skalibrowanym prawdopodobieństwem.
8. **LLM (stretch A):** jeden model, jeden zamrożony prompt, **385 wiadomości testowych
   (5 na klasę) wybranych z góry** — te same 385 dla wszystkich modeli w tabeli; raport jakości,
   błędów formatu, tokenów, latencji i kosztu. Dobór promptu tylko na walidacji. **Ekspozycja LLM
   na benchmark nieustalona** — brak gwarancji czystego testu; wynik nie dowodzi generalizacji.

**Zamrożenie — 14.10.2026, przed pierwszym wynikiem na teście:**

- TF-IDF+LR: próba #12; MiniLM+kNN: k = 1, `uniform`, cosinus;
  DistilBERT: LR 6e-5, epoka 4 przy harmonogramie na 5 epok, `max_len=128`.
- Pełne parametry: `models.<nazwa>.frozen` i baza w configu; przypięte rewizje wag, seed 42.
  Bez nowego HPO i zmian konfiguracji po wyniku testowym.
- Kolejność: commit z tym wpisem i [ADR-0002](docs/adr/0002-protokol-ewaluacji.md)
  → tag `experiments-frozen` na tym commicie → refit → jednorazowa ocena na teście.
- Artefakty W2 zachowane; finalne w `models/final/`. Koszty finalnych modeli mierzone ponownie.

**Punkty odniesienia (protokół Full, pełny train i test):** DistilBERT-base **92,1%**
(Qi i in. 2021: batch 32, max_len 128, AdamW wd 0,01, warmup 10% do LR 4e-5, 5 epok);
BERT-base 93,3%; RoBERTa-base 94,1%; BERT-large 93,66% (Casanueva i in. 2020).
Dummy (jedna klasa) = 1/77 = **1,30%**. Kanonicznego wyniku TF-IDF+LR nie ma — wyznaczamy własny.
Liczby z publikacji to konkretne konfiguracje, nie gwarantowany przedział wyników tego projektu.

## Modele i budżet strojenia

| Model | Budżet strojenia (walidacja, MLflow) | Cel | Wagi / licencja |
|---|---|---|---|
| Dummy | brak | kontrola rozkładu i metryk | — |
| **TF-IDF + Logistic Regression** | `tfidf_lr.hpo_trials` = 16 konfiguracji wektoryzatora (słowa/znaki, n-gramy, min_df), regularyzacji (C) i wag klas (`class_weight`); punkt startu = pierwsza konfiguracja z configu (W1) | tani, interpretowalny baseline; kandydat do wdrożenia | — |
| **all-MiniLM-L6-v2 (zamrożony) + kNN** | kilka `k` × ważenie (uniform/distance) | wartość reprezentacji semantycznej bez dostrajania | HF `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, rewizja przypięta w configu |
| **DistilBERT + głowica 77 klas** | 5 LR {2–6}e-5 × 5 epok = 25 kandydatów; budżet ustalony przed wynikami, pozostałe parametry wg Qi 2021 | efekt dostrajania reprezentacji; RTX 5080 | HF `distilbert/distilbert-base-uncased`, Apache-2.0, rewizja przypięta |
| *(stretch A)* LLM zero-shot | brak strojenia; prompt dobrany na walidacji | koszt/latencja/błędy formatu vs modele dostrajane | Groq (plan płatny, Developer tier) przez LiteLLM; model w `llm.model` |
| *(fallback)* TF-IDF + LinearSVC | mały grid C | trzecia metoda, gdy transformer nie ruszy — niewykorzystany | — |

Wartości liczbowe (budżety, hiperparametry startowe, seed) — wyłącznie z `configs/config.yaml`.
Budżet DistilBERT zwiększono z 2–3 do 5 LR, by sprawdzić wartości po obu stronach 4e-5
z publikacji; decyzja przed wynikami zapobiega dokładaniu prób pod wynik, ale nie wyrównuje
budżetów między modelami.
Czas treningu, VRAM i koszty CPU na sprzęcie autora zapisano w MLflow.

## Architektura (wzorce i SOLID)

Trzy adaptery w `models/classifiers.py` realizują interfejs `IntentClassifier`
z `models/predictor.py`. Fabryka w `models/factory.py` buduje lub wczytuje wybrany predyktor.

- **Strategy:** wspólny interfejs predyktora (`fit / predict / predict_proba / save / load`) z
  implementacjami dla TF-IDF+LR, MiniLM+kNN i DistilBERT. API i ewaluacja widzą tylko interfejs
  — podmiana modelu = zmiana wpisu w configu, nie kodu. LLM (stretch) **poza** interfejsem
  (decyzja autora 18.10.2026): kontrakt wymaga `predict` == argmax(`predict_proba`), a odpowiedź
  spoza 77 nazw ma zostać błędem formatu, nie klasą 0 — ocena w `evaluation/llm_eval.py`.
- **Factory:** `build_predictor` tworzy adapter zamrożonej konfiguracji,
  a `load_predictor` wczytuje zapisany artefakt; wybór klasy przez rejestr `STRATEGIES`.
- **Pipeline sklearn** w TF-IDF+LR: `TfidfVectorizer` → `LogisticRegression`, zapis razem
  w `model.joblib`. W MiniLM+kNN kodowanie tekstu przez `features/embeddings.py`, a indeks
  kNN zapisany osobno; bez własnych transformerów sklearn.
- **LabelMap** — jedna klasa mapująca nazwa ↔ id, ładowana z przypiętego pliku; wszystkie modele
  i API korzystają z tej samej mapy etykiet.
- **LLMClient** (stretch) — wąski interfejs `complete(prompt)`, implementacja `LiteLLMClient`
  (Groq przez LiteLLM) i `CachedClient` (cache odpowiedzi) — DIP, w testach atrapa bez sieci.
- Mapowanie SOLID: SRP (moduły `data/features/models/evaluation/api`), OCP (nowy model = nowa
  klasa i wpis w rejestrze fabryki), LSP (każdy predyktor przechodzi ten sam test kontraktu
  zapis→odczyt→identyczne predykcje), DIP (API zależy od interfejsu, nie od torcha).

```mermaid
classDiagram
    class IntentClassifier {
        <<abstract>>
        +fit(texts, labels)
        +predict(texts)
        +predict_proba(texts)
        +save(dir)
        +load(dir)$
    }
    IntentClassifier <|-- TfidfLrClassifier
    IntentClassifier <|-- MiniLmKnnClassifier
    IntentClassifier <|-- DistilBertClassifier
    ClassifierFactory ..> IntentClassifier : buduje z config.yaml
    IntentClassifier --> LabelMap : nazwa <-> id
    PredictService --> IntentClassifier : FastAPI /predict
    LlmEvaluation --> LLMClient : DIP, stretch A (poza IntentClassifier)
    LLMClient <|.. LiteLLMClient
    LLMClient <|.. CachedClient
```

Na diagramie `ClassifierFactory` oznacza funkcje w `models/factory.py`, a `PredictService`
obsługę żądań w `api/app.py`; nie są to osobne klasy w kodzie.

## Przepływ artefaktów modelu

**MLflow** (lokalnie, poza gitem: baza `mlflow.db` z runami + artefakty w `mlruns/`; podgląd:
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) = tracking wszystkich runów: parametry, metryki na walidacji,
czas treningu, rozmiar artefaktu. **Wybrany model ([ADR-0003](docs/adr/0003-wybor-modelu-do-wdrozenia.md), 15.10): DistilBERT** — oceniony
artefakt z refitu w `models/final/distilbert`, bez kopiowania. Sekcja `deploy` w configu
(`model_version: distilbert-v1`, `artifact_dir`) — jedno źródło dla build-arg obrazu, tagu,
manifestu k8s i odpowiedzi `/health` (`MODEL_VERSION`; bez tej zmiennej API zwraca
`deploy.model_version`). Manifest z hashami plików powstaje przy pakowaniu do Release
(`scripts/model_release.py pack`, W4).
Artefakt jest **wpiekany w obraz CPU podczas `docker build`** (bez pobierania na starcie
kontenera — brak cold startu i uprawnień runtime). Tag obrazu zawiera wersję modelu.

**Skąd CI bierze artefakt do `docker build` — rozstrzygnięte w ADR-0003: opcja (a)**, GitHub
Release asset przypięty tagiem (`deploy.model_version`) + sha256 w configu; pliki bit w bit
zgodne z ocenionym artefaktem. Rozważane były też: (b) trening lekkiego modelu w CI z danych
pobranych przez `fetch_data.py --only train` (bez `test.csv`; działa tylko dla TF-IDF; manifest
splitu niesie tylko hash, więc w CI trzeba odtworzyć podział `make_splits` i porównać hash
z manifestem — albo refitować na całych 10 003 bez podziału) — odrzucone, bo tworzy nowy
artefakt; (c) GCS (tylko jeśli GCP wejdzie jako stretch) — odrzucone. W4 implementuje (a).
Wybrany DistilBERT wymaga torcha CPU w obrazie — rozmiar 3,18 GB (W3); budowa od zera w CI
98 s (run `deploy-kind` na PR #1, 18.10 — [Status](docs/STATUS.md), W4). Po feedbacku prowadzącego (8.11.2026)
zależności treningu i analizy wydzielono z obrazu serwującego — zob. *Wdrożenie: Docker*.

**Release `distilbert-v1` (18.10):**
[github.com/MarcinPrDev/banking77-intent-ml/releases/tag/distilbert-v1](https://github.com/MarcinPrDev/banking77-intent-ml/releases/tag/distilbert-v1)
— tag na `d809c57` (commit refitu, `experiments-frozen`), asset `distilbert-v1.tar`
(268 789 760 B), SHA-256 w `deploy.release_sha256`. Archiwum: niezmienione pliki
`models/final/distilbert` + `manifest.json` (SHA-256 każdego pliku, hash mapy etykiet, wersje
bibliotek, runy refit `5b883a97` i ocena `fb0d0bde`); tar bez kompresji z ustalonymi
metadanymi — te same pliki dają tę samą sumę. `scripts/model_release.py unpack` sprawdza sumę
archiwum, wersję, mapę etykiet, skład archiwum i sumę każdego pliku.

**MiniLM+kNN (niewybrany w ADR-0003; opis zostaje dla porządku):** adapter zapisuje indeks kNN i metadane, a wagi enkodera
z przypiętej rewizji pozostają w cache Hugging Face. Obraz musi zawierać również enkoder:
do wyboru wpieczenie cache HF albo zapis enkodera wewnątrz artefaktu. W tabeli kosztów
rozmiar obejmuje **indeks kNN + wagi, tokenizer i konfigurację enkodera**. `load()` rozgrzewa
enkoder przed gotowością API; pomiar zimnego startu nadal obejmuje `load` i pierwszą predykcję
w nowym procesie.

## Quickstart

**Wymagania przed startem:** Python 3.11 i uv; `curl` (model z GitHub Release — repo publiczne
od 31.10.2026, pobranie bez logowania); do obrazu i klastra uruchomiony Docker
(Docker Desktop) oraz kind v0.33.0 i kubectl ≥ 1.36 w `PATH`; do treningu DistilBERT GPU NVIDIA.
Na Windows sprawdź też porty (niżej: *Porty lokalne*).

**Kontrola na świeżym klonie (29.10.2026, autor):** klon `46f47df` → środowisko → dane z SHA-256
→ weryfikacja plików W1 → testy i lint → modele bazowe (wyniki jak w *Wyniki na walidacji*)
→ model z Release → obraz (budowa 97 s) → smoke `[OK]` — **ok. 7 min 15 s**, jedno potknięcie
obsługi (smoke trzeba uruchomić w drugim terminalu, przy działającym kontenerze); ciepłe cache uv
i Dockera, bez kind i bez treningu DistilBERT.

```bash
# Środowisko lokalne (Python 3.11, uv) — JEDNA komenda z pełną listą extras używanych lokalnie:
uv sync --extra dev --extra nlp --extra gpu --extra serving
# Jednorazowo po klonie: filtr gita czyści wyjścia notebooków przy commicie (lokalnie zostają);
# --keep-id zachowuje identyfikatory komórek (bez niego nbstripout numeruje je od nowa)
uv run nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean "$(git config filter.nbstripout.clean) --keep-id"
uv run python scripts/fetch_data.py        # train.csv + test.csv z przypiętego commitu, weryfikacja SHA-256
uv run pytest                              # strażnik protokołu + testy dymne
uv run ruff check .
# UWAGA: `uv sync` jest "exact" — extras pominięte w komendzie są USUWANE ze środowiska
# (torch/transformers znikną po gołym `uv sync --extra dev`). Goły `--extra dev` jest tylko dla CI.
# Po każdym sync na maszynie z GPU: python -c "import torch; print(torch.cuda.is_available())"
# gpu = torch z indeksu cu128 (RTX 5080/sm_120) + transformers; nlp = sentence-transformers;
# serving = fastapi, uvicorn, httpx (API od W2); llm (stretch A): dodatkowo --extra llm
# + GROQ_API_KEY w .env (poza gitem); ui (stretch 3, Streamlit): dodatkowo --extra ui
# Bez GPU NVIDIA albo na macOS: zamiast --extra gpu użyj --extra cpu (torch 2.11.0 z indeksu CPU,
# jak w obrazie; koła cu128 są tylko dla Linuksa i Windows). Wystarcza do testów, API i obrazu;
# trening DistilBERT wymaga --extra gpu (accelerate + CUDA).
```

**Model do API i obrazu** — oceniony artefakt `models/final/distilbert` z GitHub Release
(bez logowania; alternatywnie `gh release download distilbert-v1 -p distilbert-v1.tar -D dist`)
albo z refitu (`uv run python scripts/refit_final.py`):

```bash
curl -L --create-dirs -o dist/distilbert-v1.tar https://github.com/MarcinPrDev/banking77-intent-ml/releases/download/distilbert-v1/distilbert-v1.tar
uv run python scripts/model_release.py unpack dist/distilbert-v1.tar   # weryfikacja → models/final/distilbert
```

**Porty lokalne:** 18080 (API, kontener, UI) i 18081 (klaster kind). Na Windows Hyper-V, WSL2
i Docker Desktop dynamicznie wykluczają bloki portów TCP, a zakresy zmieniają się po restarcie
(21.10.2026 m.in. 7969–8068 i 8071–8170, czyli 8000, 8080 i 8081). Sprawdzenie:
`netsh interface ipv4 show excludedportrange protocol=tcp`. Jeśli port wypada w zakresie, wybierz
inny — dla kind także `hostPort` w `k8s/kind/cluster.yaml`.

**API lokalnie** — serwuje model z `deploy.artifact_dir` (ADR-0003: DistilBERT); dokumentacja
endpointów: `http://localhost:18080/docs`.

```bash
uv run uvicorn banking77_intent_ml.api.app:create_app --factory --port 18080
```

**UI demo** (stretch 3, `--extra ui`) — Streamlit nad działającym API: formularz, stan
`/health`, wynik `/predict`; model zostaje w API. Adres API w panelu bocznym (domyślnie
`ui.api_url` z configu = `http://localhost:18080`; dla kind `http://localhost:18081`):

```bash
uv run streamlit run src/banking77_intent_ml/ui/app.py
```

**Obraz Docker CPU** (W3) — z artefaktem pobranym wyżej; `MODEL_VERSION` = `deploy.model_version`
z configu:

```bash
docker build --build-arg MODEL_VERSION=distilbert-v1 -t banking77-intent-api:distilbert-v1 .
docker run --rm -p 18080:8000 banking77-intent-api:distilbert-v1
uv run python scripts/smoke_api.py --url http://localhost:18080   # drugi terminal: smoke
# Obraz treningowo-analityczny na CPU (ten sam Dockerfile, cel `train`: grupa `train` + MiniLM;
# trening DistilBERT nadal wymaga GPU poza obrazem) — poprawka po feedbacku, 8.11.2026:
docker build --target train -t banking77-intent-train .
```

**Klaster kind lokalnie** (W4; kind v0.33.0, kubectl ≥ 1.36, obraz zbudowany wyżej) — te same
manifesty co w CI (`k8s/`); Service NodePort dostępny z hosta pod `127.0.0.1:18081`
(`k8s/kind/cluster.yaml`); tag obrazu = `deploy.model_version`:

```bash
kind create cluster --name banking77 --config k8s/kind/cluster.yaml
kind load docker-image banking77-intent-api:distilbert-v1 --name banking77
kubectl apply -f k8s/
kubectl set image deployment/banking77-intent-api api=banking77-intent-api:distilbert-v1
kubectl rollout status deployment/banking77-intent-api --timeout=180s
uv run python scripts/smoke_api.py --url http://localhost:18081
kind delete cluster --name banking77
```

Ten sam przebieg w GitHub Actions: `.github/workflows/deploy-kind.yaml` (push na `main`, pull
request, ręcznie) — pierwszy zielony run na `main`:
[39138227351](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39138227351).

**GKE Autopilot (stretch, płatne, wykonane raz 22.10):** ten sam obraz i te same manifesty; komendy,
przebieg i koszt — [docs/GKE.md](docs/GKE.md).

## Demo

Obraz z *Quickstart* (DistilBERT wpieczony w obraz CPU), `/health` i `/predict` na trzech
zdaniach. Zdania pochodzą z `configs/smoke_cases.json` — napisane na potrzeby testu dymnego,
nie ze zbioru BANKING77. Demo pokazuje, że usługa działa; o jakości mówią tabele wyników,
nie te trzy zdania.

**Online (31.10.2026):** [Hugging Face Space `MarcinPr/banking77-intent-demo`](https://huggingface.co/spaces/MarcinPr/banking77-intent-demo)
— Streamlit z tym samym artefaktem `distilbert-v1` ładowanym w procesie, bez API; kod
w [`hf_space/`](hf_space/), wagi wysłane z `models/final/distilbert` (SHA-256 w karcie Space'a).
Predykcja jak w adapterze projektu: 14 zdań kontrolnych daje te same intencje co API, score różni
się najwyżej o 9,5e-7. Space Docker na CPU basic (na Hugging Face wymaga konta PRO); po dłuższej
bezczynności usypia, więc pierwsze wejście trwa dłużej.

```bash
docker run --rm -p 18080:8000 banking77-intent-api:distilbert-v1
# drugi terminal (bash; bez terminala: http://localhost:18080/docs)
curl -s http://localhost:18080/health
curl -s -X POST http://localhost:18080/predict -H "Content-Type: application/json" \
  -d "{\"text\": \"My new card still hasn't arrived, when will it come?\"}"
```

Prawdziwe odpowiedzi (28.10.2026, obraz zbudowany z kodu zgodnego z `main`):

```text
{"status":"ok","model":"distilbert","model_version":"distilbert-v1"}
{"intent":"card_arrival","intent_id":12,"score":0.9861456751823425,"model_version":"distilbert-v1"}
{"intent":"exchange_rate","intent_id":33,"score":0.8856154680252075,"model_version":"distilbert-v1"}
{"intent":"lost_or_stolen_card","intent_id":42,"score":0.6754384636878967,"model_version":"distilbert-v1"}
```

| Zdanie | Intencja (`intent_id`) | `score` |
|---|---|---:|
| My new card still hasn't arrived, when will it come? | `card_arrival` (12) | 0,986 |
| What exchange rate will I get if I pay in euros? | `exchange_rate` (33) | 0,886 |
| Someone stole my card, please block it right away. | `lost_or_stolen_card` (42) | 0,675 |

Pusty tekst albo same spacje → `422`. `score` to wynik modelu dla zwróconej klasy, nie
skalibrowane prawdopodobieństwo (zob. *Ograniczenia i granice weryfikacji*). Ten sam Dockerfile
i model z Release przechodzą w CI deploy do kind i test HTTP — ostatni zielony run na `main`:
[Deploy kind 40617499359](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/40617499359).

## Odtworzenie wyników

Quickstart odtwarza **wdrożenie** z ocenionym artefaktem z Release. Ta sekcja odtwarza **liczby**
z README (walidacja, koszt, test) od zera, w dwóch rozdzielnych przebiegach. Każdy przebieg to
**osobny klon i nowy magazyn MLflow**: `mlflow.db` i `mlruns/` powstają w katalogu klonu przy
pierwszym runie. Nie kopiuj ich z innego klonu — blokady budżetu liczą runy w magazynie.

- **A. Zamrożone modele** — konfiguracje z `models.*.frozen` w configu, bez strojenia: wyniki
  walidacji, refit na 10 003, koszt i **jedna** ocena testowa.
- **B. Ponowne HPO** — pełne badania W2 na walidacji; sprawdza procedurę wyboru konfiguracji.
  Nie zmienia configu i nie dotyka testu.

Przygotowanie (A i B): środowisko i dane jak w Quickstart (`--extra gpu` — trening DistilBERT
wymaga GPU NVIDIA; MiniLM liczy się też na CPU, wolniej). Wagi MiniLM i DistilBERT pobiorą się
przy pierwszym użyciu z HF Hub, pod rewizjami przypiętymi w configu. Pliki W1 są w gicie — skrypty
W1 tylko je weryfikują i niczego nie nadpisują:

```bash
uv run python scripts/make_label_map.py      # mapa etykiet = configs/label_map.json
uv run python scripts/make_splits.py         # podział train/val = manifest („zgodny”)
uv run python scripts/make_test_subsets.py   # podzbiory 3 055 / 385 = manifest (same liczby i hashe)
```

**A. Zamrożone modele** — kolejność obowiązkowa: refit i test dopiero po krokach walidacji.

| Krok | Komenda (`uv run python scripts/…`) | Blokada | Porównaj z |
|---|---|---|---|
| A1. Modele bazowe | `train_baselines.py --model dummy`, potem `--model tfidf_lr` | bez `--model` odmowa | *Wyniki na walidacji*: Dummy, TF-IDF + LR (W1) |
| A2. Zamrożone konfiguracje na train | `export_model.py tfidf_lr`, `minilm_knn`, `distilbert` | istniejący `models/<nazwa>` = odmowa | *Wyniki na walidacji* (accuracy i macro-F1 wypisuje skrypt; bez runu MLflow) |
| A3. Koszt CPU artefaktów W2 | `measure_cost.py` | badanie `cpu_cost_w2` raz w magazynie | *Koszt na CPU* |
| A4. Refit na 10 003 | `refit_final.py` | brak tagu `experiments-frozen`, istniejący artefakt albo run refitu = odmowa | runy `final_refit`, artefakty `models/final/` |
| A5. Koszt finalnych artefaktów | `measure_cost.py --final` | badanie `cpu_cost_final` raz w magazynie | tabela końcowa (koszt) |
| A6. **Ocena na teście — raz** | `evaluate_test.py` | brak tagu albo `models/final/`, istniejący run `split=test` = odmowa | tabela końcowa (3 080, 3 055) |

Po A6 analiza błędów: `notebooks/02_analiza_bledow.ipynb` (artefakty runów testowych z MLflow
i predykcje artefaktów W2 z A2 na walidacji). Jedna ocena testowa: drugiego przebiegu A6 w tym
samym magazynie skrypt nie wykona, a wynik testu nie wraca do kroków A1–A5. Na tym samym sprzęcie
(RTX 5080) ścieżki odtwarzały zamrożone runy co do bitu; na innym GPU lub sterowniku możliwe są
małe różnice. Nowe artefakty nie zastępują Release `distilbert-v1` — wdrożenie zawsze bierze
oceniony artefakt z Release (suma w configu).

**B. Ponowne HPO** — osobny klon i magazyn; bez refitu i bez testu.

| Krok | Komenda (`uv run python scripts/…`) | Blokada | Porównaj z |
|---|---|---|---|
| B1. TF-IDF + LR, Optuna TPE (16 prób) | `hpo_tfidf_lr.py` | badanie (tag `hpo_study`) raz w magazynie | `models.tfidf_lr.frozen` (próba #12) |
| B2. MiniLM + kNN, siatka k × wagi | `knn_minilm.py` | jak wyżej | `models.minilm_knn.frozen` (k = 1, `uniform`) |
| B3. DistilBERT, 5 LR × 5 epok | `train_distilbert.py` | jak wyżej; checkpointy ok. 6,3 GB w `dane/cache/distilbert` | `models.distilbert.frozen` (LR 6e-5, epoka 4) |
| B4. Seedy DistilBERT (stretch 1) | `seed_study_distilbert.py` | badanie `seed_study` raz; wymaga runu z B3 w tym samym magazynie (kontrola seeda 42) | *Rozrzut DistilBERT po seedach* |

Skrypty B drukują zwycięzcę jako fragment YAML — configu nie zmieniamy: konfiguracje zamrożono
14.10, przed testem (ADR-0002). Inny zwycięzca to wniosek do raportu, nie powód do nowego
strojenia.

**Stretch LLM** (opcjonalnie; klucz `GROQ_API_KEY` w `.env`, środowisko z `--extra llm`, płatne
API): `llm_zero_shot.py --stage val --prompt v1`, `--prompt v2`, potem `--stage test` z
`llm.frozen_prompt` z configu. Blokady jak wyżej: każdy wariant raz na walidacji, test raz w
magazynie; odpowiedzi w cache `dane/cache/llm` (przerwany przebieg bez ponownego kosztu).

## Wdrożenie

- **FastAPI:** `POST /predict` (tekst → intencja, id, wersja modelu), `GET /health` (status +
  `MODEL_VERSION`); walidacja wejścia (Pydantic), test kontraktu odpowiedzi w pytest.
- **Metryki (dodatek W5, 29–30.10):** `GET /metrics` w formacie Prometheusa — liczba i czas żądań
  według metody, trasy i statusu, predykcje według intencji, rozkład `score`. Rozkład predykcji
  i score to sygnał zmiany ruchu, nie miara jakości. Adnotacje `prometheus.io/*` w Deployment;
  serwera Prometheus w klastrze nie ma. W CI krok po smoke sprawdza `/metrics` przez Service.
- **Docker:** obraz CPU z wpieczonym artefaktem; jeden wybrany model. Config i mapa etykiet
  w obrazie: `COPY configs/ /app/configs/` + `ENV BANKING77_CONFIG=/app/configs/config.yaml`
  (settings liczą ścieżki od położenia configu). DistilBERT z torchem z indeksu CPU, nie
  z PyPI (na Linuxie koło z PyPI zawiera CUDA). Rozmiar (wersja `v1.0-submission`): 2,21 GB na
  runnerze CI (`docker image ls`); lokalnie Docker Desktop pokazuje 3,18 GB zajętości dysku przy
  822 MB skompresowanej treści — inna księgowość magazynu obrazów. Budowa od zera w CI: 98 s.
  **Poprawka po feedbacku prowadzącego (8.11.2026):** zależności treningu i analizy (pandas, pyarrow,
  optuna, mlflow, matplotlib, seaborn) przeniesione z `dependencies` do grupy `train` (PEP 735;
  `uv sync` instaluje ją domyślnie, więc komendy lokalne i CI bez zmian); obraz serwujący budowany
  z `--no-default-groups` — lokalnie **3,19 GB → 2,49 GB** (`docker image inspect`: 3 185 001 276 →
  2 493 569 815 B), smoke `[OK]`, w obrazie brak pandas/mlflow/optuna. Ten sam Dockerfile ma drugi cel
  `train` (środowisko treningowo-analityczne na CPU, 4,74 GB lokalnie). Na runnerze CI (run
  `deploy-kind` 42502188218, PR #10, 8.11): **1,71 GB wobec 2,21 GB** przed zmianą, budowa od zera
  ok. 56 s wobec 98 s; smoke `[OK]`.
- **kind w CI (obowiązkowe):** workflow GitHub Actions: `docker build` → `kind create cluster`
  → `kind load docker-image` → `kubectl apply` (Deployment + Service) → `kubectl rollout status`
  → **test HTTP** `/health` i `/predict` → teardown klastra. Dokumentujemy, że to klaster
  testowy; dowodem wdrożenia są manifesty, logi workflow i test HTTP, nie żywy URL.
  Pierwszy zielony run na `main`:
  [39138227351](https://github.com/MarcinPrDev/banking77-intent-ml/actions/runs/39138227351)
  (18.10, 6 min 28 s). Manifesty (W4): `k8s/deployment.yaml` (2 repliki, sondy startowa/gotowości/życia na `/health`,
  requests 500m CPU / 512Mi, limit pamięci 1Gi), `k8s/service.yaml` (NodePort 30080),
  `k8s/kind/cluster.yaml` (konfiguracja klastra kind, węzeł przypięty digestem).
- **GKE (stretch #4, wykonane 22.10 ręcznie, poza CI):** ten sam obraz i te same manifesty na GKE
  Autopilot w `europe-central2`; obraz z Artifact Registry przypięty digestem, test przez port-forward,
  klaster żył ok. 14 min i został usunięty tego samego wieczoru. Koszt z rozliczeń GCP: 0,10 zł, do
  zapłaty 0,00 zł. Przebieg, czasy, uprawnienia i log: [docs/GKE.md](docs/GKE.md).

- **Sekrety:** `GROQ_API_KEY` tylko w `.env` (poza gitem) / GitHub Secrets; w kodzie i configu
  żadnych kluczy.

## Dlaczego bez Sparka i CV

Dane to **~1 MB** (13 083 krótkich zdań; mediana 10 słów w train i 9 w test, maksimum 79 słów).
Każdy krok — wczytanie, TF-IDF, embeddingi, nawet fine-tuning — mieści się w pamięci jednej maszyny;
Spark ETL dodałby infrastrukturę bez uzasadnienia w objętości danych („big data w chmurze" — Spark
ETL, Dataproc, CI/CD — zrealizowano w siostrzanym repo
[us-accidents-etl](https://github.com/MarcinPrDev/us-accidents-etl); to repo jest prywatne —
ćwiczenie z kursu wzorowane na materiałach prowadzącego, dostęp na prośbę).
Zadanie jest czysto tekstowe — CV nie ma tu sensownego zastosowania. Zakres powtórki
kursu: Python, klasyczne ML, OOP/testy, eksperymenty (MLflow), DL/NLP (transformery), API,
Docker, Kubernetes, CI/CD; LLM jako stretch.

## Wymagania projektu końcowego — gdzie są spełnione

Punkty *End-to-end* z dokumentu kursu „Wszystko o projekcie końcowym” (SAGES, gr. 06.12.2025)
i miejsca, w których projekt je realizuje:

| Wymaganie | Gdzie w repo |
|---|---|
| 1a. Analiza rozkładów cech | [Dane](#dane) — liczność klas w train i teście, długości tekstów (znaki, słowa, tokeny); [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) §2 i §4 |
| 1b. Analiza braków danych i ich powodów | [Dane](#dane) — zero pustych tekstów i etykiet, pochodzenie i czyszczenie nieudokumentowane; notebook 01 §1 |
| 1c. Analiza wartości odstających | [Dane](#dane) — maksima długości i teksty skrajne; notebook 01 §4 i §6 (teksty skrajne i niejednoznaczne) |
| 1d. Utworzenie nowych cech | TF-IDF (słowa lub n-gramy znakowe) i embeddingi MiniLM — [Modele i budżet strojenia](#modele-i-budżet-strojenia); słownictwo: notebook 01 §7 |
| 2a. Przetestowanie kilku algorytmów | Dummy, TF-IDF + LR, MiniLM + kNN, DistilBERT, LLM zero-shot (stretch) — [Wyniki](#wyniki) |
| 2b. Optymalizacja hiperparametrów | Optuna (TF-IDF + LR), siatka (kNN), LR × epoki (DistilBERT) — [Modele i budżet strojenia](#modele-i-budżet-strojenia); [Wrażliwość na budżet strojenia](#wrażliwość-na-budżet-strojenia) |
| 3a. Programowanie obiektowe, wzorce, SOLID | [Architektura (wzorce i SOLID)](#architektura-wzorce-i-solid) |
| 3b. Pipeline'y do przetwarzania danych | pipeline'y sklearn i przepływ artefaktów — [Architektura](#architektura-wzorce-i-solid), [Przepływ artefaktów modelu](#przepływ-artefaktów-modelu) |
| 3c. Testy jednostkowe | `tests/` (pytest w CI: [`ci.yaml`](.github/workflows/ci.yaml)), w tym strażnik protokołu [`tests/test_contract.py`](tests/test_contract.py) |
| 4a. API, konteneryzacja, wdrożenie na klaster, CI/CD | FastAPI ([`src/banking77_intent_ml/api/`](src/banking77_intent_ml/api/)), [`Dockerfile`](Dockerfile), [`k8s/`](k8s/), workflow [`deploy-kind.yaml`](.github/workflows/deploy-kind.yaml) z testem HTTP — [Wdrożenie](#wdrożenie) |
| 4b. Lokalnie lub na GCP | kind lokalnie i w CI; GKE Autopilot (stretch, 22.10) — [Wdrożenie](#wdrożenie) |

## Wyniki

### Wyniki na walidacji

W1–W2; zamrożona walidacja z manifestu — do wyboru konfiguracji, nie do porównań z literaturą;
wszystkie liczby z MLflow:

| Model | Konfiguracja (zamrożona) | Acc (val) | Macro-F1 (val) | N-train / N-val | Budżet HPO | Czas treningu |
|---|---|---:|---:|---:|---:|---:|
| Dummy `most_frequent` | — | 1,87% | 0,05% | 8 504 / 1 499 | 0 | < 0,01 s |
| Dummy `stratified` (seed 42) | — | 1,13% | 1,04% | 8 504 / 1 499 | 0 | < 0,01 s |
| TF-IDF + LR | W1, przed HPO: wartości domyślne sklearn 1.9 (unigramy, `C` = 1,0, bez wag klas) | 84,39% | 83,89% | 8 504 / 1 499 | 0 (HPO w W2) | 0,50 s |
| TF-IDF + LR | **zamrożona (HPO W2, próba #12):** znaki `char_wb` (2,4), `min_df` 2, `sublinear_tf`, `C` ≈ 49,7, wagi `balanced` | 89,26% | 89,32% | 8 504 / 1 499 | 16 (Optuna TPE) | 2,95 s |
| MiniLM + kNN | **zamrożona (siatka W2):** all-MiniLM-L6-v2 @ `1110a243` (zamrożone embeddingi, cosinus) + kNN k = 1, `uniform` | 91,06% | 91,27% | 8 504 / 1 499 | 8 (siatka 4 × 2) | 7,7 s (kodowanie train, GPU) + 0,01 s (kNN) |
| DistilBERT | **zamrożona (W2):** distilbert-base-uncased @ `12040acc`, LR 6e-5, epoka 4 z 5 (wariant 4; reszta jak Qi 2021: batch 32, `max_len` 128, warmup 10%, wd 0,01) | 90,99% | 90,76%[^distilbert] | 8 504 / 1 499 | 25 (5 LR × 5 epok) | 22,8 s (5 epok z oceną co epokę, GPU) |

[^distilbert]: 90,76% to `val_macro_f1` z epoki 4 runu `f37e7471`; główne `macro_f1` tego runu to 90,78% z epoki 5. Różnica jest poniżej 0,001 (0,1 punktu procentowego), więc reguła remisu wybrała wcześniejszą epokę.

### Rozrzut DistilBERT po seedach

Stretch 1, 18.10; decyzje autora przed wynikiem: 5 nowych seedów + kontrolnie 42, zamrożona
konfiguracja, wynik po epoce 4 bez wyboru epoki per seed; ta sama zamrożona walidacja
8 504 / 1 499, test nietknięty; liczby z MLflow, badanie `seed_study=distilbert_val_seeds`,
commit `f93d533`:

| Seed | Acc (val) | Macro-F1 (val) | Run |
|---|---:|---:|---|
| 42 — kontrola (odtwarza run W2 `f37e7471` co do bitu) | 90,99% | 90,76% | `e2de28e4` |
| 1 | 91,06% | 90,80% | `b5601e4a` |
| 2 | 91,33% | 91,14% | `b2d3378f` |
| 3 | 91,26% | 91,02% | `72aa5341` |
| 4 | 91,26% | 91,12% | `4ca957e5` |
| 5 | 90,93% | 90,80% | `5adb3736` |
| **6 seedów: średnia (odch. std)** | **91,14% (0,17)** | **90,94% (0,17)** | |
| min – max | 90,93–91,33% | 90,76–91,14% | |

Odniesienie: MiniLM + kNN (deterministyczny, bez seeda) — macro-F1 91,27% na tej samej
walidacji. Trening jednego seeda 22–25 s (GPU).

Zmiana seeda nie odwróciła rankingu macro-F1 na tej walidacji: MiniLM + kNN pozostał wyżej
we wszystkich sześciu przebiegach. To ocena losowości treningu na jednym podziale, nie dowód
przewagi na nowych danych.

### Wrażliwość na budżet strojenia

Macro-F1 na tej samej zamrożonej walidacji 1 499 zdań, jeden seed; liczby z MLflow:

| Ścieżka | Bez strojenia | Mniejszy budżet | Pełny budżet (zamrożona) | Zysk ze strojenia |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 0,8389 — domyślne sklearn (W1) | 0,8833 — 8 pierwszych prób | 0,8932 — 16 prób Optuny | +0,0543 |
| MiniLM + kNN | 0,9006 — k = 5, `uniform` (domyślne sklearn) | — | 0,9127 — siatka 8 punktów | +0,0121 |
| DistilBERT | 0,9022 — konfiguracja Qi 2021 (LR 4e-5, 5 epok) | — | 0,9076 — 25 kandydatów (5 LR × 5 epok) | +0,0054 |

„8 pierwszych prób” to próby 0–7 tego samego badania Optuny (najlepsza: próba 6); TPE losuje
pierwsze 10 prób, więc ta kolumna to w praktyce przeszukiwanie losowe. Pełny budżet DistilBERT
to wartość wybrana regułą remisu (epoka 4: 0,9076; epoka 5: 0,9078).

**Zysk ze strojenia (W2):** najwięcej na tej walidacji zyskał TF-IDF+LR (+0,0543 macro-F1);
drugie osiem prób dodało 0,0099. MiniLM+kNN zachował najwyższy wynik, a DistilBERT poprawił
konfigurację Qi 2021 o 0,0054. Przy różnych punktach startowych, budżetach i jednym seedzie
nie jest to porównanie pełnych możliwości modeli.

**Ograniczenie DistilBERT:** zwycięskie LR 6e-5 leży na brzegu siatki, a macro-F1 dla LR
2e-5–4e-5 nadal rosło w 5. epoce, więc ten budżet nie wyczerpuje możliwości modelu.

### Koszt na CPU

W2, pomiar `cpu_cost_w2` z 14.10, kod z commita `aa210d7`; batch=1 przez wspólny interfejs
predyktora, tylko CPU; liczby z MLflow:

| Ścieżka | Wątki | Latencja p50 | Latencja p95 | Zimny start: mediana (max) | Artefakt | Run MLflow |
|---|---:|---:|---:|---:|---:|---|
| TF-IDF + LR | 1 | 0,28 ms | 0,42 ms | 0,85 s (0,86 s) | 5,0 MB | `ee671463` |
| TF-IDF + LR | 8 | 0,27 ms | 0,36 ms | 0,85 s (0,86 s) | 5,0 MB | `7fe52480` |
| MiniLM + kNN | 1 | 9,57 ms | 13,83 ms | 4,40 s (4,45 s) | 104,7 MB | `48ddd8ae` |
| MiniLM + kNN | 8 | 7,58 ms | 8,97 ms | 4,38 s (4,50 s) | 104,7 MB | `53dd2727` |
| DistilBERT | 1 | 17,52 ms | 31,27 ms | 3,20 s (3,24 s) | 268,8 MB | `36c5c183` |
| DistilBERT | 8 | 7,28 ms | 9,91 ms | 3,13 s (3,13 s) | 268,8 MB | `1312a257` |

Metodyka (decyzje autora 14.10, `evaluation.cost` w configu): latencja po rozgrzaniu = 20
predykcji rozgrzewki, potem 200 zdań z walidacji (seed 42, te same dla każdej ścieżki) × 3
przebiegi, `predict_proba` na jednym tekście, percentyle z `numpy.percentile`. Zimny start =
mediana i maksimum z 10 nowych procesów: import kodu + `load` + pierwsza predykcja, bez startu
interpretera; pliki modeli były już w cache systemu plików. Zimny start to głównie import kodu
(~0,78 s na każdej ścieżce, sklearn) i `load` (TF-IDF 0,07 s, MiniLM+kNN 3,60 s z rozgrzaniem
enkodera, DistilBERT 2,31–2,34 s). 8 wątków = rdzenie fizyczne tej maszyny (AMD Zen 5,
Windows 11). Artefakt = cały katalog `models/<nazwa>`; dla MiniLM+kNN z wagami enkodera
z cache Hugging Face (91,6 MB). Bez narzutu HTTP z API.

**Jakość a koszt (W2, pomiary `cpu_cost_w2` w MLflow):** TF-IDF+LR ma najniższą latencję,
najmniejszy artefakt i najkrótszy zimny start, kosztem niższego macro-F1.
MiniLM+kNN wygrywa jakością; DistilBERT ma niższe macro-F1 i większy artefakt, ale krótszy
zimny start i podobną latencję po rozgrzaniu przy 8 wątkach CPU.
Przy k = 1 `score` kNN w API zawsze wynosi 1,0 — nie informuje o pewności predykcji.

TF-IDF + LR: dwa runy tej samej konfiguracji (`89d5ccf6`, `bb347398` — ten drugi dał artefakt
w `models/`) mają identyczne metryki, co przy okazji potwierdza determinizm treningu.

**Interpretacja (autor, 7.10):** TF-IDF + LR uzyskał accuracy 84,39% i macro-F1 83,89% na 1 499
zdaniach walidacji wydzielonej z train, po treningu na 8 504 zdaniach; literaturowe 92,1%
DistilBERT dotyczy testu 3 080 zdań po treningu na pełnych 10 003, więc wyniki nie są
bezpośrednio porównywalne. Niewielka różnica między metrykami nie wyklucza słabości pojedynczych
klas: `virtual_card_not_working` ma najniższe F1 (50,0%) i recall (33,3%) przy precision 100% —
model wskazał tę klasę dwukrotnie, oba razy trafnie, ale pozostałe cztery przykłady przypisał
innym klasom; przy sześciu przykładach jedna pomyłka zmienia recall o 16,7 punktu procentowego.

### Tabela końcowa

W3, test dotknięty raz; koszt finalnych artefaktów po reficie; N-train 10 003 dla wszystkich
ścieżek; wszystkie liczby z MLflow / wykonanego kodu:

| Model | Acc (test 3 080) | Acc (3 055) | Macro-F1 (3 080) | Macro-F1 (3 055) | Czas treningu (10 003) | Artefakt | Latencja CPU p50 / p95 | Zimny start (1 w. / 8 w.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 1,30% (arytmetyka) | — | — | — | 0 | — | — | — |
| TF-IDF + LR | 90,81% | 90,77% | 90,83% | 90,78% | 4,9 s (CPU) | 5,2 MB | 0,28 / 0,43 ms (1 w.) · 0,27 / 0,42 ms (8 w.) | 0,85 s / 0,85 s |
| MiniLM + kNN | 92,05% | 92,01% | 92,03% | 91,99% | 3,0 s (GPU) | 107,0 MB | 12,16 / 16,83 ms (1 w.) · 9,26 / 10,89 ms (8 w.) | 4,69 s / 4,69 s |
| DistilBERT | 93,02% | 92,96% | 93,05% | 93,00% | 25,5 s (GPU) | 268,8 MB | 17,48 / 31,42 ms (1 w.) · 8,22 / 11,45 ms (8 w.) | 3,30 s / 3,19 s |

Źródła (MLflow): ocena końcowa — runy `0c08edb4`, `04ec980a`, `fb0d0bde` (`split=test`,
commit `6cae45d`; kolumnę macro-F1 (3 055) dopisano 28.10 z metryki `clean_macro_f1` tych
samych runów, bez ponownej oceny); refit na 10 003 — runy `f7e23471`, `4509d9d4`, `5b883a97` (`final_refit`,
commit `d809c57` = tag `experiments-frozen`); koszt — badanie `cpu_cost_final`, po dwa runy
na ścieżkę (1 i 8 wątków: `4e0fe1ce`/`9b7c4505`, `a8276f36`/`bea6c48a`, `56a7fbb4`/`0ec07580`),
metodyka jak w tabeli *Koszt na CPU*. Runy zapisały hashe commitów sprzed przepisania historii
14.10; odpowiedniki — artefakt `provenance/git-commit-map-2026-10-14.json` w każdym runie.
Czas treningu = `fit` adaptera na 10 003 zdaniach z wczytaniem wag lub enkodera (GPU: RTX 5080).
Artefakt MiniLM + kNN = indeks 10 003 wektorów + wagi enkodera z cache Hugging Face.

**Interpretacja oceny końcowej (14.10, refit na 10 003):**

- DistilBERT najlepszy w tym przebiegu: accuracy 93,02%, macro-F1 93,05%.
  Przewaga accuracy: 0,97 p.p. nad MiniLM+kNN i 2,21 p.p. nad TF-IDF+LR.
- Protokół Full: accuracy o 0,92 p.p. wyższe niż DistilBERT u Qi i in. (92,1%)
  i o 0,28 p.p. niższe niż BERT-base (93,3%). Inny LR, liczba epok i jeden seed;
  bez wniosku o powtarzalnej przewadze nad wynikiem z publikacji.
- Na walidacji prowadził MiniLM+kNN, na teście DistilBERT. Inny zbiór oceny i więcej danych
  treningowych po reficie; brak podstaw do przypisania zmiany rankingu samemu refitowi.
- Po wyłączeniu 25 nakładek: spadek accuracy najwyżej o 0,06 p.p., ranking bez zmian.
  Mały wpływ wykrytego pokrycia train/test na wynik ogólny.

Źródło: runy `fb0d0bde`, `04ec980a`, `0c08edb4` z commita `6cae45d`; jedna ocena na ścieżkę.
Wyniki na 385 przykładach — niżej, razem z LLM (stretch 2, 20.10).
O wyborze modelu po teście i jego konsekwencjach — [Odpowiedź na pytanie badawcze](#odpowiedź-na-pytanie-badawcze) i [ADR-0003](docs/adr/0003-wybor-modelu-do-wdrozenia.md).

### Istotność różnic accuracy (3 080)

Dodatek z 29–30.10. Prerejestracja analizy w commicie `0fe484a`, zapisanym przed obliczeniami.

Analiza zapisanych predykcji, bez ponownego treningu i predykcji. Dokładny dwustronny test
McNemara, korekta Holma dla trzech par, próg 0,05. Parowany bootstrap: 10 000 losowań,
seed 42, 95% przedziały percentylowe. Parametry z `evaluation.significance` i `models.seed`.

| Para (A − B) | Różnica accuracy (p.p.) | 95% przedział bootstrapu (p.p.) | p po Holmie | Istotna przy 0,05 |
|---|---:|---:|---:|---|
| DistilBERT − MiniLM+kNN | +0,97 | [0,03; 1,92] | 0,050315 | nie |
| DistilBERT − TF-IDF+LR | +2,21 | [1,23; 3,15] | 0,00001598 | tak |
| MiniLM+kNN − TF-IDF+LR | +1,23 | [0,19; 2,27] | 0,042138 | tak |

Oba modele semantyczne mają istotnie wyższe accuracy niż TF-IDF+LR na tym teście.
DistilBERT ma o 30 błędów mniej niż kNN, ale p po Holmie = 0,050315 jest większe niż 0,05.
To nie dowód równej jakości. Przedział bootstrapu tej pary nie obejmuje zera, lecz jest
przybliżony i liczony osobno dla pary, bez korekty wielokrotnych porównań. To inna procedura
niż dokładny McNemar; dla tej pary korekta Holma nie zmieniła p.
O istotności rozstrzyga wcześniej ustalony test z korektą Holma.

Źródło: MLflow `546ebc7e1d2b4823b975964549568892`, tag `analysis=significance`, artefakt
`significance.csv`; predykcje z runów `fb0d0bde`, `04ec980a`, `0c08edb4`.
Run zapisał commit `b6d938b`; poprawka wyboru runów w loaderze została zapisana później
jako `8120316`, więc hash runu nie obejmuje tej poprawki.

### Kalibracja score na teście (3 080)

Dodatek z 31.10. Prerejestracja `2b81dcf`, przed obliczeniami: cztery modele,
15 przedziałów równej szerokości na [0, 1]. Przedziały domknięte z prawej `(a, b]`,
pierwszy obejmuje także 0. Analiza zapisanych predykcji, bez ponownej predykcji,
dopasowania kalibracji ani zmiany wyboru modelu.

Pewność to `score` wybranej klasy, a trafienie to zgodność z etykietą.
ECE: średnia bezwzględna różnica między accuracy a średnią pewnością w przedziałach,
ważona ich licznościami. Niższa wartość oznacza lepszą zgodność w tym pomiarze.

| Model | N | Accuracy | Średnia pewność | ECE (skala 0–1) |
|---|---:|---:|---:|---:|
| DistilBERT | 3 080 | 0,9302 | 0,9355 | 0,0096 |
| TF-IDF + LR | 3 080 | 0,9081 | 0,8763 | 0,0332 |
| Basal 4,5B v2 | 3 080 | 0,8039 | 0,8894 | 0,0908 |
| Basal ultra v2 | 3 080 | 0,8481 | 0,9295 | 0,0814 |

Źródło: MLflow `5203c896710a4ddd803958e0935be1cb`, tag `analysis=calibration`,
artefakty `calibration_summary.csv` i `calibration_bins.csv`.

![Kalibracja score czterech modeli na 3 080 zdaniach testu](docs/figures/calibration-reliability-test.png)

Góra: słupki — accuracy w przedziale, czerwone kreski — średnia pewność (kreska nad słupkiem:
pewność zawyżona, poniżej jego szczytu: zaniżona), linia przerywana — idealna kalibracja.
Dół: liczności w skali logarytmicznej. Wykres z tabel CSV runu (`calibration_report.py --redraw`).
DistilBERT ma najniższe ECE w tym pomiarze. TF-IDF+LR średnio zaniża pewność względem
accuracy, a oba warianty Basala ją zawyżają. Kalibracja dostarczona z Basalem nie zapewniła
zgodności score z częstością trafień na BANKING77. Wyższa accuracy ultra niż 4,5B nie usuwa
tego problemu. Niskie ECE DistilBERT nie gwarantuje poprawnej pewności dla każdej klasy ani OOD.

ECE zależy od liczby i granic przedziałów. Mierzymy tylko score wybranej klasy, nie cały
rozkład po 77 klasach. Brak przedziałów ufności; różnic ECE nie uznajemy za potwierdzone
statystycznie. kNN pominięty zgodnie z prerejestracją (przy k = 1 score zawsze 1,0),
LLM — z powodu braku score w zapisanym runie testowym.
Temperature scaling pozostaje w *Dalszych pracach*, na nowych oznaczonych danych.
Ten pomiar opisuje zamrożone modele; nie służy dopasowaniu kalibracji ani progów na teście.

### LLM zero-shot na 385 wiadomościach testu

Stretch 2, 20.10; decyzje autora: `groq/openai/gpt-oss-20b`, temperatura 0, `reasoning_effort`
low; odpowiedź = dokładnie jedna nazwa etykiety, inaczej błąd formatu liczony jako pomyłka; prompt
dobrany na walidacji, test raz. Te same 385 wiadomości (5 na klasę, wybrane w W1) dla wszystkich
ścieżek; metryki `llm385_*` z MLflow:

| Ścieżka | Acc (385) | Macro-F1 (385) | Błędy formatu | Koszt i latencja | Run |
|---|---:|---:|---:|---|---|
| DistilBERT | 91,95% | 92,02% | — | tabela końcowa (CPU) | `fb0d0bde` |
| MiniLM + kNN | 91,69% | 91,57% | — | tabela końcowa (CPU) | `04ec980a` |
| TF-IDF + LR | 91,17% | 91,12% | — | tabela końcowa (CPU) | `0c08edb4` |
| LLM zero-shot (prompt v2) | 80,78% | 79,67% | 3 | 0,0213 USD za 385 zapytań; p50 0,15 s / p95 0,65 s przez API | `219ab1e3` |

LLM zero-shot był tani w użyciu (całe badanie około 0,064 USD), ale uzyskał niższe accuracy
i macro-F1 niż wszystkie trzy modele uczone na BANKING77. Błędy formatu to tylko 3 z 74
pomyłek, więc główny problem to wybór intencji. DistilBERT wyprzedził kNN o jedną poprawną
odpowiedź; nie uznajemy tej różnicy za dowód stabilnej przewagi. Wniosek o LLM dotyczy
badanego modelu z promptem v2, nie wszystkich modeli zero-shot.

Dobór promptu na walidacji (ta sama próbka 385 zdań, 5 na klasę, seed `split.random_state`):
v1 — accuracy 79,74%, macro-F1 79,71%, 2 błędy formatu (run `526e5400`); **v2** — 81,30% /
81,47%, 4 błędy (run `081d183e`); zamrożony v2 wg macro-F1 (decyzja autora 20.10). Tokeny na
teście: 215 396 wejścia / 17 285 wyjścia (ok. 560 na zapytanie — lista 77 etykiet w promptcie).
Latencja LLM obejmuje sieć i kolejkę API, więc nie porównuje się wprost z pomiarami CPU modeli
lokalnych. Ekspozycja na benchmark — zob. protokół pkt 8 i *Ograniczenia*.

### Basal zero-shot — dodatek z 30–31.10

Skąd ten dodatek: w drugiej połowie października 2026 r. o Basalu było w Polsce głośno po rankingu
JevBench (v1.6.28, 17.10.2026), który ocenia trafność i kalibrację decyzji, nie pisanie tekstu.
Wariant `basal-omni-1.0-ultra` znalazł się tam w czołówce modeli z otwartymi wagami: według
doniesień od 1. do 4. miejsca, zależnie od zestawienia
([ITwiz, 21.10.2026](https://itwiz.pl/polski-model-ai-wyprzedzil-rozwiazania-openai-i-microsoftu/),
[Bezprawnik, 25.10.2026](https://bezprawnik.pl/basal-ai/)). Badanie 1 dotyczy modelu 4,5B
(inny wariant), badanie 2 — właśnie wariantu ultra.
Basal to polski model decyzyjny (Remigiusz Kinas; wersja 4,5B dostrojona z
`speakleash/Bielik-4.5B-v3.0-Instruct` według NOTICE pakietu): wybiera jedną z podanych opcji
i zwraca rozkład prawdopodobieństw.
Klasyfikacja intencji ma właśnie taki kształt, więc Basal nadawał się do sprawdzenia bez dostrajania.
Dodatek jest hobbystyczny i wykracza poza wymagania zaliczenia: powstał z ciekawości,
żeby sprawdzić nowy model w praktyce i dowiedzieć się jak najwięcej.

Rozszerzenie po głównym badaniu, bez zmiany wybranego modelu do wdrożenia. Pełny opis obu badań (prerejestracje, środowisko, wybór wariantu opisów v1/v2 na walidacji, latencje, koszty): [docs/BASAL.md](docs/BASAL.md); środowisko ultra na H100: [docs/basal-ultra.md](docs/basal-ultra.md).

#### Basal 4,5B — badanie 1, 30–31.10

Model `Remek/basal-2.0-4.5B`, rewizja `e5ba914`, silnik `basal-serve` 2.4.1.
Wybór jednej z 77 podanych intencji, bez swobodnego generowania odpowiedzi.
Zero-shot: bez dostrajania wag w tym projekcie i bez przykładów w żądaniu.
Ekspozycja Basala na BANKING77 podczas wcześniejszego treningu nieustalona.
Wspólna instrukcja dla obu wariantów; v1 to nazwy etykiet zapisane jako frazy,
a v2 — krótkie opisy. Niejednoznaczne nazwy sprawdzone wyłącznie na części treningowej.
Kalibracja dostarczona z modelem, bez własnego dopasowania. Jej jakość na BANKING77 zmierzona
na zapisanych predykcjach — zob. [Kalibracja score](#kalibracja-score-na-teście-3-080).

Zamrożony wybór: `basal_4_5b` / `v2`, commit `4a87d48` przed testem.
Test 31.10: jeden przebieg na 3 080 zdaniach, z tych samych predykcji także wynik na
podzbiorze 385. Run `8cdd3c0f`, tagi `split=test`, `stage=extension`; zapisane predykcje
i tabele per klasa. Czas około 57 min; p50 żądania HTTP 1 089,8 ms, p95 1 144,1 ms.

| Model | Accuracy (3 080) | Macro-F1 (3 080) | Accuracy (385) | Macro-F1 (385) |
|---|---:|---:|---:|---:|
| DistilBERT | 0,9302 | 0,9305 | 0,9195 | 0,9202 |
| MiniLM + kNN | 0,9205 | 0,9203 | 0,9169 | 0,9157 |
| TF-IDF + LR | 0,9081 | 0,9083 | 0,9117 | 0,9112 |
| Basal ultra v2 (zero-shot) | 0,8481 | 0,8441 | 0,8286 | 0,8107 |
| LLM v2 (zero-shot) | — | — | 0,8078 | 0,7967 |
| Basal 4,5B v2 (zero-shot) | 0,8039 | 0,7966 | 0,7870 | 0,7787 |

Źródło: metryki `test_*` i `llm385_*` w MLflow. Basal 4,5B: `8cdd3c0f`, DistilBERT: `fb0d0bde`,
MiniLM+kNN: `04ec980a`, TF-IDF+LR: `0c08edb4`, LLM: `219ab1e3`, Basal ultra: `b855251a`.
LLM oceniony tylko na 385 wiadomościach; porównania w tej części dotyczą tych samych wierszy.

Basal 4,5B uzyskał niższe accuracy i macro-F1 niż wszystkie trzy modele uczone na BANKING77,
zarówno na 3 080, jak i na 385 zdaniach. W tym badaniu brak dostrajania nie zrekompensował
niższej jakości i kosztu predykcji na GPU. Wniosek dotyczy modelu 4,5B z opisami v2 w trybie
`eager`, nie całej rodziny Basal ani najlepszego możliwego czasu tego silnika.

#### Basal ultra — badanie 2, 31.10

Osobna prerejestracja `9f36a77`, przed obliczeniami. Model `Remek/basal-omni-1.0-ultra`
(31B), rewizja `6d5ba67`, na wynajętej karcie H100 80 GB w RunPod.
Te same instrukcje, opisy v1/v2 i próbka walidacji 385 zdań co dla 4,5B; zgodność skrótów
zapisana w MLflow. Bez dostrajania wag w projekcie; wcześniejsza ekspozycja na BANKING77
pozostaje nieustalona. Szczegóły środowiska i przebiegu: [notatka](docs/basal-ultra.md).

Ultra trafiła 2 612 z 3 080 wiadomości, o 136 więcej niż 4,5B (2 476).
Nadal niższe accuracy i macro-F1 niż wszystkie trzy modele uczone na BANKING77.
Na tych samych 385 zdaniach: ultra 319 trafień, LLM v2 311, Basal 4,5B 303,
a modele uczone 351–354. Ultra uzyskała tu najwyższe wyniki spośród badanych ścieżek
zero-shot. To wynik tej próbki, nie dowód trwałej przewagi nad LLM.
Dla porównań z Basalem nie zaplanowano z góry ani nie wykonano testów istotności.

Koszt z rozliczeń RunPod: 9,90 USD, w tym GPU 9,83 USD i dyski 0,07 USD.
Pod działał około 2 h 28 min, w tym około 1 h 18 min oczekiwania na wpis frozen.
Kwota obejmuje przygotowanie, walidację, oczekiwanie i test. Sam test trwał 48 min,
czyli około 3,2 USD za GPU według stawki 3,99 USD/h — to szacunek, nie osobna pozycja rachunku.
Pod usunięty po teście, tunel zamknięty.

## Analiza błędów (W3)

Notebook: [`notebooks/02_analiza_bledow.ipynb`](notebooks/02_analiza_bledow.ipynb) — najczęstsze pary
pomyłek i recall per klasa z oceny końcowej (same agregaty, bez czytania zdań testowych), porównanie
błędów trzech ścieżek po `row_id`, przegląd 20 błędów modeli W2 na walidacji; etykiety bez zmian.

![Macierz pomyłek DistilBERT na teście 3 080 zdań, wiersze znormalizowane](docs/figures/distilbert-confusion-test.png)

Macierz pomyłek DistilBERT na teście (3 080 zdań, wiersze znormalizowane, więc przekątna
to recall klasy; zapisany wynik komórki notebooka 02, §1, bez tekstów zdań).

Główna trudność to rozróżnianie bliskich intencji. Krótkie zdania często nie zawierają
wystarczającego kontekstu.

Modele semantyczne poprawiają wynik ogólny względem TF-IDF. Nadal potrafią jednak skupić się
na pobocznym fragmencie zdania.

DistilBERT popełnia najmniej błędów. Wysoki score nie gwarantuje poprawności, a score kNN
przy k równym 1 nie informuje o pewności odpowiedzi.

## Odpowiedź na pytanie badawcze

Częściowo: w tym projekcie poprawa jakości względem TF-IDF+LR uzasadnia dla mnie większy
koszt DistilBERT, ale wobec MiniLM+kNN wyniki nie dają jednoznacznej odpowiedzi.
Na teście 3 080 zdań: accuracy 93,02% wobec 92,05% MiniLM+kNN i 90,81% TF-IDF+LR.
To odpowiednio 30 i 68 błędów mniej.
(Macro-F1 na tym samym teście: odpowiednio 93,05%, 92,03% i 90,83%.)
Zgodnie z ADR-0003 pierwszeństwo ma jakość; progu opłacalności nie ustalono.

Cena wyboru: większy artefakt i wyższe p95 predykcji na CPU przy jednym wątku.
Dla DistilBERT, MiniLM+kNN i TF-IDF+LR odpowiednio: 268,8 / 107,0 / 5,2 MB
oraz p95 31,42 / 16,83 / 0,43 ms.
Pomiar po rozgrzaniu, batch=1, 200 zdań z dawnej walidacji, trzy przebiegi; bez narzutu HTTP.
Zimny start przy jednym wątku krótszy niż dla kNN: 3,30 s wobec 4,69 s,
ale dłuższy niż dla TF-IDF+LR (0,85 s); mediana z 10 nowych procesów.
Sam refit, bez strojenia, na 10 003 zdaniach: 25,5 s dla DistilBERT i 3,0 s dla MiniLM+kNN na GPU,
4,9 s dla TF-IDF+LR na CPU. Różny sprzęt, więc bez prostego przeliczenia czasu na koszt.

To kompromis projektowy, nie dowód opłacalności w banku.
Argument jakościowy mocniejszy wobec TF-IDF+LR: DistilBERT wyżej na walidacji (1 499 zdań)
i teście (3 080 zdań). Wobec kNN wynik niejednoznaczny: na walidacji (1 499 zdań)
wygrywał kNN, na teście (3 080 zdań) DistilBERT.
Na teście 3 080 zdań przewaga DistilBERT nad TF-IDF+LR jest istotna po korekcie Holma,
a nad kNN nie osiąga progu 0,05 (p po Holmie = 0,050315).
DistilBERT wybrano dopiero po obejrzeniu wyników testu;
93,02% na 3 080 zdaniach nie stanowi niezależnego potwierdzenia tego wyboru.
Bez dalszego strojenia po teście i bez dowodu trwałej przewagi nad kNN.

## Ograniczenia i granice weryfikacji

- Pochodzenie i anonimizacja tekstów nieudokumentowane. Skan 13 083 tekstów: zero trafień
  prostych wzorców, bez dowodu braku PII. Szczegóły w sekcji [Dane](#dane).
- Wskazania Ying & Thomas dotyczą potencjalnych problemów etykiet, nie potwierdzonych błędów.
  Etykiety pozostawione bez zmian.
- Test 3 080 zdań jest zrównoważony: 40 przykładów na klasę. Jedna pomyłka to 2,5 p.p.
  recallu klasy. Wynik nie odwzorowuje częstości rzeczywistych zgłoszeń.
  Brak czasu i identyfikatorów rozmów ogranicza kontrolę podziału danych.
  Podzbiór 3 055 służy kontroli wpływu pokrycia z train, nie porównaniu z publikacjami.
- DistilBERT oceniony na teście 3 080 zdań dla jednego seeda treningu. TF-IDF+LR i kNN
  przy zamrożonej konfiguracji są deterministyczne. Seed Optuny wpływał na dobór
  konfiguracji TF-IDF+LR; siatka kNN była pełna. Badanie sześciu seedów DistilBERT
  dotyczyło tylko walidacji (1 499 zdań), bez różnych podziałów danych.
  Nie potwierdza stabilności rankingu na teście. O wyborze po teście — w sekcji
  [Odpowiedź na pytanie badawcze](#odpowiedź-na-pytanie-badawcze).
- [Analiza istotności na 3 080 zdaniach](#istotność-różnic-accuracy-3-080): przewaga obu modeli
  semantycznych nad TF-IDF+LR istotna po Holmie; DistilBERT–kNN bez odrzucenia hipotezy
  równej accuracy (p po Holmie = 0,050315). To nie dowód równoważności modeli. Bootstrap opisuje
  niepewność próby przy ustalonych modelach, bez losowości treningu i strojenia.
  Analiza nie usuwa ograniczenia wyboru modelu po teście. Nie obejmuje podzbioru 385,
  na którym DistilBERT ma tylko jedno trafienie więcej niż kNN.
- Ograniczone i różne budżety strojenia. LR DistilBERT 6e-5 na brzegu siatki;
  krzywe walidacyjne dla niższych LR jeszcze rosły. Porównanie wybranych konfiguracji,
  nie pełnych możliwości modeli. Porównanie z literaturą tylko na teście Full
  (3 080 zdań, refit na 10 003), z zastrzeżeniem różnic LR, epok i seeda.
- System obsługuje tylko język angielski. Brak walidacji intencji spoza 77 klas (OOD)
  i rzeczywistych zgłoszeń firmy.
  Model zawsze wybiera znaną klasę. Score nie jest skalibrowaną pewnością;
  przy k=1 w kNN zawsze wynosi 1,0. Dawna walidacja weszła do refitu,
  więc kalibracja i próg przekazania sprawy człowiekowi wymagają nowych danych.
- LLM oceniony na 385 wiadomościach, tych samych co modele lokalne w tabeli porównawczej.
  Ekspozycja LLM na BANKING77 nieustalona. Wynik nie dowodzi generalizacji;
  latencja obejmuje sieć i API, więc nie jest wprost porównywalna z lokalną predykcją CPU.
- Koszt zależy od sprzętu i warunków pomiaru. Pomiary CPU bez narzutu HTTP i obciążenia
  równoległymi żądaniami; zimny start z plikami w cache systemu.
  Czasy refitu nie obejmują strojenia, a rozmiar artefaktu nie oznacza rozmiaru obrazu ani RAM.
  Zielone smoke testy potwierdzają działanie wdrożenia, nie jakość na nowych danych
  ani gotowość do pracy produkcyjnej.
- Zgodność predykcji obrazu CPU z ocenionym artefaktem sprawdzono na 50 zdaniach dawnej
  walidacji: 50 zgodnych id, różnica score najwyżej 3e-07, oraz na 14 zdaniach smoke.
  Bez ponownej oceny obrazu na teście. Odtworzenie wyników na świeżym klonie wykonano
  na tej samej maszynie; nie potwierdza zgodności na innym sprzęcie.

## Co nie zadziałało i co zrobiłbym inaczej

- Plan czasu był zbyt ciasny. Rezerwa wyczerpała się już po W1, a W4 wymagał nadgodzin.
  Następnym razem wcześniej ograniczyłbym stretch i zostawił więcej czasu na odtworzenie
  projektu oraz redakcję raportu.
- Model do wdrożenia wybrałem dopiero po wynikach testu. Następnym razem decyzja na podstawie
  walidacji i kosztów, przed otwarciem testu. Dane do kalibracji wydzielone przed refitem;
  badanie kilku seedów i niepewności różnic zaplanowane od początku.
- Analiza błędów w notebooku początkowo miała tekst w komórkach kodu i liczby bez obliczeń
  pokazujących ich źródło. Następnym razem każda tabela i wniosek od razu obok kodu,
  który je uzasadnia, oraz uruchomienie całego notebooka przed review.
- Problemy środowiska i usług zabierały czas: zależności, zmiany API bibliotek, porty Windows,
  limity Groq i brak ról konta w GCP. Wcześniej sprawdziłbym wymagania, uprawnienia i limity
  krótkim przebiegiem. Dla LLM także to, czy budżet tokenów wystarcza na niepustą odpowiedź.
- Obraz CPU zawierał zależności treningu i analizy (poprawione 8.11.2026 po feedbacku
  prowadzącego: grupa `train`, obraz 3,19 → 2,49 GB). Następnym razem oddzieliłbym
  środowisko serwowania od razu, a własną próbę z README na świeżym klonie zrobiłbym zaraz
  po pierwszym działającym wdrożeniu, nie dopiero 21–22.10. Pozwoliłoby to wcześniej
  wykryć brakujące kroki instrukcji.

## Dalsze prace

Po oddaniu projektu (decyzja C z 13.10):

- Szersze HPO dla wszystkich trzech ścieżek.
- Zagnieżdżona grupowana CV: strojenie na podziałach wewnętrznych, ocena na zewnętrznych;
  znormalizowane duplikaty zawsze razem.
- Kilka seedów, żeby sprawdzić stabilność wyników.
- Front jakość–koszt: warianty, których nie da się poprawić w jednym kryterium bez pogorszenia drugiego.

Dodatkowo (27.10):

- OOD: nowe zgłoszenia z intencjami spoza 77 klas, obok przykładów znanych intencji.
  Sprawdzenie odrzucania nieznanych spraw i przekazywania ich człowiekowi,
  z uwzględnieniem błędnego odrzucania znanych intencji. Dobór reguły i ocena na osobnych danych.
- Kalibracja score DistilBERT, np. temperature scaling, na nowych oznaczonych danych.
  Osobne części do kalibracji, wyboru progu przekazania człowiekowi i końcowej oceny.
  Ocena wykresem kalibracji i ECE; bez użycia obecnego testu ani dawnej walidacji z refitu.
  Sama kalibracja nie zastępuje sprawdzenia OOD.

Dodatkowo (31.10):

- Kolejne modele Basal: niebadane warianty, np. `mini`, oraz 11B FP8. Wariant
  `basal-omni-1.0-ultra` został już oceniony w badaniu 2. Ten sam protokół: wybór wariantu
  na walidacji 385, jeden przebieg testu dla zamrożonego wyboru. Dla 11B FP8 potrzebne GPU,
  na którym zmieszczą się tryby z grafami CUDA.

## Tooling — współpraca z AI

Projekt rozwijany z asystą **Claude (Anthropic)** w roli pair-programmera i reviewera
(plan, szkielet, code review, debugowanie). Decyzje projektowe, analiza, modelowanie
i wnioski — autor. Commity współtworzone z asystentem są oznaczone stopką
`Co-Authored-By` w historii git. ADR-y w `docs/adr/`: sekcje *Decyzja* i *Konsekwencje* pisze
autor; w ADR-0001 sekcje *Kontekst* i *Rozważane opcje* zestawił asystent z notatek researchu
(oznaczone w pliku).

## Licencja

**Kod:** MIT — plik [`LICENSE`](LICENSE).
**Dane:** BANKING77 — Casanueva, Temčinas, Gerz, Henderson, Vulić (2020), *Efficient Intent
Detection with Dual Sentence Encoders*, PolyAI; repozytorium `PolyAI-LDN/task-specific-datasets`,
licencja **CC BY 4.0**. Dane nie są przechowywane w repo — pobiera je `scripts/fetch_data.py`
z przypiętego commitu. Zmiany własne względem oryginału: wydzielenie walidacji z train (indeksy
w manifeście), normalizacja tekstu wyłącznie do audytu duplikatów; teksty i etykiety nie są
modyfikowane. Wagi modeli: all-MiniLM-L6-v2 i distilbert-base-uncased — Apache-2.0.
