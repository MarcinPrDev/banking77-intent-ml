# ADR-0002: Protokół ewaluacji — zamrożony podział, mapa etykiet, podzbiory testowe i refit przed testem

- **Status:** zaakceptowany
- **Data decyzji:** 2026-10-04 (podział i mapa), 2026-10-05 (podzbiory testowe), 2026-10-14 (refit)
- **Data akceptacji:** 2026-10-14
- **Etap:** W1–W3

> Podstawy: ustalenia sesji, manifest, config i kod. Układ według szablonu MADR i ADR-0001.

## Kontekst i problem

Trzy ścieżki, te same dane i zasady oceny. Ryzyka: duplikaty po obu stronach podziału,
różne id klas, dobór modelu pod wynik testowy. Podział i reguły przed treningiem;
decyzja o reficie przed testem. Macro-F1 na walidacji do wyboru konfiguracji;
accuracy na teście jako główna metryka raportu.

Źródła: [README — protokół](../../README.md#protokół-ewaluacji-wspólny-dla-wszystkich-modeli),
[config](../../configs/config.yaml) (`split`, `evaluation`).

## Decyzje

### 1. Zamrożony podział train/val, znormalizowane duplikaty razem

- **Opcje:** losowanie pojedynczych wierszy albo całych grup duplikatów.
- **Decyzja:** grupy razem. Klucz: NFKC → casefold → tokeny `\w+` połączone spacją.
  Normalizacja tylko do audytu i grupowania; teksty do modeli bez tych zmian.
- **Podział:** reprezentant każdej grupy, stratyfikacja po etykiecie, potem rozwinięcie grup.
  `val_size=0.15`, seed 42 z configu; wynik: 8 504 train i 1 499 val.
- **Powód:** brak wariantów tego samego zdania po obu stronach podziału.
  Te same indeksy dla wszystkich modeli; hashe w manifeście. Bez ponownego losowania.

Źródła: [manifest](../../configs/split_manifest.json), [indeksy](../../dane/splits/),
[podział](../../src/banking77_intent_ml/data/splits.py).

### 2. Grupa konfliktowa (te same teksty z różnymi etykietami)

- **Opcje:** usunięcie grupy, ręczna zmiana etykiet albo zachowanie całej grupy w train.
- **Decyzja:** oba wiersze (2002, 3160) w train przed losowaniem; etykiety
  `pending_top_up` i `top_up_reverted` bez zmian. Wykluczone wiersze: 0.
- **Powód:** brak podstaw do wyboru jednej poprawnej etykiety; zachowanie oryginalnych danych.
  Cena: sprzeczne przykłady w treningu, brak tej niejednoznacznej grupy w walidacji.

Źródła: [manifest](../../configs/split_manifest.json) (`conflict_groups`, `conflict_rule`),
[EDA](../../notebooks/01_eda.ipynb).

### 3. Mapa etykiet nazwa ↔ id

- **Opcje:** osobne mapy dla modeli albo jedna mapa projektu.
- **Decyzja:** `configs/label_map.json` dla modeli, API i ewaluacji.
  77 nazw z train; id 0–76 według `sorted()` nazw (porządek Unicode).
- **Powód:** to samo znaczenie id i kolumn wyników w każdej ścieżce.
  Bez map z innych kopii zbioru; zmiana mapy wykrywana przy odczycie artefaktu.

Źródła: [mapa](../../configs/label_map.json), [obsługa etykiet](../../src/banking77_intent_ml/data/labels.py),
[test kontraktu](../../tests/test_contract.py).

### 4. Podzbiory testowe: 3 055 bez pokrycia z train i 385 dla LLM

- **Opcje:** tylko pełny test, tylko test bez pokrycia albo pełny test z podzbiorami.
- **Decyzja:** pełne 3 080 jako główny wynik; 3 055 jako kontrola wrażliwości.
  Usunięte z podzbioru: 25 tekstów z odpowiednikiem w całym train po normalizacji.
  Podzbiór 3 055 bez porównań z publikacjami na 3 080.
- **LLM:** 385 przykładów z podzbioru 3 055; po 5 na klasę, seed 42 z configu.
  Te same przykłady dla wszystkich ścieżek, jeśli rozszerzenie LLM zostanie wykonane.
- **Powód:** kontrola wpływu pokrycia oraz ograniczony koszt LLM.
  Indeksy ustalone 5.10, przed treningiem; hashe w manifeście.
  Jeden przebieg predykcji klasyfikatora na pełnym teście; metryki podzbiorów z tych samych predykcji.

Źródła: [indeksy 3 055](../../dane/splits/test_clean_ids.json),
[indeksy 385](../../dane/splits/llm_eval_ids.json), [manifest](../../configs/split_manifest.json),
[audyt](../../src/banking77_intent_ml/data/audit.py).

### 5. Refit na 10 003 przed oceną na teście (decyzja 14.10)

- **Opcje:** modele uczone na 8 504 zdaniach albo refit trzech ścieżek na 10 003.
- **Decyzja:** refit na train + dawnej walidacji, w oryginalnej kolejności wierszy.
  Zamrożone parametry z configu, seed 42; bez nowego HPO.
- **TF-IDF+LR:** nowy słownik, IDF i klasyfikator; zamrożona próba #12.
- **MiniLM+kNN:** enkoder z przypiętej rewizji bez dostrajania;
  nowy indeks na 10 003 zdaniach, k = 1, `uniform`, cosinus.
- **DistilBERT:** od bazowych wag z przypiętej rewizji; LR 6e-5, `max_len=128`, batch 32.
  Harmonogram na 5 epok, zatrzymanie po 4.; pozostałe parametry z configu.
- **Powód:** ten sam budżet danych co w protokole Full.
  Porównanie accuracy na 3 080 z publikacjami; nadal różnice strojenia i jeden seed.
- **Cena:** brak niezależnej walidacji po reficie. Wyniki W2 zostają wynikami modeli sprzed refitu.
  Nowe artefakty w `models/final/`; stare zachowane. Ponowny pomiar kosztów finalnych modeli.
- **Kolejność:** zapis decyzji → tag `experiments-frozen` → refit → jednorazowa ocena na teście.
  Bez zmiany konfiguracji po pierwszym wyniku testowym.

Źródła: [config](../../configs/config.yaml) (`models.<nazwa>.frozen`),
[adaptery](../../src/banking77_intent_ml/models/classifiers.py),
[refit](../../src/banking77_intent_ml/models/refit.py), [CLI](../../scripts/refit_final.py).

## Konsekwencje

### Dobre

- Wspólne dane, mapa klas i metryki dla trzech ścieżek.
- Odtwarzalny podział; brak przecieku z duplikatów między train i val.
- Pełny budżet danych w ocenie końcowej; osobna kontrola pokrycia train/test.

### Złe i ryzyka

- Jeden podział i seed; brak oceny zmienności wyników.
- Różne budżety HPO; porównanie konkretnych konfiguracji, nie pełnych możliwości modeli.
- Grupa konfliktowa tylko w train; walidacja nie mierzy błędów na tej grupie.
- Po reficie dawna walidacja w treningu; brak dalszego wyboru epoki lub parametrów.
- Dodatkowy trening i pomiary kosztu; wyniki CPU z W2 nie opisują finalnych artefaktów.
- Nieustalona ekspozycja LLM na benchmark; wysoki wynik bez gwarancji generalizacji.

### Kiedy wrócić do tej decyzji

- Błąd danych, mapy lub podziału: wstrzymanie oceny i jawny zapis problemu.
- Szersze HPO, kilka seedów i zagnieżdżona grupowana CV: po oddaniu projektu.
  Bez dostrajania obecnych modeli do wyników tego testu.
