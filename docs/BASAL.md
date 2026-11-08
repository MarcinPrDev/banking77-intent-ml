> Przeniesione z README 8.11.2026 (redakcja po feedbacku prowadzącego); treść bez zmian.

# Basal zero-shot — pełny opis badań 1 i 2 (30–31.10.2026)

## Basal zero-shot — dodatek z 30–31.10

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

Rozszerzenie po głównym badaniu, bez zmiany wybranego modelu do wdrożenia.

### Basal 4,5B — badanie 1, 30–31.10

Prerejestracja: `9360375`, infrastruktura: `3e054ef`, instrukcja i opisy v2: `5c6797f`.
Plan obejmował dwa modele i dwa warianty opisów. Model 11B FP8 odpadł technicznie:
w tym środowisku Windows tryby z grafami CUDA nie zmieściłyby się w 16 GB lub wymagałyby
niedostępnego Tritona, a `eager` nie obsługuje poprawnie jego formatu ModelOpt FP8.
Ustalone z kodu silnika i rozmiarów wag oraz buforów KV, bez uruchomienia modelu 11B.
Zmiana prerejestracji (`16fb241`) przed odczytem wyników walidacji 4,5B, nie na podstawie
jakości 11B. Pozostało porównanie v1 i v2 na modelu 4,5B.

Model `Remek/basal-2.0-4.5B`, rewizja `e5ba914`, silnik `basal-serve` 2.4.1.
Wybór jednej z 77 podanych intencji, bez swobodnego generowania odpowiedzi.
Zero-shot: bez dostrajania wag w tym projekcie i bez przykładów w żądaniu.
Ekspozycja Basala na BANKING77 podczas wcześniejszego treningu nieustalona.
Wspólna instrukcja dla obu wariantów; v1 to nazwy etykiet zapisane jako frazy,
a v2 — krótkie opisy. Niejednoznaczne nazwy sprawdzone wyłącznie na części treningowej.
Kalibracja dostarczona z modelem, bez własnego dopasowania. Jej jakość na BANKING77 zmierzona
na zapisanych predykcjach — zob. [Kalibracja score](../README.md#kalibracja-score-na-teście-3-080).

Walidacja: te same 385 zdań co dla LLM, po 5 na klasę, zgodny `sha256_val_sample` w runach.
Każdy wariant oceniony raz. Wybór według macro-F1; różnica od najlepszego wyniku poniżej
0,001 oznacza remis rozstrzygany kosztem modelu, potem pierwszeństwem v1.

| Kandydat | Macro-F1 (walidacja 385) | Accuracy (walidacja 385) | p50 żądania HTTP | Run MLflow |
|---|---:|---:|---:|---|
| **4,5B v2** | **0,8080** | **0,8156** | 1 101,5 ms | `ad6c563c` |
| 4,5B v1 | 0,7435 | 0,7506 | 482,6 ms | `c4aff0b3` |

Opisy v2 poprawiły macro-F1 o 0,0645, przy wyższej latencji. Zwyciężyła jakość, reguła remisu
nie zadziałała. Na tej samej walidacji LLM v2 miał macro-F1 0,8147 (`081d183e`), a LLM v1
0,7971 (`526e5400`). Wynik Basal 4,5B v2 jest bliski LLM v2, ale dotyczy to tylko walidacji.

Pomiar na RTX 5080, tryb `eager`, `--template-prefix off --deterministic`, batch=1.
p50 to `latency_wall_p50_ms`: czas po stronie klienta, z lokalnym HTTP.
Tryby z grafami CUDA dla 4,5B nie zmieściły się w pamięci; pomiar nie pokazuje najlepszej możliwej
wydajności silnika. Inne warunki niż predykcja lokalnych klasyfikatorów na CPU i zdalne API LLM.

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

Na 385 wiadomościach Basal 4,5B trafił 303 razy, a LLM 311 — różnica 8 odpowiedzi.
Na walidacji Basal 4,5B miał wyższe accuracy (o jedną wiadomość: 314 wobec 313),
ale niższe macro-F1 niż LLM v2.
Nie rozstrzygam trwałej przewagi jednej z tych ścieżek. Testu istotności dla tej pary
nie było w prerejestracji i nie został wykonany.

Macro-F1 Basala 4,5B spadło z 0,8080 na walidacji (385) do 0,7966 na pełnym teście (3 080)
i 0,7787 na podzbiorze testowym (385). To różne próbki; sam spadek nie dowodzi przeuczenia
opisów. Pokazuje ograniczenie przenoszenia wyniku z małej walidacji na test.

### Basal ultra — badanie 2, 31.10

Osobna prerejestracja `9f36a77`, przed obliczeniami. Model `Remek/basal-omni-1.0-ultra`
(31B), rewizja `6d5ba67`, na wynajętej karcie H100 80 GB w RunPod.
Te same instrukcje, opisy v1/v2 i próbka walidacji 385 zdań co dla 4,5B; zgodność skrótów
zapisana w MLflow. Bez dostrajania wag w projekcie; wcześniejsza ekspozycja na BANKING77
pozostaje nieustalona. Szczegóły środowiska i przebiegu: [notatka](basal-ultra.md).

| Ultra | Macro-F1 (walidacja 385) | Accuracy (walidacja 385) | Run MLflow |
|---|---:|---:|---|
| v2 | 0,8562 | 0,8597 | `b8944d95` |
| v1 | 0,7896 | 0,7948 | `c6e90ce6` |

Wybrany v2 według tej samej reguły macro-F1; remis nie wystąpił. Wpis frozen `aad7425`
przed jedynym testem, run `b855251a`. Wyniki na 3 080 i 385 zdaniach w tabeli powyżej.

Ultra trafiła 2 612 z 3 080 wiadomości, o 136 więcej niż 4,5B (2 476).
Nadal niższe accuracy i macro-F1 niż wszystkie trzy modele uczone na BANKING77.
Na tych samych 385 zdaniach: ultra 319 trafień, LLM v2 311, Basal 4,5B 303,
a modele uczone 351–354. Ultra uzyskała tu najwyższe wyniki spośród badanych ścieżek
zero-shot. To wynik tej próbki, nie dowód trwałej przewagi nad LLM.
Dla porównań z Basalem nie zaplanowano z góry ani nie wykonano testów istotności.

Macro-F1 ultra spadło z 0,8562 na walidacji (385) do 0,8441 na pełnym teście (3 080)
i 0,8107 na podzbiorze testowym (385). Tak jak dla 4,5B, różne próbki nie pozwalają
przypisać samego spadku przeuczeniu opisów.

Test: p50/p95 po stronie serwera 716/722 ms, po stronie klienta 952/978 ms.
Czas klienta obejmuje tunel SSH do Norwegii. Także czas serwera nie daje porównania
szybkości samych modeli: ultra działała na H100, a 4,5B na RTX 5080.
Oba badania w trybie `eager`, zgodnie z protokołem. Nie mierzono ultra z grafami CUDA;
716 ms nie należy traktować jako najlepszego możliwego czasu silnika na H100.

Koszt z rozliczeń RunPod: 9,90 USD, w tym GPU 9,83 USD i dyski 0,07 USD.
Pod działał około 2 h 28 min, w tym około 1 h 18 min oczekiwania na wpis frozen.
Kwota obejmuje przygotowanie, walidację, oczekiwanie i test. Sam test trwał 48 min,
czyli około 3,2 USD za GPU według stawki 3,99 USD/h — to szacunek, nie osobna pozycja rachunku.
Pod usunięty po teście, tunel zamknięty.
