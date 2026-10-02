# ADR-0001: Zmiana tematu projektu z US Accidents (Severity) na BANKING77 (klasyfikacja intencji)

- **Status:** zaproponowany
- **Data:** 2026-09-30
- **Etap:** W1

> Sekcje „Kontekst i problem" oraz „Rozważane opcje" zestawił asystent z notatek researchu
> (Obsidian: `03 Research tematu — Claude.md` §1, §2.3, §3, §8; drugi research (notatka 03, §1, §3);
> `04 BANKING77 — weryfikacja tematu.md`; porównanie obu researchów (notatka 05) §3, §5, §6;
> weryfikacja porównania (notatka 06)). Liczby oznaczone „obliczenia soczewek researchu"
> policzyli agenci researchu (Claude) na lokalnym parquecie US Accidents — skrypty nierecenzowane,
> zarchiwizowane w `P:\Repos\us-accidents-audit\`. Własne pomiary autora z 28.09: rozkład klas
> i wolumen rok × źródło (skrypt `scripts/sanity_w1.py` — w historii gita do commita `5c94e05`,
> kopia robocza w `P:\Repos\us-accidents-audit\repo-leftovers\`).
> Sekcje „Decyzja" i „Konsekwencje" pisze **Marcin własnymi słowami** (test obrony pyta o ten ADR).

## Kontekst i problem

**W skrócie:** etykieta `Severity` opisuje proces raportowania dostawcy, a nie porównywalny wpływ
zdarzenia (sygnały: zgodność między źródłami kappa ≈ 0; macro-F1 spada z ~0,81 do ~0,27, gdy
usunąć skróty protokołu), a stary plan wymagał 59–89 h przy budżecie 50–65 h i terminie 31.10.
Szczegóły w punktach 1–7.

Pierwotny temat (wybór 2026-07-03): klasyfikacja `Severity` 1–4 na US Accidents (7 728 394 wierszy,
46 kolumn, CC BY-NC-SA 4.0) trzema ścieżkami (tabular, tekst `Description`, LLM) z kontraktem
„predykcja w chwili zgłoszenia" i listą `forbidden_features`. W1 (EDA + audyt leakage, 28–29.09)
oraz dwa niezależne researche (29–30.09) wykazały, że etykieta nie nadaje się na pytanie badawcze
w takiej formie — nie dlatego, że zbiór jest „zepsuty", lecz dlatego, że **`Severity` mierzy przede
wszystkim proces raportowania danego dostawcy**, a nie porównywalny wpływ zdarzenia:

1. **Etykieta pochodzi od dostawców API o różnych skalach.** Bing: 1–4 (LowImpact…Serious) z osobnym
   `roadClosed`; MapQuest: 0–4. Publikacja o zbiorze (arXiv 1906.05409) nie opisuje, jak wyznaczano
   severity; współautor nazywa ją „coarse-grained" (arXiv 2411.07537). Brak wspólnej kalibracji
   Source1/2/3 i wiarygodnego mapowania nazw źródeł na dostawców.
2. **Rozkład klas i skład źródeł zmieniają się w czasie** (obliczenia soczewek researchu, potwierdzone
   trzykrotnie; wolumen rok × źródło zgodny z pomiarem autora z 28.09):
   udział S3 w Source1 w latach 2016–2023: 17,4 / 18,4 / 18,4 / 11,7 / 6,5 / 0 / 0 / 0 % (uskok
   IX 2020); klasa 1 istnieje tylko na „wyspach" (Source1 2020: 4,0 %, Source2 2022: 15,4 %,
   Source3 2022: 24,3 %); udział Source2 spada z 68–80 % (2016–19) do 12,4 % (2022) i 0 % (2023).
3. **Zgodność etykiet między źródłami dla par zdarzeń dopasowanych miejscem i czasem jest bliska
   losowej** (obliczenia soczewek researchu, cztery protokoły parowania: od 724 do 14 902 par): kappa Cohena **0,009–0,053** przy surowej
   zgodności 58–68 %. To **sygnał** problemu etykiety, nie dowód jej błędności — część par może być
   różnymi zdarzeniami, a kappa ≈ 0 przy dominacji klasy 2 to częściowo paradoks kappa (stąd
   w metodyce PABAK/Gwet AC1). W Source1 od 2021: 284 006 dokładnych duplikatów; 45 596 wierszy
   S4 ma bliźniaka z etykietą S2.
4. **Drabina leakage** (XGBoost, stałe HP, próbka 1,5 mln, trzy niezależne replikacje soczewek
   + sonda wstępna researchu):
   macro-F1 **~0,81** przy protokole z literatury (balans 12,5 tys./klasę, split losowy, `Source`
   + pola post-hoc) → 0,78 bez post-hoc → 0,68 bez `Source` → 0,45 rozkład naturalny →
   **~0,27 przy splicie czasowym** (≤2021 → ≥2022) → 0,24 tylko Source1. Replikacja TF-IDF+LR
   (3 % danych): 0,701 split losowy vs 0,532 czasowy (F1 klasy 1 = 0). Sonda wstępna researchu
   (4 %, jeden seed): sam `Source` dodaje +0,09–0,14 macro-F1.
5. **Tekst zdradza regułę dostawcy** (obliczenia soczewek researchu): 96,7 % rekordów S4 w Source1 z lat 2016–19
   zawiera „closed"; P(S4 | „closed") = 0,91 w 2016–19 i tylko 0,26 w 2020–23. Pola post-hoc to
   w dużej mierze wartości domyślne (Distance = 0 w 85,0 % Source2 i 93,0 % Source3; Bing
   dokumentuje backfill końca zdarzenia jako start + 2 h).
6. **Literatura jest zawyżona przez leakage lub protokół:** Umer i in. 0,974 (selekcja cech na całym
   zbiorze, split losowy), Grigorev i in. F1 0,89 (balans + `Source`, `Distance`, `End_Lat/Lng`,
   `Description` na wejściu), Aboulola 98,12 %. Uczciwego publicznego punktu odniesienia dla
   kontraktu „w chwili zgłoszenia" nie ma; Alhawarat i in. 2026 (Severity = 4, train 2016–21 /
   val 2022 / test 2023, XGBoost AUROC 0,8215, AUPRC 0,1306) to odniesienie dla szerszego zadania,
   nie baseline 1:1.
7. **Czas:** plan K0 szacowany na 59–89 h (mediana ~72 h) przy budżecie 50–65 h; na 29.09 projekt
   był ok. 1 weekend za planem (W1 niedomknięte: brak splitów i deduplikacji). Termin 31.10 stały.

Oba researche dały werdykt **ZMIEŃ** (Claude: K0 = 2,917/5, wiarygodność 1,83; drugi research: 3,20/5).
Rubrykę K0 formalnie by zaliczył (ocena rubryki 4,17) — powodem zmiany są wiarygodność wyników
i bezpieczeństwo terminu, nie ryzyko niezaliczenia. Decyzja jest potrzebna teraz, bo każdy kolejny
weekend na US Accidents zmniejsza budżet na nowy temat, a 3–4.10 to pierwszy pełny slot.

## Rozważane opcje

Dwie niezależne macierze (wagi Claude: wiarygodność 25 %, wykonalność 20 %, nauka 20 %, rubryka 15 %,
portfolio 10 %, hak 10 %; drugi research: identyczne wagi i analogiczne kryteria — m.in. „czas/sprzęt"
zamiast „wykonalność" — inny zbiór kandydatów). Macierze
nie miały wspólnego zbioru kandydatów, więc część opcji ma ocenę tylko z jednej strony.

1. **K0 — US Accidents bez zmian** (Severity 1–4, tabular vs tekst vs LLM, fuzja). Claude **2,917**
   (#11 z 11), drugi research **3,20** (najniżej z 10). Wnioski merytoryczne nieinterpretowalne (pkt 1–6 wyżej);
   sygnał porażki już wystąpił; 59–89 h przy budżecie 50–65 h. Odrzucone przez obu.
2. **Pivot w obrębie zbioru: K2 + K1-min** (audyt etykiety i drabina leakage jako rdzeń — „MSA dla
   etykiet"; prosty model S4 vs reszta w Source1 2021–23 za API). Claude **4,008** (#3; najwyższy
   „hak" 4,50, ale słaby model serwowany: ~60 % wierszy S4 ma bliźniaka S2, sufit AP ~0,78 w grupach
   wybranych pól, Groq bez logprobs). Najbliższy odpowiednik w drugim researchu: **P1 — rekonstrukcja kategorii
   jednego źródła, 4,00** („najlepszy pivot, 50–60 h", ale poniżej najlepszych alternatyw). Koszt
   zmiany zerowy, rdzeń 50–69 h i tylko z K1-min; ryzyko odbioru jako „wynik negatywny".
3. **NHTSA Recall Intelligence** (przypisanie komponentu, 16 klas, z opisu wady w raportach Part 573;
   ~16,6 tys. kampanii, domena publiczna; benchmark preprint 09.2026 bez recenzji: TF-IDF macro-F1
   0,8157 na stałej liście 3 325, RoBERTa 0,79–0,80; 3 protokoły przecieku). Claude **#1 — 4,075**
   (przewaga 0,008 nad TEP, w granicach szumu; wygrywa rozstrzygaczami: domena automotive, kontrola
   przed/po cutoffie LLM, brak wyników LLM w literaturze). W drugim researchu NHTSA nie miało oceny punktowej;
   zweryfikowano dane (16 626 kampanii / 16 klas / 3 325 ID) i wskazano je jako alternatywę, „jeśli
   przeważy domena". Ryzyka: ETL 5–7 h (szacunek, nie pomiar), szum mapowania analityka
   (klasa-worek EQUIPMENT), TF-IDF ≈ RoBERTa (sufit), kontrola cutoffu zakłócona dryfem.
4. **Tennessee Eastman (MSPC)** — detekcja usterek przy równym FAR: PCA-T²/Q vs AE/1D-CNN/XGBoost.
   Claude **#2 — 4,067** (wiarygodność 4,67 — najwyższa, ex aequo z M5; etykiety z konstrukcji
   symulacji; hak 3,00; wygrywa przy wagach premiujących wiarygodność). Brak NLP, LLM tylko jako
   opcja, 1,4 GB .RData, 54–71 h, odbiór
   „akademicki". Bez oceny w drugim researchu.
5. **BANKING77** — klasyfikacja intencji bankowych, 77 klas, 10 003 / 3 080, ~1 MB, CC BY 4.0,
   ugruntowany benchmark (DistilBERT 92,1 %, BERT-large 93,66 %). Drugi research: **4,45** (#2 za VisA 4,60;
   przy haku VisA = 3 wygrywa BANKING77; „najlepszy przy priorytecie szerokości nauki i małego ryzyka
   wykonania"). Claude nie miał BANKING77 w macierzy; w porównaniu head-to-head z NHTSA (05 §3):
   wygrywa ryzykiem wykonania (gotowy split, 1 MB, benchmark z protokołem), przegrywa domeną/portfolio
   i węższą EDA; ekspozycja LLM na benchmark nieustalona (bez gwarancji czystego testu). Znane
   ograniczenia: ~14 % potencjalnych problemów etykiet (wskazania automatyczne), 25 nakładek
   train/test po normalizacji, pochodzenie/anonimizacja nieudokumentowane (skan PII 0 trafień,
   bez gwarancji). Ocena podtrzymana w drugim researchu po trzech rundach weryfikacji; Claude przyjął jako jedyny zbiór
   podstawowy (05 §6).

Poza tą piątką: VisA (drugi research: #1, 4,60 — CV, PCB; Claude: benchmark nasycony, Dinomaly 98,7 AUROC;
mniej NLP/LLMOps), Scania CX, VAD, OSHA, M5, Crimp, Bosch CNC (Claude 3,4–3,9), Hydraulic, Steel,
SECOM, SciFact, FinQA (drugi research: 3,8–4,25) — odrzucone wcześniej, szczegóły w notatkach 03.

## Decyzja

<!-- TODO-Marcin: napisz własnymi słowami (3–6 zdań). Fakty do użycia: kryterium = bezpieczeństwo
terminu 31.10; wariant A (rdzeń TF-IDF+LR / MiniLM+kNN / DistilBERT, LLM jako pierwszy stretch);
Groq plan płatny; kind obowiązkowo z auto-deployem i testem HTTP w GitHub Actions, GKE stretch;
Spark i CV poza zakresem; NHTSA i audyt US Accidents po projekcie. -->

Wybrano: **TODO-Marcin: napisz własnymi słowami**, ponieważ TODO-Marcin: napisz własnymi słowami.

Pytania pomocnicze (odpowiedz sobie, zanim napiszesz — test obrony zapyta o to samo):
1. Dlaczego „bezpieczeństwo terminu" przeważyło nad domeną (NHTSA) i nad najwyższą wiarygodnością (TEP)?
   Co konkretnie w BANKING77 zmniejsza ryzyko: gotowy split, rozmiar danych, benchmark z opisanym
   protokołem, brak ETL — i którego z tych elementów NHTSA nie miał?
2. Dlaczego nie pivot K2 + K1-min, skoro koszt zmiany był zerowy, a audyt leakage już zrobiony?
   Co tracisz, rezygnując z „MSA dla etykiet" jako rdzenia, i jak to odzyskujesz w sekcji „Po projekcie"?
3. Dlaczego rdzeń **bez** LLM (wariant A), skoro Z9 i LLMOps są na topie i masz działający klucz Groq?
   Co ekspozycja BANKING77 na benchmark robi z interpretacją wyniku LLM i jak to napiszesz w README?

## Konsekwencje

<!-- TODO-Marcin: napisz własnymi słowami. Szkielet poniżej wskazuje, co powinno się tu znaleźć;
nie przepisuj punktów z notatek — nazwij to, co Ty zyskujesz i za co Ty płacisz. -->

- Dobre: TODO-Marcin: napisz własnymi słowami
  <!-- słowa-klucze do przemyślenia (nie do przepisania): punkt odniesienia · rozmiar danych
  a czas na protokół i wdrożenie · jedno pochodzenie liczb · co przenosisz ze szkieletu ·
  pokrycie rubryki -->
- Złe / ryzyka: TODO-Marcin: napisz własnymi słowami
  <!-- słowa-klucze do przemyślenia (nie do przepisania): domena vs profil zawodowy · popularność
  zbioru · węższa EDA · jakość etykiet i nakładki train/test · PII · ekspozycja LLM ·
  test zrównoważony vs realny ruch · co przepada z pracy nad US Accidents -->
- Kiedy wrócić do tej decyzji: TODO-Marcin: napisz własnymi słowami
  <!-- słowa-klucze do przemyślenia: PII · bramka po W1 · warunek udokumentowanej anonimizacji -->

Pytania pomocnicze do Konsekwencji:
1. Co z audytu US Accidents (drabina leakage, kappa, dryf) **zostaje** w tym projekcie — jako ADR,
   jako sekcja „dlaczego zmiana tematu" w README, jako materiał portfolio po oddaniu — a co przepada?
2. Które ograniczenia BANKING77 wpisujesz do README **przed** treningiem (żeby nie wyglądały na
   wymówki po wynikach): nakładki train/test, potencjalne błędy etykiet, brak OOD, ekspozycja LLM, PII?
3. Jaki jest Twój warunek powrotu do tej decyzji i kto go sprawdza w niedzielę 4.10 wieczorem —
   co dokładnie musi działać (dane, split, baseline > Dummy, zapis/odczyt), a co świadomie **nie** jest
   warunkiem (92 %, GPU)?
