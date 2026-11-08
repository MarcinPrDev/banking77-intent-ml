> Przeniesione z README 8.11.2026 (redakcja po feedbacku prowadzącego); treść bez zmian.

# Historia projektu

> **Zmiana tematu 30.09.2026.** Do 29.09 projekt dotyczył predykcji `Severity` na US Accidents.
> Powody i liczby z audytu: sekcja *Historia decyzji* i `docs/adr/0001-zmiana-tematu-na-banking77.md`.
> Repo i pakiet przemianowano 2.10.2026 (wcześniej `us-accidents-ml` / `us_accidents_ml`) — zapis
> w `docs/RENAME.md`; stare adresy GitHuba przekierowują.

## Historia decyzji

- **[ADR-0001](../adr/0001-zmiana-tematu-na-banking77.md) — zmiana tematu z US Accidents na BANKING77**.
  Etykieta `Severity` w US Accidents okazała się mieszanką skal różnych dostawców API z dryfem
  schematu (udział klasy 3 w Source1 spada z ~18% do 0% od 2021; zgodność między źródłami
  kappa 0,01–0,05), bez uczciwego publicznego punktu odniesienia dla kontraktu „w chwili
  zgłoszenia". Przy budżecie 50–65 h do 31.10 i szacunku 59–89 h dla starego planu wybrano
  zbiór z ugruntowanym benchmarkiem i gotowym testem (kryterium: bezpieczeństwo terminu).
- Wnioski z audytu US Accidents (drabina leakage, kappa źródeł) i temat NHTSA Recalls →
  **po oddaniu projektu**, nie jako stretch.
- **[ADR-0002](../adr/0002-protokol-ewaluacji.md) — protokół ewaluacji** (zaakceptowany 14.10):
  grupowany split, grupa konfliktowa, mapa etykiet, podzbiory testowe 3 055 / 385 i refit
  na 10 003 przed testem.
- **[ADR-0003](../adr/0003-wybor-modelu-do-wdrozenia.md) — wybór modelu do wdrożenia i źródło
  artefaktu w CI** (zaakceptowany 15.10): DistilBERT, model w CI z GitHub Release (opcja a);
  wybór po ocenie na teście, bez liczbowego progu — zob. *Odpowiedź na pytanie badawcze*.
