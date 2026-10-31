---
title: BANKING77 — klasyfikacja intencji
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: DistilBERT, 77 intencji bankowych — demo projektu kursu
---

# BANKING77 — klasyfikacja intencji (demo)

*English: a demo of a DistilBERT model that assigns an English banking customer message to one
of 77 intents (BANKING77). Final project of an AI & ML Engineering course.*

Demo modelu `distilbert-v1` z projektu końcowego kursu AI & ML Engineering (SAGES): wiadomość
klienta banku po angielsku → jedna z 77 intencji BANKING77. Kod, protokół ewaluacji i wyniki:
[repozytorium projektu](https://github.com/MarcinPrDev/banking77-intent-ml).

## Model

- `distilbert-base-uncased` (rewizja `12040ac`) dostrojony na 10 003 wiadomościach
  (train + walidacja BANKING77), `max_len` 128.
- Test BANKING77 (3 080 wiadomości): accuracy 93,02%, macro-F1 93,05%; ECE score wybranej klasy
  (15 przedziałów) 0,0096. Porównanie z innymi metodami i koszt predykcji — w repozytorium.
- Ten sam artefakt co w API projektu (Release `distilbert-v1`);
  SHA-256 `model/model.safetensors`: `a90ac88b06d6fdb493628308d220e80c5cce8d014f9a760cc3e41b3e134852b2`.
- Predykcja jak w repozytorium: CPU, softmax na logitach; aplikacja pokazuje intencję, score
  i dwie kolejne klasy.

## Ograniczenia

- Model zawsze wybiera jedną z 77 znanych intencji — nie wykrywa spraw spoza zbioru (OOD).
- Tylko angielski. Test BANKING77 jest zrównoważony (40 wiadomości na klasę); w banku rozkład
  intencji jest inny.
- Score to wynik modelu dla zwróconej klasy. Niskie ECE na teście nie gwarantuje poprawnej
  pewności dla każdej klasy ani dla nowych rodzajów wiadomości.
- To demo, nie system produkcyjny. Darmowy Space usypia po bezczynności — pierwsze uruchomienie
  po przerwie trwa dłużej.

## Dane i licencje

- Dane: BANKING77 — Casanueva, Temčinas, Gerz, Henderson, Vulić (2020), PolyAI;
  licencja CC BY 4.0, [źródło](https://github.com/PolyAI-LDN/task-specific-datasets).
  Przykłady w aplikacji to zdania kontrolne spoza zbioru.
- Model bazowy: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
  (Apache-2.0).
- Kod aplikacji: licencja MIT, jak repozytorium projektu.
