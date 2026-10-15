# ADR-0003: Wybór modelu do wdrożenia i źródło artefaktu w CI

- **Status:** zaakceptowany
- **Data decyzji:** 2026-10-15
- **Data akceptacji:** 2026-10-15
- **Etap:** W3

## Kontekst i problem

Jeden zamrożony model po reficie na 10 003 zdaniach. Docker CPU, potem kind w CI.
Decyzja po ocenie testowej, przed budową obrazu.

API klasyfikuje angielski tekst do jednej z 77 intencji.
Brak oceny na rzeczywistym ruchu i intencjach spoza tych klas.
Wymagania dotyczące wolumenu, latencji i kosztu pomyłki nie zostały ustalone.

Źródła: [README](../../README.md#status), [analiza błędów](../../notebooks/02_analiza_bledow.ipynb),
[ADR-0002](0002-protokol-ewaluacji.md).
MLflow: runy testowe 0c08edb4, 04ec980a, fb0d0bde i badanie cpu_cost_final.

## Kryteria

Warunki konieczne: CPU, oceniony artefakt, działanie bez sieci po starcie.
Porównujemy liczbę błędów, accuracy, p95 predykcji, rozmiar artefaktu,
zimny start i zależności obrazu. Bez liczbowego progu poprawy jakości lub latencji.
Pomiary przy jednym i ośmiu wątkach opisują komputer autora, nie wydajność poda.
Nie obejmują HTTP. Latencja usługi wymaga osobnego pomiaru na przykładach spoza testu.
Zimny start ma znaczenie przy restarcie.
Gotowość dopiero po załadowaniu i rozgrzaniu modelu.

Score nie jest skalibrowaną pewnością. Bez automatycznego progu kierowania do człowieka.
Taki próg wymaga nowej niezależnej walidacji; dawna weszła do refitu.

## Rozważane opcje

TF-IDF z LR: accuracy 90,81 procent, 283 błędy, 5,2 MB.
P95 predykcji przy jednym wątku 0,43 ms, zimny start 0,85 s. Bez torcha.

MiniLM z kNN: accuracy 92,05 procent, 245 błędów, 107,0 MB z enkoderem.
P95 predykcji przy jednym wątku 16,83 ms, zimny start 4,69 s.
Torch; score zawsze 1,0 przy k równym 1.

DistilBERT: accuracy 93,02 procent, 215 błędów, 268,8 MB.
P95 predykcji przy jednym wątku 31,42 ms, zimny start 3,30 s. Torch.
Względem TF-IDF artefakt około 52 razy większy, p95 około 73 razy wyższe.

## Decyzja

DistilBERT z models/final/distilbert.
Najwyższa accuracy w obecnej ocenie: 93,02 procent.
68 błędów mniej niż TF-IDF i 30 mniej niż MiniLM z kNN.
W tym wdrożeniu dajemy pierwszeństwo mniejszej liczbie błędów.
Akceptujemy koszt: artefakt 268,8 MB, p95 predykcji przy jednym wątku 31,42 ms
i zależność od torcha CPU. To kompromis projektowy, nie dowód opłacalności w banku.

TF-IDF odrzucamy mimo najniższego kosztu: ma 68 dodatkowych błędów.
Poprawa DistilBERT wynosi 2,21 punktu procentowego, kosztem około 52 razy większego
artefaktu, około 73 razy wyższego p95 przy jednym wątku i dłuższego zimnego startu.

MiniLM odrzucamy mimo mniejszego artefaktu i niższego p95 przy jednym wątku:
ma 30 dodatkowych błędów i również wymaga torcha.
DistilBERT ma krótszy zmierzony zimny start: 3,30 s wobec 4,69 s.
Różnice accuracy liczone z liczby błędów, przed zaokrągleniem.

Przewaga nad kNN to 0,97 punktu, czyli 30 błędów.
Jeden seed i odwrócenie rankingu względem walidacji nie dowodzą stabilnej przewagi.
Nie rozstrzygamy, ile różnicy wynika z losowości treningu i składu próby.
To wybór konkretnego artefaktu. Konfiguracje i etykiety bez zmian.

## Skąd model w CI

GitHub Release asset. Przypięta wersja i SHA-256 archiwum w configu.
Pakujemy istniejące wagi, tokenizer, konfigurację i metadata.json bez ponownego zapisu.
Manifest: hashe plików i mapy etykiet, wersje bibliotek, refit 5b883a97, ocena fb0d0bde.

Pliki modelu bit w bit zgodne z ocenionym artefaktem.
CI sprawdza hash archiwum i plików, potem kopiuje je do obrazu.
Bez pobierania wag przy starcie ani podczas predykcji.
Trening w CI odrzucamy: tworzyłby nowy artefakt.
GCS odrzucamy: dodatkowa infrastruktura i uprawnienia poza minimum.

Test przeniesienia na ustalonych przykładach z dawnej walidacji:
te same id lokalnie i w obrazie CPU, score z tolerancją.
To kontrola techniczna, nie ocena jakości. Obliczenia CPU i GPU mogą się różnić.
Bez ponownej oceny na teście.

## Konsekwencje

### Dobre

Mniej błędów w obecnej ocenie. Kompletny artefakt bez osobnego cache enkodera.
CI wdraża ocenione wagi. Wersja i hashe potwierdzają pochodzenie.

### Złe i ryzyka

Więcej zależności i pracy nad obrazem CPU.
Rozmiar obrazu, RAM, czas budowy i latencja HTTP do zmierzenia.
268,8 MB opisuje artefakt, nie obraz ani RAM.
Zimny start 3,30 s zmierzono lokalnie, z plikami w cache systemu.
Nie obejmuje pobierania obrazu ani startu poda.
Jeden seed i zbalansowany test ograniczają wnioski o rzeczywistym ruchu.

### Co wynika dla W3 i W4

W3: api.model wskazuje distilbert.
deploy.model_version: distilbert-v1.
deploy.artifact_dir: models/final/distilbert.
Tag Release odpowiada deploy.model_version.
Config źródłem wersji, ścieżki, tagu Release i hasha.

Extra cpu z torchem z indeksu CPU i transformers. Bez CUDA.
Testy API z atrapą oraz integracyjne z wybranym artefaktem.
Dockerfile kopiuje model, config i mapę. Kontrola działania bez sieci i GPU.

W3 kończy się działającym obrazem. W4 pobiera Release, sprawdza hashe, buduje obraz,
wdraża do kind, sprawdza HTTP i wersję oraz mierzy latencję odpowiedzi.
Limity zasobów, liczba wątków i sondy wynikną z pomiarów kontenera.
Czas pracy nad torchem nieznany.
Brak działającego obrazu na starcie W4 oznacza fallback TF-IDF zgodnie z README.
Wtedy aktualizacja ADR i wersji. Termin 31.10 ma pierwszeństwo.

### Kiedy wrócić do tej decyzji

Wyniki pomiarów kontenera, problemy z RAM, startem lub buildem.
Ustalenie wymagań dotyczących ruchu, latencji i kosztu pomyłki.
Nowe dane i kilka seedów po oddaniu projektu.
Nowe uzasadnienie i wersja, bez dostrajania do obecnego testu.
