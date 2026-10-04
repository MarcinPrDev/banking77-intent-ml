"""Loader BANKING77: wczytanie `train.csv` / `test.csv` z `paths.raw_dir` + walidacja struktury.

Kod: asystent (Claude) na prośbę autora, 4.10.2026; decyzje projektowe (niżej) — autor.

Decyzje (Marcin, 4.10.2026):
1. Brak wartości = WYŁĄCZNIE napis pusty po `strip()`. CSV czytany bez zamiany napisów na NaN
   (pandas domyślnie uznaje za brak m.in. "NA", "None", "null", "N/A" — w zbiorze tekstowym
   to poprawne wiadomości). Każdy brak -> `DataValidationError`; wiersz NIE znika.
2. Pozycja wiersza w pliku = indeks DataFrame (0..n-1, nadany przez pandas). Loader niczego
   nie usuwa, nie sortuje i nie przenumerowuje: indeksy splitu, hash w manifeście i podzbiory
   3 055 / 385 to pozycje w pliku pod przypiętym commitem. Na danych do splitu żadnego
   `reset_index(drop=True)`.
3. Test jest dostępny wyłącznie przez `load_test()`; nazwa `load_test` to zakazany literał
   strażnika (`tests/test_contract.py`) poza `data/`, `evaluation/` i `audit/`.

Czego loader NIE robi: normalizacji i duplikatów (audyt, `make_splits`), mapowania nazwa -> id
(mapa etykiet), porównania nazw kategorii train/test (audyt integralności), podziału train/val
(`make_splits`), weryfikacji sum (`fetch_data.py --check`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from banking77_intent_ml.config import Settings, load_settings

EXPECTED_COLUMNS: tuple[str, ...] = ("text", "category")  # nagłówek CSV pod przypiętym commitem
_MAX_POSITIONS_IN_MESSAGE = 10


class DataValidationError(ValueError):
    """Dane niezgodne z kontraktem loadera (komunikat mówi, co i gdzie).

    Wyjątek zamiast `assert`: asserty znikają przy `python -O`, a walidacja danych ma działać
    zawsze.
    """


def _read_and_validate(path: Path, expected_rows: int, n_classes: int) -> pd.DataFrame:
    """Czyta jeden plik CSV, sprawdza kontrakt i zwraca dane BEZ żadnych zmian.

    Kontrole idą od "czy to w ogóle ten plik" do "czy treść jest kompletna"; każde naruszenie
    -> `DataValidationError`. Komunikaty podają pozycje wierszy, nigdy treść wiadomości
    (teksty testu i potencjalne PII nie trafiają do logów ani tracebacków).
    """
    if not path.is_file():
        raise DataValidationError(
            f"brak pliku {path} — pobierz dane: uv run python scripts/fetch_data.py"
        )

    # Decyzja 1: wszystko jako tekst, bez zamiany "NA" / "None" / "null" / ... na NaN.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")

    if tuple(df.columns) != EXPECTED_COLUMNS:
        raise DataValidationError(
            f"{path.name}: kolumny {list(df.columns)}, oczekiwano {list(EXPECTED_COLUMNS)}"
        )
    if len(df) != expected_rows:
        raise DataValidationError(f"{path.name}: {len(df)} wierszy, config: {expected_rows}")

    for column in EXPECTED_COLUMNS:
        empty = df[column].str.strip().eq("")  # pusty napis albo same białe znaki
        if empty.any():
            positions = df.index[empty].tolist()
            raise DataValidationError(
                f"{path.name}: {len(positions)} pustych wartości w kolumnie {column!r}, "
                f"pozycje wierszy: {positions[:_MAX_POSITIONS_IN_MESSAGE]}"
            )

    n_found = df["category"].nunique()
    if n_found != n_classes:
        raise DataValidationError(f"{path.name}: {n_found} kategorii, config: {n_classes}")

    # Decyzja 2: pozycja w pliku = indeks. Kontrola na końcu, tuż przed zwrotem, i względem
    # expected_rows (nie len(df)) — złapie każdą przyszłą zmianę wyżej, która coś usunie,
    # posortuje albo przenumeruje, także dropna() + reset_index() dopisane po liczeniu wierszy.
    if not df.index.equals(pd.RangeIndex(expected_rows)):
        raise DataValidationError(f"{path.name}: indeks nie jest 0..{expected_rows - 1}")
    return df


def load_train(settings: Settings | None = None) -> pd.DataFrame:
    """Train -> DataFrame z kolumnami `EXPECTED_COLUMNS`; indeks = pozycja wiersza w pliku.

    `settings=None` -> domyślny config. Ścieżka i oczekiwane liczby wyłącznie z configu.
    """
    s = settings if settings is not None else load_settings()
    return _read_and_validate(
        s.paths.raw_dir / s.dataset.train.filename,
        expected_rows=s.dataset.train.n_rows,
        n_classes=s.dataset.n_classes,
    )


def load_test(settings: Settings | None = None) -> pd.DataFrame:
    """Test -> DataFrame jak w `load_train` (z `settings.dataset.test`) — TYLKO audyt i ocena.

    Wyłącznie do audytu integralności (W1) i oceny końcowej (W3); wywołania tylko z `data/`,
    `evaluation/` i `audit/` — każde inne zatrzyma strażnik. Treści testu nie oglądamy przed
    oceną końcową.
    """
    s = settings if settings is not None else load_settings()
    return _read_and_validate(
        s.paths.raw_dir / s.dataset.test.filename,
        expected_rows=s.dataset.test.n_rows,
        n_classes=s.dataset.n_classes,
    )
