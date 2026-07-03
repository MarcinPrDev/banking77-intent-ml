"""Wczytywanie surowych danych US Accidents."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from us_accidents_ml.config import Settings


def load_raw(
    settings: Settings,
    columns: Sequence[str] | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Czyta surowy CSV US Accidents.

    sample_rows nadpisuje wartość z configu; None w obu miejscach = pełny zbiór
    (7,7 mln wierszy — na pełny zbiór planowany jest odczyt przez ETL/parquet).
    """
    n = sample_rows if sample_rows is not None else settings.data.sample_rows
    if not settings.paths.raw_csv.exists():
        raise FileNotFoundError(
            f"Brak pliku danych: {settings.paths.raw_csv}. "
            "Pobierz dataset z Kaggle (sobhanmoosavi/us-accidents) do katalogu dane/."
        )
    return pd.read_csv(settings.paths.raw_csv, usecols=columns, nrows=n)
