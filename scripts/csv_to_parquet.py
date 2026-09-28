"""Jednorazowa konwersja CSV -> parquet (DuckDB, bez ladowania 3 GB do RAM) + sanity check.

Plumbing pod W1 (Blok 1). Uruchom raz:  uv run python scripts/csv_to_parquet.py
Powtorna konwersja tylko z flaga --force (parquet jest gotowy w ~30 s, ale po co).
"""

import sys
import time
from pathlib import Path

import duckdb

SRC = Path("dane/US_Accidents_March23.csv")
DST = Path("dane/processed/us_accidents.parquet")


def convert(con: duckdb.DuckDBPyConnection) -> None:
    t0 = time.perf_counter()
    # Kolumny o mieszanych formatach trzymamy jako VARCHAR: czasy (czesc wpisow ma ulamki
    # sekund) i Zipcode (5 cyfr vs ZIP+4 z myslnikiem) — autodetekcja typow na probce
    # wywraca sie w polowie pliku. sample_size=-1 = sniffing typow po calym pliku.
    con.sql(
        f"""
        COPY (
            SELECT * FROM read_csv('{SRC.as_posix()}',
                header=true, sample_size=-1,
                types={{'Start_Time': 'VARCHAR', 'End_Time': 'VARCHAR',
                        'Weather_Timestamp': 'VARCHAR', 'Zipcode': 'VARCHAR'}})
        ) TO '{DST.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    size_gb = DST.stat().st_size / 1e9
    print(f"Konwersja: {time.perf_counter() - t0:.0f} s, parquet: {size_gb:.2f} GB")


def main() -> None:
    # konsola Windows domyslnie cp1250 — ramki DuckDB (znaki Unicode) wywalaja print()
    sys.stdout.reconfigure(encoding="utf-8")
    if not SRC.exists():
        raise SystemExit(f"Brak pliku {SRC} — pobierz dataset z Kaggle.")
    DST.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    if DST.exists() and "--force" not in sys.argv:
        print(f"{DST} juz istnieje — pomijam konwersje (wymus: --force).")
    else:
        convert(con)

    sanity = con.sql(
        f"""
        SELECT COUNT(*) AS wiersze,
               COUNT(DISTINCT ID) AS unikalne_id,
               MIN(left(Start_Time, 19)) AS pierwszy,
               MAX(left(Start_Time, 19)) AS ostatni
        FROM '{DST.as_posix()}'
        """
    ).df()
    print(sanity.to_string(index=False))


if __name__ == "__main__":
    main()
