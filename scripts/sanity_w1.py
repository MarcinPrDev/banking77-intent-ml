import sys

import duckdb

sys.stdout.reconfigure(encoding="utf-8")
P = "'dane/processed/us_accidents.parquet'"
con = duckdb.connect()


def q(title: str, sql: str) -> None:
    print(f"\n=== {title} ===")
    print(con.sql(sql).df().to_string(index=False))


q("Wolumen per rok x Source (skok ~2020?)", f"""
    SELECT year(CAST(left(Start_Time, 19) AS TIMESTAMP)) AS rok, Source, COUNT(*) AS n
    FROM {P} GROUP BY 1, 2 ORDER BY 1, 2""")
q("Rozklad Severity", f"""
    SELECT Severity, COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM {P} GROUP BY 1 ORDER BY 1""")
q("Braki wspolrzednych (pod geohash)", f"""
    SELECT COUNT(*) FILTER (WHERE Start_Lat IS NULL OR Start_Lng IS NULL) AS brak_geo,
           COUNT(*) AS wszystkie
    FROM {P}""")