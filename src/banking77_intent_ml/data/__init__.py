"""Wczytywanie i audyt integralności danych BANKING77 — loader powstaje w W1 (pisze autor).

Katalog `data/` jest zwolniony ze skanu strażnika protokołu (`tests/test_contract.py`): tu wolno
znać plik testowy, ale tylko przez `loader.load_test()` — do audytu integralności (sumy,
liczebności, pokrycie) i oceny końcowej; poza `data/`, `evaluation/` i `audit/` strażnik
zabrania nawet nazwy `load_test`.
"""
