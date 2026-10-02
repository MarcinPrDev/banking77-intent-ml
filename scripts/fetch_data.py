"""Pobranie BANKING77 (train.csv, test.csv) z przypiętego commitu + weryfikacja SHA-256.

Plumbing pod W1 (30.09.2026). Dane są poza gitem (`dane/banking77/` w .gitignore) — każdy klon
repo odtwarza je tym skryptem. Commit, ścieżki i sumy kontrolne żyją w `configs/config.yaml`,
nie tutaj: skrypt tylko je czyta i egzekwuje.

Uruchomienie:
    uv run python scripts/fetch_data.py            # pobiera brakujące lub uszkodzone pliki
    uv run python scripts/fetch_data.py --force    # pobiera wszystko od nowa
    uv run python scripts/fetch_data.py --check    # tylko weryfikuje sumy istniejących plików
    uv run python scripts/fetch_data.py --only train   # jeden plik (np. w CI: bez test.csv)

`--force` i `--check` wykluczają się; `--only` łączy się z każdym z nich.

Kody wyjścia: 0 = wybrane pliki obecne i zgodne z sumami, 1 = brak pliku / błąd pobrania /
niezgodna suma (plik z niezgodną sumą NIE jest zapisywany).

Zależności: stdlib + PyYAML (zależność główna projektu). Celowo bez importu pakietu
`us_accidents_ml` — skrypt ma działać także zanim pakiet zostanie zainstalowany lub przemianowany.

Licencja danych: CC BY 4.0 (PolyAI-LDN/task-specific-datasets) — atrybucja w README → Licencja.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
CHUNK = 1 << 16  # 64 KiB
TIMEOUT_S = 60


def fail(msg: str) -> NoReturn:
    print(f"\n[FAIL] FETCH DATA: {msg}")
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _abs(p: str) -> Path:
    candidate = Path(p)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_targets(only: str | None = None) -> tuple[Path, list[tuple[str, str, str, int]]]:
    """Zwraca (raw_dir, [(nazwa, url, sha256, n_rows), ...]) na podstawie configu.

    `only` ("train" / "test") ogranicza listę do jednego pliku — np. w CI, gdzie test.csv
    nie jest do niczego potrzebny.
    """
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    base_url = str(ds["base_url"]).rstrip("/")
    commit = str(ds["commit"])
    raw_dir = _abs(cfg["paths"]["raw_dir"])
    targets = []
    for name in ("train", "test"):
        if only and name != only:
            continue
        f = ds["files"][name]
        url = f"{base_url}/{commit}/{f['path']}"
        targets.append((Path(f["path"]).name, url, str(f["sha256"]).lower(), int(f["n_rows"])))
    return raw_dir, targets


def download(url: str, dst: Path, expected_sha: str) -> None:
    """Pobiera do pliku tymczasowego, liczy sumę w locie, podmienia atomowo po weryfikacji."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.sha256()
    size = 0
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "banking77-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, tmp.open("wb") as out:
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
                size += len(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        fail(f"HTTP {exc.code} dla {url} — sprawdź commit i ścieżkę w configs/config.yaml.")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        fail(f"brak połączenia lub błąd zapisu ({type(exc).__name__}: {exc}).")

    got = h.hexdigest()
    if got != expected_sha:
        tmp.unlink(missing_ok=True)
        fail(
            f"niezgodna suma SHA-256 dla {dst.name}:\n"
            f"  oczekiwana: {expected_sha}\n"
            f"  pobrana:    {got}\n"
            "Plik NIE został zapisany. Albo źródło się zmieniło (commit jest przypięty — to nie "
            "powinno się zdarzyć), albo config ma błędną sumę. Nie poprawiaj sumy 'na oko'."
        )
    tmp.replace(dst)
    print(f"  pobrano {size:,} B w {time.perf_counter() - t0:.1f} s, SHA-256 zgodna.")


def count_data_rows(path: Path) -> int:
    """Liczy rekordy CSV bez nagłówka (moduł csv — odporny na cytowane pola z przecinkami)."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # nagłówek
        return sum(1 for _ in reader)


def main() -> None:
    # konsola Windows domyślnie cp1250 — polskie znaki w komunikatach wywalałyby print()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Pobranie BANKING77 z przypiętego commitu.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="pobierz od nowa mimo zgodnych sum")
    mode.add_argument("--check", action="store_true", help="tylko zweryfikuj istniejące pliki")
    parser.add_argument(
        "--only",
        choices=("train", "test"),
        help="ogranicz do jednego pliku (np. --only train w CI — bez test.csv)",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        fail(f"brak {CONFIG_PATH.relative_to(REPO_ROOT)}.")
    raw_dir, targets = load_targets(args.only)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Katalog danych: {raw_dir}")

    problems = 0
    for name, url, expected_sha, n_rows in targets:
        dst = raw_dir / name
        print(f"\n[{name}]")
        if dst.exists() and not args.force:
            got = sha256_of(dst)
            if got == expected_sha:
                print("  plik istnieje, SHA-256 zgodna — pomijam (wymuś: --force).")
            elif args.check:
                print(f"  NIEZGODNA suma: {got}")
                problems += 1
                continue
            else:
                print("  plik istnieje, ale suma się NIE zgadza — pobieram ponownie.")
                download(url, dst, expected_sha)
        elif args.check:
            print("  brak pliku.")
            problems += 1
            continue
        else:
            print(f"  pobieram: {url}")
            download(url, dst, expected_sha)

        # suma SHA-256 jest źródłem prawdy; liczba wierszy to sanity check configu (ostrzeżenie)
        rows = count_data_rows(dst)
        marker = "OK" if rows == n_rows else f"UWAGA: config oczekuje {n_rows} — popraw n_rows"
        print(f"  rekordy CSV (bez nagłówka): {rows:,} — {marker}")

    if problems:
        fail(f"{problems} problem(y) — patrz wyżej. Nie ruszaj do EDA na niepewnych danych.")
    scope = f"plik {args.only}.csv obecny" if args.only else "oba pliki obecne"
    print(f"\n[OK] FETCH DATA — {scope}, sumy SHA-256 zgodne z configiem.")
    print("Dane: CC BY 4.0, PolyAI — poza gitem. Następny krok W1: mapa etykiet + make_splits.")


if __name__ == "__main__":
    main()
