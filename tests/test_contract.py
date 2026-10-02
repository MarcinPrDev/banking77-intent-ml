"""Strażnik protokołu ewaluacji BANKING77 (stan 30.09.2026).

Test infra, szybki (<1 s), bez sieci i bez danych — uruchamia go hook PostToolUse po każdej
edycji w src/, configs/ i tests/ (czerwony = exit 2 = blokada). Zastępuje poprzedni test
kontraktu predykcji US Accidents (grep po `forbidden_features`).

Co ten test DOWODZI:
1. `configs/config.yaml` przypina źródło danych: commit (40 hex) i sumy SHA-256 (64 hex) obu
   plików — nikt nie podmieni danych "po cichu" bez zmiany w gicie.
2. Żaden moduł w `src/` poza zwolnionymi nie odwołuje się LITERALNIE do pliku testowego ani do
   plików od niego pochodnych (`test.csv`, nazwa pliku testowego z configu, `dataset.test`,
   nazwy plików `evaluation.clean_test_subset_path` i `llm.eval_ids_path`). Zwolnienie dotyczy
   KATALOGU lub MODUŁU o nazwie dokładnie `evaluation`, `data` albo `audit` (części ścieżki
   względem `src/` i nazwa pliku bez `.py`) — nie dowolnej ścieżki zawierającej ten ciąg
   (`models/metadata.py` NIE jest zwolniony). Kod modeli/cech/API nie powinien w ogóle wiedzieć,
   że test istnieje. Świadome odstępstwo: komentarz `# allow-testset` w tej samej linii.
3. Jeśli istnieje `configs/split_manifest.json` (powstaje w W1 z `make_splits`), ma wymagane
   klucze (`seed`, `val_size`, `n_train`, `n_val`, `sha256_val_ids`), `seed` i `val_size` zgadzają
   się z configiem (`split.random_state`, `split.val_size`), `sha256_val_ids` jest 64-hex,
   a `n_train + n_val` (+ opcjonalne `n_excluded`, domyślnie 0) równa się
   `dataset.files.train.n_rows` — grupowany podział nie gubi ani nie dubluje wierszy; wiersze
   wyłączone świadomie muszą być zadeklarowane w manifeście, nie znikać po cichu.
4. Jeśli istnieje `configs/label_map.json` (powstaje w W1), ma dokładnie `dataset.n_classes`
   unikalnych nazw, a id to dokładnie 0..n_classes-1 bez luk i powtórzeń. Akceptowane formaty:
   obiekt JSON `{nazwa: id}` albo lista nazw (pozycja = id).
5. `split.val_size` mieści się w (0, 0.3): walidacja ~15% z 10 003 wierszy; poza tym zakresem
   albo brakuje danych treningowych, albo walidacja jest za mała na 77 klas.

Czego ten test NIE DOWODZI (i co pilnuje człowiek + /review-ml):
- że test set nie uczestniczy w fit/HPO/doborze promptu pośrednio (wczytanie przez zmienną,
  f-string, konkatenację, DataFrame przekazany z zewnątrz) — skan jest literalny;
- że indeksy w `dane/splits/` faktycznie odpowiadają manifestowi (to sprawdza osobny test
  z danymi, uruchamiany ręcznie po `fetch_data` + `make_splits`);
- że TF-IDF/tokenizer był dopasowany tylko na train — to własność kodu pipeline'u, nie pliku;
- poprawności sum plików na dysku (to `scripts/fetch_data.py --check`) ani tego, że nazwy
  w mapie etykiet zgadzają się z danymi (test bez CSV).
Skanowany jest tylko `src/`; notebooki i `scripts/` są poza kontrolą tego testu — DLATEGO logika
treningu/HPO/ewaluacji/pomiarów ma żyć w `src/banking77_intent_ml/...`, a w `scripts/` tylko cienkie
CLI (`fetch_data`, `smoke_api`, `demo`). Hook też patrzy tylko na `src|configs|tests`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
MANIFEST_PATH = ROOT / "configs" / "split_manifest.json"
LABEL_MAP_PATH = ROOT / "configs" / "label_map.json"
SRC_DIR = ROOT / "src"

ALLOW_MARKER = "allow-testset"
# Dokładne nazwy katalogów (części ścieżki względem src/) lub modułów (nazwa pliku bez .py),
# które MOGĄ znać plik testowy: ewaluacja końcowa, loader/audyt danych.
EXEMPT_PATH_PARTS = frozenset({"evaluation", "data", "audit"})
MANIFEST_REQUIRED_KEYS = {"seed", "val_size", "n_train", "n_val", "sha256_val_ids"}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# (a) przypięte źródło danych ------------------------------------------------------------


def test_dataset_commit_is_pinned_40_hex():
    commit = str(_config()["dataset"]["commit"])
    assert _HEX40.match(commit), f"dataset.commit nie jest 40-znakowym hashem hex: {commit!r}"


def test_dataset_files_have_sha256_64_hex():
    files = _config()["dataset"]["files"]
    assert {"train", "test"} <= set(files), "dataset.files musi zawierać train i test"
    for name in ("train", "test"):
        digest = str(files[name]["sha256"])
        assert _HEX64.match(digest), f"dataset.files.{name}.sha256 nie jest 64-hex: {digest!r}"
    assert files["train"]["sha256"] != files["test"]["sha256"], "train i test mają tę samą sumę"


# (b) kod modelowy nie zna pliku testowego -----------------------------------------------


def _forbidden_testset_literals(cfg: dict) -> set[str]:
    """Literały, których kod poza evaluation/data/audit nie ma prawa zawierać."""
    literals = {"test.csv", "dataset.test"}
    literals.add(Path(str(cfg["dataset"]["files"]["test"]["path"])).name)
    for section, key in (("evaluation", "clean_test_subset_path"), ("llm", "eval_ids_path")):
        derived = cfg.get(section, {}).get(key)
        if derived:
            literals.add(Path(str(derived)).name)
    return literals


def _is_exempt(py: Path) -> bool:
    """Zwolnione: pliki w katalogu evaluation/, data/ lub audit/ (dowolny poziom pod src/)
    albo moduły evaluation.py / data.py / audit.py. Dopasowanie po CAŁYCH częściach ścieżki,
    nie po podciągu — `models/metadata.py` nie jest zwolniony."""
    rel = py.relative_to(SRC_DIR)
    parts = set(rel.parts[:-1]) | {rel.stem}
    return bool(parts & EXEMPT_PATH_PARTS)


def test_is_exempt_matches_whole_path_parts_only():
    """Samokontrola reguły zwolnienia (bez plików na dysku)."""
    pkg = SRC_DIR / "banking77_intent_ml"
    assert _is_exempt(pkg / "evaluation" / "metrics.py")
    assert _is_exempt(pkg / "data" / "loader.py")
    assert _is_exempt(pkg / "audit.py")
    assert not _is_exempt(pkg / "models" / "metadata.py")  # "data" jako podciąg nie zwalnia
    assert not _is_exempt(pkg / "api" / "data_models.py")
    assert not _is_exempt(pkg / "features" / "dataset_stats.py")


def test_src_outside_evaluation_data_audit_does_not_reference_testset():
    literals = _forbidden_testset_literals(_config())
    offenders: list[str] = []
    for py in SRC_DIR.rglob("*.py"):
        if _is_exempt(py):
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            for lit in literals:
                if lit in line:
                    offenders.append(f"{py.relative_to(ROOT)}:{lineno}: {lit!r}")
    assert not offenders, (
        "Odwołanie do pliku testowego (lub pliku od niego pochodnego) poza modułami "
        f"{sorted(EXEMPT_PATH_PARTS)} (świadome odstępstwo: `# {ALLOW_MARKER}` w tej linii):\n"
        + "\n".join(offenders)
    )


# (c) manifest splitu zgodny z configiem -------------------------------------------------


def test_split_manifest_if_present_matches_config():
    if not MANIFEST_PATH.exists():
        return  # manifest powstaje w W1 (make_splits); przed tym test jest pusty, nie czerwony
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    missing = MANIFEST_REQUIRED_KEYS - set(manifest)
    assert not missing, f"split_manifest.json bez kluczy: {sorted(missing)}"

    cfg = _config()
    split = cfg["split"]
    assert int(manifest["seed"]) == int(split["random_state"]), (
        f"manifest.seed={manifest['seed']} != config.split.random_state={split['random_state']}"
    )
    assert abs(float(manifest["val_size"]) - float(split["val_size"])) < 1e-9, (
        f"manifest.val_size={manifest['val_size']} != config.split.val_size={split['val_size']}"
    )
    assert _HEX64.match(str(manifest["sha256_val_ids"])), "sha256_val_ids nie jest 64-hex"

    n_train, n_val = int(manifest["n_train"]), int(manifest["n_val"])
    n_excluded = int(manifest.get("n_excluded", 0))
    assert n_train > n_val > 0, f"n_train={n_train}, n_val={n_val}: oczekiwano n_train > n_val > 0"
    assert n_excluded >= 0, f"n_excluded={n_excluded} < 0"
    n_expected = int(cfg["dataset"]["files"]["train"]["n_rows"])
    assert n_train + n_val + n_excluded == n_expected, (
        f"n_train+n_val+n_excluded = {n_train}+{n_val}+{n_excluded} != "
        f"dataset.files.train.n_rows={n_expected} — podział zgubił lub zdublował wiersze "
        "(wiersze wyłączone świadomie deklaruj w manifeście jako n_excluded)"
    )


# (d) mapa etykiet: n_classes unikalnych nazw, id 0..n_classes-1 --------------------------


def _label_pairs(raw: object) -> list[tuple[str, int]]:
    """Normalizuje oba akceptowane formaty do listy (nazwa, id)."""
    if isinstance(raw, dict):
        return [(str(name), int(idx)) for name, idx in raw.items()]
    if isinstance(raw, list):
        return [(str(name), idx) for idx, name in enumerate(raw)]
    raise AssertionError(
        "label_map.json: oczekiwano obiektu {nazwa: id} albo listy nazw, "
        f"jest {type(raw).__name__}"
    )


def test_label_map_if_present_is_bijection_0_to_n_classes():
    if not LABEL_MAP_PATH.exists():
        return  # mapa powstaje w W1; przed tym test jest pusty, nie czerwony
    n_classes = int(_config()["dataset"]["n_classes"])
    pairs = _label_pairs(json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8")))
    names = [name for name, _ in pairs]
    ids = [idx for _, idx in pairs]
    assert len(pairs) == n_classes, f"label_map.json ma {len(pairs)} wpisów, config: {n_classes}"
    assert len(set(names)) == n_classes, "label_map.json: powtórzone nazwy kategorii"
    assert all(name.strip() for name in names), "label_map.json: pusta nazwa kategorii"
    assert sorted(ids) == list(range(n_classes)), (
        f"label_map.json: id muszą być dokładnie 0..{n_classes - 1} bez luk i powtórzeń"
    )


# (e) rozsądny rozmiar walidacji ---------------------------------------------------------


def test_val_size_in_sane_range():
    val_size = float(_config()["split"]["val_size"])
    assert 0.0 < val_size < 0.3, f"split.val_size={val_size} poza (0, 0.3)"
