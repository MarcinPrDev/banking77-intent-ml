"""Zapis i odczyt wytrenowanych modeli — artefakty do wdrożenia w `paths.models_dir` (N2).

Decyzje autora (9.10.2026):
- format: joblib + obok plik JSON z metadanymi (`<nazwa>.joblib` i `<nazwa>.json`), bez
  kompresji (pomiar asystenta 9.10: `compress=3` zmniejsza plik TF-IDF+LR tylko o ~6%);
- metadane: wersje sklearn i numpy, hash mapy etykiet, commit danych, `sha256_val_ids`, id runu;
- odczyt odmawia (`ModelArtifactError`), gdy mapa etykiet albo wersja sklearn różni się od
  bieżącej — model zwraca same id klas, więc inna mapa = po cichu złe nazwy intencji w API,
  a sklearn przy innej wersji zgłasza tylko ostrzeżenie (`InconsistentVersionWarning`);
- artefakt powstaje z flagą `--save` w `scripts/train_baselines.py` (run MLflow + zapis z jego id).

joblib (pickle) wykonuje kod przy wczytaniu — ładujemy wyłącznie własne artefakty.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator

from banking77_intent_ml.config import Settings

MODEL_SUFFIX = ".joblib"
METADATA_SUFFIX = ".json"


class ModelArtifactError(RuntimeError):
    """Artefakt niekompletny albo niezgodny z bieżącym środowiskiem (mapa etykiet, sklearn)."""


def label_map_sha256(path: Path) -> str:
    """Wspólny hash mapy dla artefaktów; kolejność kluczy i format JSON nie mają znaczenia."""
    label_map = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(label_map, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_metadata(*, model_name: str, run_id: str | None, settings: Settings) -> dict[str, Any]:
    """Metadane artefaktu zapisywane obok modelu.

    Kontrakt:
    - co najmniej: nazwa modelu, id runu MLflow, wersje sklearn i numpy, hash mapy etykiet
      (`dataset.label_map_path`), commit danych, `sha256_val_ids` z manifestu;
    - wyłącznie typy JSON (str, int, float, None) — plik czyta też człowiek i API.
    """
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    return {
        "model_name": model_name,
        "run_id": run_id,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "label_map_sha256": label_map_sha256(settings.dataset.label_map_path),
        "data_commit": settings.dataset.commit,
        "sha256_val_ids": manifest["sha256_val_ids"],
    }


def save_model(model: BaseEstimator, path: Path, *, metadata: dict[str, Any]) -> Path:
    """Zapisuje model do `path` (.joblib) i metadane obok (.json); zwraca ścieżkę modelu.

    Kontrakt:
    - tworzy katalog docelowy, jeśli go nie ma;
    - nie nadpisuje modelu ani metadanych — istniejący plik to `FileExistsError`;
    - po błędzie zapisu modelu usuwa utworzony plik i przekazuje wyjątek dalej.
    """
    if path.suffix != MODEL_SUFFIX:
        raise ValueError(f"oczekiwano ścieżki {MODEL_SUFFIX}: {path}")
    metadata_path = path.with_suffix(METADATA_SUFFIX)
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"artefakt już istnieje: {path} lub {metadata_path}")
    text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Otwarcie poza try: jeśli plik już istnieje, nie możemy go usunąć przy sprzątaniu.
    file = path.open("xb")
    try:
        with file:
            joblib.dump(model, file, compress=0)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    with metadata_path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(text)
    return path


def load_model(path: Path, *, settings: Settings) -> BaseEstimator:
    """Wczytuje model po sprawdzeniu metadanych; niezgodność -> `ModelArtifactError`.

    Kontrakt:
    - brak pliku metadanych = błąd (artefakt bez metadanych nie przechodzi);
    - hash mapy etykiet i wersja sklearn z metadanych == bieżące (settings, zainstalowany sklearn);
    - metadane sprawdzane PRZED `joblib.load` — wczytanie pliku wykonuje zapisany w nim kod.

    `data_commit`, `sha256_val_ids` i `numpy_version` są tylko śladem pochodzenia artefaktu
    i środowiska treningu — nie są sprawdzane przy odczycie.
    """
    metadata_path = path.with_suffix(METADATA_SUFFIX)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"nie można odczytać metadanych: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ModelArtifactError(f"plik metadanych musi zawierać obiekt JSON: {metadata_path}")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ModelArtifactError(
            f"niezgodna wersja sklearn: artefakt {metadata.get('sklearn_version')}, "
            f"środowisko {sklearn.__version__}"
        )
    try:
        label_hash = label_map_sha256(settings.dataset.label_map_path)
    except (OSError, ValueError) as exc:
        raise ModelArtifactError("nie można sprawdzić mapy etykiet") from exc
    if metadata.get("label_map_sha256") != label_hash:
        raise ModelArtifactError("hash mapy etykiet niezgodny z artefaktem")
    if not path.is_file():
        raise ModelArtifactError(f"brak pliku modelu: {path}")
    return joblib.load(path)
