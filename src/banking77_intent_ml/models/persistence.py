"""Zapis i odczyt wytrenowanych modeli — artefakty do wdrożenia w `paths.models_dir` (N2).

Decyzje autora (9.10.2026):
- format: joblib + obok plik JSON z metadanymi (`<nazwa>.joblib` i `<nazwa>.json`), bez
  kompresji (pomiar asystenta 9.10: `compress=3` zmniejsza plik TF-IDF+LR tylko o ~6%);
- metadane: wersja sklearn, hash mapy etykiet, commit danych, `sha256_val_ids`, id runu MLflow;
- odczyt odmawia (`ModelArtifactError`), gdy mapa etykiet albo wersja sklearn różni się od
  bieżącej — model zwraca same id klas, więc inna mapa = po cichu złe nazwy intencji w API,
  a sklearn przy innej wersji zgłasza tylko ostrzeżenie (`InconsistentVersionWarning`);
- artefakt powstaje z flagą `--save` w `scripts/train_baselines.py` (run MLflow + zapis z jego id).

joblib (pickle) wykonuje kod przy wczytaniu — ładujemy wyłącznie własne artefakty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator

from banking77_intent_ml.config import Settings

MODEL_SUFFIX = ".joblib"
METADATA_SUFFIX = ".json"


class ModelArtifactError(RuntimeError):
    """Artefakt niekompletny albo niezgodny z bieżącym środowiskiem (mapa etykiet, sklearn)."""


def build_metadata(*, model_name: str, run_id: str | None, settings: Settings) -> dict[str, Any]:
    """Metadane artefaktu zapisywane obok modelu.

    TODO(Marcin, N2). Kontrakt:
    - co najmniej: nazwa modelu, id runu MLflow, wersja sklearn, hash mapy etykiet
      (`dataset.label_map_path`), commit danych, `sha256_val_ids` z manifestu;
    - wyłącznie typy JSON (str, int, float, None) — plik czyta też człowiek i API.
    """
    raise NotImplementedError("N2: build_metadata — pisze autor")


def save_model(model: BaseEstimator, path: Path, *, metadata: dict[str, Any]) -> Path:
    """Zapisuje model do `path` (.joblib) i metadane obok (.json); zwraca ścieżkę modelu.

    TODO(Marcin, N2). Kontrakt:
    - tworzy katalog docelowy, jeśli go nie ma;
    - istniejący artefakt: nadpisać czy odmówić — Twoja decyzja.
    """
    raise NotImplementedError("N2: save_model — pisze autor")


def load_model(path: Path, *, settings: Settings) -> BaseEstimator:
    """Wczytuje model po sprawdzeniu metadanych; niezgodność -> `ModelArtifactError`.

    TODO(Marcin, N2). Kontrakt:
    - brak pliku metadanych = błąd (artefakt bez metadanych nie przechodzi);
    - hash mapy etykiet i wersja sklearn z metadanych == bieżące (settings, zainstalowany sklearn);
    - metadane sprawdzane PRZED `joblib.load` — wczytanie pliku wykonuje zapisany w nim kod.
    """
    raise NotImplementedError("N2: load_model — pisze autor")
