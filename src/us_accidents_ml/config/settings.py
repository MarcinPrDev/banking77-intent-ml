"""Wczytywanie konfiguracji projektu z configs/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# repo root = .../us-accidents-ml (settings.py -> config -> us_accidents_ml -> src -> root)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_csv: Path
    processed_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class DataConfig:
    target: str
    sample_rows: int | None
    random_state: int
    test_size: float


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment: str


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    data: DataConfig
    mlflow: MlflowConfig


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Zwraca ustawienia projektu; ścieżki względne rozwiązywane względem korzenia repo."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def _abs(p: str) -> Path:
        candidate = Path(p)
        return candidate if candidate.is_absolute() else _REPO_ROOT / candidate

    return Settings(
        paths=PathsConfig(
            raw_csv=_abs(raw["paths"]["raw_csv"]),
            processed_dir=_abs(raw["paths"]["processed_dir"]),
            models_dir=_abs(raw["paths"]["models_dir"]),
        ),
        data=DataConfig(
            target=raw["data"]["target"],
            sample_rows=raw["data"]["sample_rows"],
            random_state=raw["data"]["random_state"],
            test_size=raw["data"]["test_size"],
        ),
        mlflow=MlflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment=raw["mlflow"]["experiment"],
        ),
    )
