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
    splits_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class DataConfig:
    target: str
    sample_rows: int | None
    random_state: int


@dataclass(frozen=True)
class DedupConfig:
    time_window_minutes: int
    geohash_precision: int


@dataclass(frozen=True)
class EvaluationConfig:
    test_size: int
    val_size: float
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    llm_eval_sample: int
    dedup: DedupConfig


@dataclass(frozen=True)
class TextConfig:
    effect_phrases: tuple[str, ...]


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment: str


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    data: DataConfig
    forbidden_features: tuple[str, ...]
    evaluation: EvaluationConfig
    text: TextConfig
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
            splits_dir=_abs(raw["paths"]["splits_dir"]),
            models_dir=_abs(raw["paths"]["models_dir"]),
        ),
        data=DataConfig(
            target=raw["data"]["target"],
            sample_rows=raw["data"]["sample_rows"],
            random_state=raw["data"]["random_state"],
        ),
        forbidden_features=tuple(raw["forbidden_features"]),
        evaluation=EvaluationConfig(
            test_size=raw["evaluation"]["test_size"],
            val_size=raw["evaluation"]["val_size"],
            primary_metric=raw["evaluation"]["primary_metric"],
            secondary_metrics=tuple(raw["evaluation"]["secondary_metrics"]),
            llm_eval_sample=raw["evaluation"]["llm_eval_sample"],
            dedup=DedupConfig(
                time_window_minutes=raw["evaluation"]["dedup"]["time_window_minutes"],
                geohash_precision=raw["evaluation"]["dedup"]["geohash_precision"],
            ),
        ),
        text=TextConfig(effect_phrases=tuple(raw["text"]["effect_phrases"] or ())),
        mlflow=MlflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment=raw["mlflow"]["experiment"],
        ),
    )
