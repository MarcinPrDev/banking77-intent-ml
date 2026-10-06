"""Wspólne fixtures testów BANKING77."""

from __future__ import annotations

from dataclasses import replace

import pytest

from banking77_intent_ml.config import load_settings


@pytest.fixture
def tmp_mlflow_settings(tmp_path):
    """Ustawienia z tymczasowym magazynem MLflow (baza SQLite + artefakty) zamiast repo."""
    base = load_settings()
    mlflow_cfg = replace(
        base.mlflow,
        tracking_uri="sqlite:///" + (tmp_path / "mlflow.db").as_posix(),
        artifact_location=(tmp_path / "mlruns").as_uri(),
        experiment="pytest",
    )
    return replace(base, mlflow=mlflow_cfg)
