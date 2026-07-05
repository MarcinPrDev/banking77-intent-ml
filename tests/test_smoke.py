"""Testy dymne szkieletu projektu."""

from pathlib import Path

import us_accidents_ml
from us_accidents_ml.config import load_settings


def test_package_imports() -> None:
    assert us_accidents_ml.__version__


def test_settings_load() -> None:
    settings = load_settings()
    assert settings.data.target == "Severity"
    assert isinstance(settings.paths.raw_csv, Path)
    assert settings.evaluation.test_size == 50000
    assert settings.evaluation.primary_metric == "macro_f1"
    assert "Distance(mi)" in settings.forbidden_features
