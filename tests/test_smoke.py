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
    assert settings.data.test_size == 0.2
