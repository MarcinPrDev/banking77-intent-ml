"""Test cienkiego CLI `scripts/train_baselines.py`: `--save` nie trenuje, gdy artefakt istnieje.

Decyzja autora 9.10.2026: artefaktów nie nadpisujemy. CLI sprawdza to przed wczytaniem danych —
inaczej w MLflow zostawałby run bez artefaktu.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from banking77_intent_ml.config import load_settings

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_baselines.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("train_baselines_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_started(*args, **kwargs):
    pytest.fail("dane wczytane mimo istniejącego artefaktu — powstałby run bez zapisu")


@pytest.mark.parametrize("existing", ["tfidf_lr.joblib", "tfidf_lr.json"])
def test_save_refuses_existing_artifact_before_training(tmp_path, monkeypatch, capsys, existing):
    cli = _load_cli()
    base = load_settings()
    settings = replace(base, paths=replace(base.paths, models_dir=tmp_path))
    (tmp_path / existing).write_bytes(b"")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_train_val", _training_started)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "tfidf_lr", "--save"])

    assert exc.value.code == 1
    assert "już istnieje" in capsys.readouterr().out
