"""CLI eksportu (scripts/export_model.py): istniejący artefakt = odmowa przed wczytaniem danych."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from banking77_intent_ml.config import load_settings

_CLI = Path(__file__).resolve().parents[1] / "scripts" / "export_model.py"


def test_cli_refuses_existing_artifact_before_loading_data(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("export_model_cli", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    settings = load_settings()
    settings = replace(settings, paths=replace(settings.paths, models_dir=tmp_path))
    (tmp_path / "tfidf_lr").mkdir()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli, "load_train_val", lambda *args: pytest.fail("dane wczytane mimo artefaktu")
    )
    monkeypatch.setattr("sys.argv", ["export_model.py", "tfidf_lr"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "już istnieje" in capsys.readouterr().out
