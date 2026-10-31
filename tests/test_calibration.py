"""Testy opisu kalibracji (evaluation/calibration.py) — prerejestracja w configu."""

from __future__ import annotations

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.models.factory import STRATEGIES


def test_preregistration_names_known_models_and_bins():
    """Modele z configu to ścieżki `STRATEGIES` albo kandydaci badań Basala; kNN tylko jako nota."""
    settings = load_settings()
    cfg = settings.evaluation.calibration
    basal = set(settings.basal.models) | set(settings.basal_ultra.models)
    assert cfg.models == ("distilbert", "tfidf_lr", "basal_4_5b", "basal_omni_ultra")
    assert set(cfg.models) <= set(STRATEGIES) | basal
    assert "minilm_knn" not in cfg.models  # n_neighbors 1: score zawsze 1,0
    assert cfg.n_bins == 15
    assert cfg.figure_path.suffix == ".png"
