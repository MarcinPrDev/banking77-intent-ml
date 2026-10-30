"""Testy dodatku Basal (W5): prerejestracja w configu i pliki opisów etykiet — bez serwera.

Decyzje autora 30.10.2026: 2 modele × 2 warianty opisów, wybór na walidacji, test raz.
"""

from __future__ import annotations

import json

from banking77_intent_ml.config import load_settings


def _label_names(settings) -> list[str]:
    labels = json.loads(settings.dataset.label_map_path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]


def test_preregistration_has_two_models_and_two_label_variants():
    basal = load_settings().basal

    assert basal is not None
    assert sorted(basal.models) == ["basal_11b_fp8", "basal_4_5b"]
    assert sorted(basal.label_variants) == ["v1", "v2"]
    assert len({model.cost_rank for model in basal.models.values()}) == 2  # remis rozstrzygalny
    assert all(len(model.revision) == 40 for model in basal.models.values())  # przypięty commit
    assert 0 < basal.tie_tolerance < 0.01


def test_label_variant_files_cover_label_map_in_id_order():
    settings = load_settings()
    names = _label_names(settings)

    for path in settings.basal.label_variants.values():
        descriptions = json.loads(path.read_text(encoding="utf-8"))
        assert list(descriptions) == names, path.name


def test_v1_labels_are_names_as_phrases():
    settings = load_settings()
    v1 = json.loads(settings.basal.label_variants["v1"].read_text(encoding="utf-8"))

    assert all(text == name.replace("_", " ") for name, text in v1.items())
