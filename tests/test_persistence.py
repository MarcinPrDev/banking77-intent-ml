"""Testy zapisu i odczytu modelu (models/persistence.py) — N2, decyzje autora 9.10.2026.

Dwa poziomy dowodu „nowy proces daje te same predykcje” (etykiety i prawdopodobieństwa co do
bitu): sztuczne dane — działa też w CI; 20 pierwszych zdań z zamrożonej walidacji — tylko
lokalnie, bo CI nie ma danych. Szkielet: treść testów pisze autor, potem usuwa `skip`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from banking77_intent_ml.config import load_settings

_SETTINGS = load_settings()
HAS_DATA = (_SETTINGS.paths.raw_dir / _SETTINGS.dataset.train.filename).exists()


@pytest.fixture
def toy_texts() -> tuple[pd.Series, pd.Series]:
    """Sztuczne teksty i id klas do treningu małego pipeline'u (3 klasy)."""
    texts = pd.Series([
        "top up failed", "top up pending", "card not arrived", "card arrived late",
        "exchange rate fee", "bad exchange rate",
    ])
    labels = pd.Series([0, 0, 1, 1, 2, 2])
    return texts, labels


@pytest.mark.skip(reason="N2: test pisze autor")
def test_toy_model_gives_identical_predictions_in_new_process(toy_texts, tmp_path):
    """Zapis -> wczytanie w nowym procesie (subprocess) -> te same etykiety i prawdopodobieństwa."""


@pytest.mark.skip(reason="N2: test pisze autor")
@pytest.mark.skipif(not HAS_DATA, reason="brak danych w dane/ (CI) — test tylko lokalny")
def test_validation_sentences_identical_after_reload_in_new_process(tmp_path):
    """TF-IDF+LR z configu, fit na części treningowej; 20 pierwszych zdań z walidacji: te same
    predykcje po wczytaniu w nowym procesie."""


@pytest.mark.skip(reason="N2: test pisze autor")
def test_load_refuses_missing_metadata(toy_texts, tmp_path):
    """Sam plik .joblib bez metadanych -> ModelArtifactError."""


@pytest.mark.skip(reason="N2: test pisze autor")
def test_load_refuses_different_label_map(toy_texts, tmp_path):
    """Metadane z innym hashem mapy etykiet -> ModelArtifactError."""


@pytest.mark.skip(reason="N2: test pisze autor")
def test_load_refuses_different_sklearn_version(toy_texts, tmp_path):
    """Metadane z inną wersją sklearn -> ModelArtifactError."""
