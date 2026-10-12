"""Factory predyktorów z configu (W2, pkt 5): nazwa strategii -> klasa `IntentClassifier`.

Nowa ścieżka = nowa klasa w `models.classifiers` i jeden wpis w `STRATEGIES` (OCP). Predyktor
buduje się wyłącznie z zamrożonej konfiguracji — pusty `models.<nazwa>.frozen` to błąd, nie cicha
wartość domyślna.
"""

from __future__ import annotations

from pathlib import Path

from banking77_intent_ml.config import Settings
from banking77_intent_ml.models.classifiers import (
    DistilBertClassifier,
    MiniLmKnnClassifier,
    TfidfLrClassifier,
)
from banking77_intent_ml.models.predictor import IntentClassifier

STRATEGIES: dict[str, type[IntentClassifier]] = {  # nazwa = sekcja `models.<nazwa>` w configu
    cls.name: cls for cls in (TfidfLrClassifier, MiniLmKnnClassifier, DistilBertClassifier)
}


def strategy_class(name: str) -> type[IntentClassifier]:
    if name not in STRATEGIES:
        raise ValueError(f"nieznana strategia {name!r}; dostępne: {sorted(STRATEGIES)}")
    return STRATEGIES[name]


def build_predictor(name: str, settings: Settings) -> IntentClassifier:
    """Nieuczony predyktor zamrożonej konfiguracji `models.<name>`."""
    cls = strategy_class(name)
    if getattr(settings.models, name).frozen is None:
        raise ValueError(f"models.{name}.frozen jest puste — najpierw zamrożenie konfiguracji")
    return cls(settings)


def load_predictor(
    name: str, settings: Settings, directory: Path | None = None
) -> IntentClassifier:
    """Predyktor z artefaktu; domyślnie `paths.models_dir/<name>`."""
    return strategy_class(name).load(directory or settings.paths.models_dir / name, settings)
