"""Wspólna ewaluacja wszystkich modeli BANKING77 — jedna funkcja, te same definicje metryk.

Decyzje projektowe autora (6.10.2026):
1. Wejście: id klas 0..n_classes-1 z `configs/label_map.json` (`y_true`, `y_pred`); mapa etykiet
   służy tylko do nazw w raporcie i w macierzy pomyłek. Modele i API mówią językiem id.
2. Metryki wg configu: `accuracy` (`evaluation.primary_metric` — porównanie z literaturą)
   i `macro_f1` (`evaluation.selection_metric` — wybór konfiguracji na walidacji). Liczone po
   WSZYSTKICH klasach z mapy (`labels=range(n_classes)`): klasa bez predykcji albo bez przykładu
   wchodzi do średniej z F1 = 0, zamiast po cichu zniknąć; `zero_division=0`.
3. Raport per klasa (precision, recall, F1, support) i macierz pomyłek n x n z nazwami klas —
   jako tabele, które `evaluation/tracking.py` zapisuje przy runie MLflow jako artefakty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    macro_f1: float
    per_class: pd.DataFrame  # indeks: nazwa klasy (kolejność id); precision/recall/f1/support
    confusion: pd.DataFrame  # wiersze: prawdziwa klasa, kolumny: przewidziana (kolejność id)
    n_samples: int

    def metrics(self) -> dict[str, float]:
        """Metryki pod nazwami z configu (`evaluation.primary_metric`, `selection_metric`)."""
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1}


def class_names(label_map: dict[str, int]) -> list[str]:
    """Nazwy klas w kolejności id 0..n-1."""
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def evaluate(
    y_true: Sequence[int], y_pred: Sequence[int], label_map: dict[str, int]
) -> EvaluationResult:
    """Accuracy, macro-F1, raport per klasa i macierz pomyłek dla id klas z mapy etykiet."""
    names = class_names(label_map)
    labels = list(range(len(names)))
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or true.shape != pred.shape:
        raise ValueError(f"y_true i y_pred: oczekiwano wektorów tej samej długości, "
                         f"jest {true.shape} i {pred.shape}")
    valid = set(labels)
    outside = sorted({repr(v) for v in set(true.tolist()) | set(pred.tolist()) if v not in valid})
    if outside:
        raise ValueError(f"wartości spoza mapy etykiet (id 0..{len(names) - 1}): {outside[:10]}")

    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"id": labels, "precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.Index(names, name="klasa"),
    )
    confusion = pd.DataFrame(
        confusion_matrix(true, pred, labels=labels),
        index=pd.Index(names, name="prawdziwa"),
        columns=pd.Index(names, name="przewidziana"),
    )
    return EvaluationResult(
        accuracy=float(accuracy_score(true, pred)),
        macro_f1=float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        per_class=per_class,
        confusion=confusion,
        n_samples=int(true.size),
    )
