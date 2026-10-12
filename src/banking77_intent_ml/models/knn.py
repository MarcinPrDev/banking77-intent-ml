"""MiniLM (zamrożone embeddingi) + kNN na zamrożonej walidacji (W2, S2) — run MLflow na punkt.

Siatka z configu: `models.minilm_knn.k_values` × `weights` (budżet = cała siatka, 4 × 2 = 8),
metryka `metric` (cosine). Embeddingi z `features.embeddings` (przypięta rewizja, cache):
klasyfikator uczy się na embeddingach części treningowej, ocena na walidacji przez wspólne
`evaluate()`. Pętla (`run_knn_grid`) to infra; budowa klasyfikatora i wybór zwycięzcy — autor.
Budżet pilnuje CLI: drugie badanie o tej samej nazwie = odmowa przed liczeniem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.evaluation.metrics import EvaluationResult, evaluate
from banking77_intent_ml.evaluation.tracking import log_validation_run
from banking77_intent_ml.features.embeddings import SplitEmbeddings
from banking77_intent_ml.models.hpo import STUDY_TAG, objective_value

STUDY_NAME = "minilm_knn_w2"  # wartość tagu `hpo_study` — jedno badanie = jedna siatka


@dataclass(frozen=True)
class KnnTrial:
    n_neighbors: int
    weights: str
    run_id: str
    result: EvaluationResult
    objective: float  # evaluation.selection_metric na walidacji


def build_knn(n_neighbors: int, weights: str, settings: Settings) -> KNeighborsClassifier:
    """Nieuczony klasyfikator kNN dla jednego punktu siatki.

    TODO(Marcin, W2 S2). Kontrakt:
    - `n_neighbors` i `weights` z argumentów, metryka z `models.minilm_knn.metric` (cosine);
    - pozostałe parametry (np. `algorithm`) — Twoja decyzja; wszystkie trafią do MLflow.
    """
    raise NotImplementedError("W2 S2: build_knn — pisze autor")


def select_best_knn(trials: list[KnnTrial]) -> KnnTrial:
    """Zwycięzca siatki.

    TODO(Marcin, W2 S2). Kontrakt: najwyższy `objective`; regułę remisu ustalasz przed
    uruchomieniem siatki (tak jak w HPO TF-IDF) i opisujesz tutaj.
    """
    raise NotImplementedError("W2 S2: select_best_knn — pisze autor")


def run_knn_grid(data: TrainVal, embeddings: SplitEmbeddings, settings: Settings) -> list[KnnTrial]:
    """Cała siatka k × weights; każdy punkt = osobny run MLflow (`minilm-knn-k{k}-{weights}`)."""
    cfg = settings.models.minilm_knn
    trials: list[KnnTrial] = []
    for n_neighbors in cfg.k_values:
        for weights in cfg.weights:
            model = build_knn(n_neighbors, weights, settings)
            start = time.perf_counter()
            model.fit(embeddings.train, data.y_train)
            fit_seconds = time.perf_counter() - start
            result = evaluate(data.y_val, model.predict(embeddings.val), data.label_map)
            run_id = log_validation_run(
                run_name=f"minilm-knn-k{n_neighbors}-{weights}",
                params={
                    "model": "minilm_knn",
                    "model_id": cfg.model_id,
                    "revision": cfg.revision,
                    **{f"knn__{key}": value for key, value in model.get_params().items()},
                },
                result=result,
                settings=settings,
                tags={"model_family": "minilm_knn", STUDY_TAG: STUDY_NAME},
                fit_seconds=fit_seconds,
                extra_metrics={"encode_train_seconds": embeddings.meta["encode_train_seconds"]},
            )
            trials.append(KnnTrial(
                n_neighbors, weights, run_id, result, objective_value(result, settings)
            ))
    return trials
