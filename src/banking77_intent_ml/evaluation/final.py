"""Ocena końcowa na teście — raz na zamrożoną ścieżkę (W3; W3.md §1.2, ADR-0002).

Decyzje autora (14.10.2026):
- finalny artefakt z `paths.final_models_dir/<nazwa>` (refit na 10 003, po tagu
  `experiments-frozen`);
- jeden przebieg predykcji na pełnym teście (3 080); z TYCH SAMYCH predykcji metryki na 3 080,
  3 055 (bez pokrycia z train — kontrola wrażliwości) i 385 (podzbiór LLM — liczony teraz,
  raportowany dopiero razem z LLM);
- jeden run MLflow na ścieżkę: tag `split=test`, metryki z prefiksami z `SUBSETS`, tabela per
  klasa i macierz pomyłek pełnego testu, predykcje per przykład (`PREDICTION_COLUMNS`) jako
  artefakt — bez tekstów testowych (analiza błędów idzie na walidacji);
- drugi run testowy tej samej ścieżki = odmowa w CLI przed wczytaniem testu.

Podział (decyzja autora 14.10): wczytanie testu z kontrolą podzbiorów, blokady i MLflow —
asystent; `predict_test` i `subset_results` — autor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.labels import LabelMapError, load_label_map
from banking77_intent_ml.data.loader import load_test
from banking77_intent_ml.data.splits import SplitError, ids_sha256
from banking77_intent_ml.evaluation.metrics import EvaluationResult
from banking77_intent_ml.evaluation.tracking import configure_mlflow
from banking77_intent_ml.models.predictor import IntentClassifier

SUBSETS = ("test", "clean", "llm385")  # 3 080, 3 055, 385 — prefiksy metryk w runie
PREDICTION_COLUMNS = ("row_id", "y_true", "y_pred", "score")
PREDICTIONS_ARTIFACT = "predictions_test.csv"
PER_CLASS_ARTIFACT = "per_class_test.json"
CONFUSION_ARTIFACT = "confusion_matrix_test.json"


@dataclass(frozen=True)
class TestSet:
    x: pd.Series  # teksty testu; indeks = pozycja w test.csv; nigdzie ich nie zapisujemy
    y: pd.Series  # id klas z mapy etykiet, ten sam indeks
    clean_ids: list[int]  # 3 055 pozycji bez pokrycia z train (W1, hash z manifestu)
    llm_ids: list[int]  # 385 pozycji podzbioru LLM (W1, hash z manifestu)
    label_map: dict[str, int]


def _subset_ids(path: Path, manifest: dict, hash_key: str, count_key: str) -> list[int]:
    ids = [int(i) for i in json.loads(path.read_text(encoding="utf-8"))]
    if ids_sha256(ids) != manifest[hash_key]:
        raise SplitError(f"{path.name}: hash niezgodny z manifestem ({hash_key})")
    if len(ids) != manifest[count_key]:
        raise SplitError(f"{path.name}: {len(ids)} pozycji, manifest ({count_key}): "
                         f"{manifest[count_key]}")
    return ids


def load_test_set(settings: Settings) -> TestSet:
    """Pełny test (3 080) i podzbiory z W1 z kontrolą hashy — wyłącznie do oceny końcowej."""
    test = load_test(settings)
    label_map = load_label_map(settings.dataset.label_map_path)
    labels = test["category"].map(label_map)
    if labels.isna().any():
        raise LabelMapError(f"{int(labels.isna().sum())} wierszy testu z kategorią spoza mapy")
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    clean_ids = _subset_ids(
        settings.evaluation.clean_test_subset_path, manifest,  # allow-testset
        "sha256_test_clean_ids", "n_test_clean",
    )
    llm_ids = _subset_ids(
        settings.llm.eval_ids_path, manifest, "sha256_llm_eval_ids", "n_llm_eval",  # allow-testset
    )
    outside = sorted((set(clean_ids) | set(llm_ids)) - set(test.index))
    if outside:
        raise SplitError(f"{len(outside)} pozycji podzbiorów spoza testu, np. {outside[:5]}")
    return TestSet(test["text"], labels.astype(int), clean_ids, llm_ids, label_map)


def predict_test(predictor: IntentClassifier, test: TestSet) -> pd.DataFrame:
    """JEDEN przebieg predykcji na pełnym teście -> ramka `PREDICTION_COLUMNS`, wiersz na przykład.

    TODO(Marcin, W3 §1.2): `row_id` = indeks testu, `y_true` z `test.y`; `y_pred` i `score`
    (wynik zwróconej klasy) z jednego wywołania `predict_proba` na wszystkich tekstach.
    """
    raise NotImplementedError("W3 §1.2: predict_test — pisze autor")


def subset_results(predictions: pd.DataFrame, test: TestSet) -> dict[str, EvaluationResult]:
    """Wyniki `evaluate()` z TYCH SAMYCH predykcji dla każdego podzbioru z `SUBSETS`.

    TODO(Marcin, W3 §1.2): `test` = wszystkie wiersze, `clean` = `test.clean_ids`,
    `llm385` = `test.llm_ids` (wybór po `row_id`).
    """
    raise NotImplementedError("W3 §1.2: subset_results — pisze autor")


def count_test_runs(settings: Settings, name: str) -> int:
    """Aktywne runy testowe ścieżki — oficjalna ocena raz, więc > 0 = odmowa w CLI."""
    configure_mlflow(settings)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment],
        filter_string=f"tags.split = 'test' and tags.model_family = '{name}'",
    )
    return len(runs)


def log_test_run(
    name: str,
    predictions: pd.DataFrame,
    results: dict[str, EvaluationResult],
    settings: Settings,
    *,
    artifact_dir: Path,
) -> str:
    """Run MLflow oceny końcowej ścieżki: metryki podzbiorów, tabele 3 080, predykcje."""
    if set(results) != set(SUBSETS):
        raise ValueError(f"oczekiwano wyników dla {SUBSETS}, są: {sorted(results)}")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"predykcje: kolumny {list(predictions.columns)}, "
                         f"oczekiwano {list(PREDICTION_COLUMNS)}")
    manifest = json.loads(settings.split.manifest_path.read_text(encoding="utf-8"))
    configure_mlflow(settings)
    with mlflow.start_run(run_name=f"test-{name}") as run:
        mlflow.set_tags({
            "model_family": name,
            "split": "test",
            "stage": "final",
            "data_commit": settings.dataset.commit,
            "sha256_test_clean_ids": manifest["sha256_test_clean_ids"],
            "sha256_llm_eval_ids": manifest["sha256_llm_eval_ids"],
        })
        mlflow.log_params({
            "artifact_dir": str(artifact_dir),
            **{f"n_{subset}": results[subset].n_samples for subset in SUBSETS},
        })
        mlflow.log_metrics({
            f"{subset}_{metric}": value
            for subset in SUBSETS
            for metric, value in results[subset].metrics().items()
        })
        full = results["test"]
        mlflow.log_table(full.per_class.reset_index(), artifact_file=PER_CLASS_ARTIFACT)
        mlflow.log_table(full.confusion.reset_index(), artifact_file=CONFUSION_ARTIFACT)
        with TemporaryDirectory(prefix="test-predictions-") as temporary:
            path = Path(temporary) / PREDICTIONS_ARTIFACT
            predictions.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        return run.info.run_id
