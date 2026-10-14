"""Wczytywanie konfiguracji projektu z configs/config.yaml (BANKING77, stan 30.09.2026).

Dataclassy są zamrożone (frozen) i odwzorowują sekcje YAML 1:1. Brak klucza w YAML-u
= KeyError przy starcie, a nie cicha wartość domyślna ujawniona w środku treningu.
Liczby protokołu (seed, val_size, budżety strojenia) żyją WYŁĄCZNIE w configu —
kod i dokumenty czytają je stąd.

Który config: argument `config_path` > zmienna środowiskowa `BANKING77_CONFIG` > domyślny
`<repo>/configs/config.yaml`. Ścieżki względne z configu rozwiązywane są względem KORZENIA
PROJEKTU wyprowadzonego z położenia pliku configu (`<korzeń>/configs/config.yaml`), a nie
z położenia tego modułu — w obrazie Docker (instalacja nieedytowalna, moduł w site-packages)
wystarczy `COPY configs/ /app/configs/` + `ENV BANKING77_CONFIG=/app/configs/config.yaml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# repo root = .../banking77-intent-ml (settings.py -> config -> banking77_intent_ml -> src -> root);
# poprawne tylko dla checkoutu / instalacji edytowalnej — stąd override przez env (docstring).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"
CONFIG_ENV_VAR = "BANKING77_CONFIG"


@dataclass(frozen=True)
class DatasetFileConfig:
    path: str  # ścieżka w repozytorium źródłowym, np. banking_data/train.csv
    sha256: str
    n_rows: int  # oczekiwana liczba wierszy (audyt 30.09) — do sprawdzenia w loaderze

    @property
    def filename(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    repo: str
    commit: str
    base_url: str
    license: str
    n_classes: int
    train: DatasetFileConfig
    test: DatasetFileConfig
    label_map_path: Path

    def raw_url(self, file: DatasetFileConfig) -> str:
        """URL surowego pliku dla przypiętego commitu (raw.githubusercontent.com)."""
        return f"{self.base_url}/{self.commit}/{file.path}"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    splits_dir: Path
    models_dir: Path
    final_models_dir: Path  # refit na całym train (10 003) przed oceną na teście (W3)
    mlruns: Path  # artefakty runów MLflow (tabele, pliki)
    mlflow_db: Path  # baza SQLite z runami MLflow (parametry, metryki, tagi)
    cache_dir: Path  # dane pochodne (embeddingi) — poza gitem, odtwarzalne


@dataclass(frozen=True)
class SplitConfig:
    val_size: float
    random_state: int
    stratify: bool
    keep_duplicates_together: bool
    dedup_normalization: str
    manifest_path: Path


@dataclass(frozen=True)
class CostConfig:
    threads: tuple[int, ...]  # liczby wątków CPU, każda mierzona osobno
    n_texts: int  # zdania z walidacji (seed `models.seed`)
    passes: int  # przebiegi po zdaniach w pomiarze latencji rozgrzanej
    warmup: int  # predykcje rozgrzewki przed pomiarem
    cold_runs: int  # nowe procesy na ścieżkę w pomiarze zimnego startu


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    selection_metric: str
    secondary_metrics: tuple[str, ...]
    clean_test_subset_path: Path  # allow-testset
    n_clean_test_expected: int
    cost: CostConfig


@dataclass(frozen=True)
class TfidfLrHpoConfig:
    sampler: str  # "tpe" (decyzja autora 9.10)
    tie_tolerance: float  # remis w celu HPO poniżej tej różnicy -> mniej cech
    space: dict[str, Any]  # przestrzeń z configu; kształt czyta `models.hpo.suggest_tfidf_lr`


@dataclass(frozen=True)
class TfidfLrConfig:
    hpo_trials: int
    # Parametry konstruktorów sklearn przekazywane 1:1: `TfidfVectorizer(**tfidf)`,
    # `LogisticRegression(**logreg)` — pierwsza konfiguracja N1 i start HPO w W2. Słowniki są
    # współdzielone przez cały proces: nie modyfikuj w miejscu, buduj kopię ({**logreg, "C": c}).
    tfidf: dict[str, Any]
    logreg: dict[str, Any]
    # W YAML-u oba klucze są wymagane (KeyError przy braku); domyślne wartości tylko dla
    # konfiguracji budowanych w kodzie (np. próby HPO, testy).
    hpo: TfidfLrHpoConfig | None = None
    frozen: dict[str, dict[str, Any]] | None = None  # zwycięzca HPO {tfidf, logreg} albo None


@dataclass(frozen=True)
class MinilmKnnConfig:
    model_id: str
    revision: str
    k_values: tuple[int, ...]
    weights: tuple[str, ...]
    metric: str = "cosine"  # w YAML-u wymagane; domyślne tylko dla konfiguracji budowanych w kodzie
    frozen: dict[str, Any] | None = None  # zwycięzca siatki {n_neighbors, weights} albo None


@dataclass(frozen=True)
class DistilbertConfig:
    model_id: str
    revision: str
    max_len: int
    epochs: int
    lr: float
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    configs: int
    # W YAML-u wymagane; domyślne tylko dla konfiguracji budowanych w kodzie (testy).
    variants: tuple[dict[str, Any], ...] = ()  # nadpisania bazy (np. {"lr": 2e-5}), len == configs
    tie_tolerance: float = 0.001
    frozen: dict[str, Any] | None = None  # zwycięzca {lr, epoch} albo None


def _distilbert(raw: dict) -> DistilbertConfig:
    variants = tuple(
        {key: float(value) if key == "lr" else value for key, value in variant.items()}
        for variant in raw["variants"]
    )
    if len(variants) != int(raw["configs"]):
        raise ValueError(
            f"models.distilbert: {len(variants)} wariantów, a budżet configs = {raw['configs']}"
        )
    return DistilbertConfig(
        model_id=raw["model_id"],
        revision=str(raw["revision"]),
        max_len=int(raw["max_len"]),
        epochs=int(raw["epochs"]),
        lr=float(raw["lr"]),
        batch_size=int(raw["batch_size"]),
        warmup_ratio=float(raw["warmup_ratio"]),
        weight_decay=float(raw["weight_decay"]),
        configs=int(raw["configs"]),
        variants=variants,
        tie_tolerance=float(raw["tie_tolerance"]),
        frozen=raw["frozen"],
    )


@dataclass(frozen=True)
class ModelsConfig:
    seed: int
    tfidf_lr: TfidfLrConfig
    minilm_knn: MinilmKnnConfig
    distilbert: DistilbertConfig


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool
    model: str
    n_eval: int
    per_class: int
    eval_ids_path: Path  # allow-testset


@dataclass(frozen=True)
class MlflowConfig:
    # Pola WYPROWADZONE z `paths`, nie przepisane z YAML (jedno źródło ścieżek; decyzja 6.10.2026):
    # - tracking_uri = `sqlite:///<bezwzględna paths.mlflow_db>` — MLflow 3.14 odrzuca magazyn
    #   plikowy, a bez jawnego URI zapisuje do `mlflow.db` w cwd procesu (np. w notebooks/);
    # - artifact_location = bezwzględny `file:///…/mlruns` dla eksperymentu — inaczej artefakty
    #   lądowałyby względem cwd procesu.
    # Użycie: `configure_mlflow(settings)` z `evaluation/tracking.py` na starcie skryptu/notebooka.
    tracking_uri: str
    artifact_location: str
    experiment: str


@dataclass(frozen=True)
class ApiConfig:
    model: str  # strategia z `models.factory.STRATEGIES`; artefakt w `paths.models_dir/<model>`
    max_text_chars: int  # dłuższy tekst -> 422 (pusty i same spacje też)
    model_version_env: str
    host: str
    port: int


@dataclass(frozen=True)
class Settings:
    dataset: DatasetConfig
    paths: PathsConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    models: ModelsConfig
    llm: LlmConfig
    mlflow: MlflowConfig
    api: ApiConfig


def _resolve_config_path(config_path: str | Path | None) -> Path:
    """Argument > zmienna środowiskowa BANKING77_CONFIG > domyślny configs/config.yaml."""
    chosen = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    return Path(chosen).resolve()


def _file(raw: dict) -> DatasetFileConfig:
    return DatasetFileConfig(path=raw["path"], sha256=raw["sha256"], n_rows=int(raw["n_rows"]))


def _sklearn_params(raw: dict) -> dict[str, Any]:
    """Parametry konstruktora sklearn z YAML-a; `ngram_range` jako krotka.

    YAML nie ma krotek, a sklearn 1.9 odrzuca `ngram_range=[1, 1]` (InvalidParameterError
    przy fit). Pozostałe listy zostają listami — np. własna lista `stop_words` musi być listą.
    """
    params = dict(raw)
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])
    return params


def _tfidf_lr(raw: dict) -> TfidfLrConfig:
    hpo, frozen = raw["hpo"], raw["frozen"]
    return TfidfLrConfig(
        hpo_trials=int(raw["hpo_trials"]),
        tfidf=_sklearn_params(raw["tfidf"]),
        logreg=_sklearn_params(raw["logreg"]),
        hpo=TfidfLrHpoConfig(
            sampler=str(hpo["sampler"]),
            tie_tolerance=float(hpo["tie_tolerance"]),
            space=dict(hpo["space"] or {}),
        ),
        frozen=None if frozen is None else {
            "tfidf": _sklearn_params(frozen["tfidf"]),
            "logreg": _sklearn_params(frozen["logreg"]),
        },
    )


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Zwraca ustawienia projektu wczytane z YAML (wybór pliku: docstring modułu)."""
    path = _resolve_config_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent  # <korzeń>/configs/config.yaml -> <korzeń>

    def _abs(p: str) -> Path:
        """Ścieżki względne z configu rozwiązywane względem korzenia projektu."""
        candidate = Path(p)
        return candidate if candidate.is_absolute() else root / candidate

    ds, sp, ev, md, llm, api = (
        raw["dataset"],
        raw["split"],
        raw["evaluation"],
        raw["models"],
        raw["llm"],
        raw["api"],
    )

    return Settings(
        dataset=DatasetConfig(
            name=ds["name"],
            repo=ds["repo"],
            commit=ds["commit"],
            base_url=ds["base_url"].rstrip("/"),
            license=ds["license"],
            n_classes=int(ds["n_classes"]),
            train=_file(ds["files"]["train"]),
            test=_file(ds["files"]["test"]),
            label_map_path=_abs(ds["label_map_path"]),
        ),
        paths=PathsConfig(
            raw_dir=_abs(raw["paths"]["raw_dir"]),
            splits_dir=_abs(raw["paths"]["splits_dir"]),
            models_dir=_abs(raw["paths"]["models_dir"]),
            final_models_dir=_abs(raw["paths"]["final_models_dir"]),
            mlruns=_abs(raw["paths"]["mlruns"]),
            mlflow_db=_abs(raw["paths"]["mlflow_db"]),
            cache_dir=_abs(raw["paths"]["cache_dir"]),
        ),
        split=SplitConfig(
            val_size=float(sp["val_size"]),
            random_state=int(sp["random_state"]),
            stratify=bool(sp["stratify"]),
            keep_duplicates_together=bool(sp["keep_duplicates_together"]),
            dedup_normalization=sp["dedup_normalization"],
            manifest_path=_abs(sp["manifest_path"]),
        ),
        evaluation=EvaluationConfig(
            primary_metric=ev["primary_metric"],
            selection_metric=ev["selection_metric"],
            secondary_metrics=tuple(ev["secondary_metrics"]),
            clean_test_subset_path=_abs(ev["clean_test_subset_path"]),  # allow-testset
            n_clean_test_expected=int(ev["n_clean_test_expected"]),
            cost=CostConfig(
                threads=tuple(int(threads) for threads in ev["cost"]["threads"]),
                n_texts=int(ev["cost"]["n_texts"]),
                passes=int(ev["cost"]["passes"]),
                warmup=int(ev["cost"]["warmup"]),
                cold_runs=int(ev["cost"]["cold_runs"]),
            ),
        ),
        models=ModelsConfig(
            seed=int(md["seed"]),
            tfidf_lr=_tfidf_lr(md["tfidf_lr"]),
            minilm_knn=MinilmKnnConfig(
                model_id=md["minilm_knn"]["model_id"],
                revision=str(md["minilm_knn"]["revision"]),
                k_values=tuple(int(k) for k in md["minilm_knn"]["k_values"]),
                weights=tuple(md["minilm_knn"]["weights"]),
                metric=str(md["minilm_knn"]["metric"]),
                frozen=md["minilm_knn"]["frozen"],
            ),
            distilbert=_distilbert(md["distilbert"]),
        ),
        llm=LlmConfig(
            enabled=bool(llm["enabled"]),
            model=llm["model"],
            n_eval=int(llm["n_eval"]),
            per_class=int(llm["per_class"]),
            eval_ids_path=_abs(llm["eval_ids_path"]),  # allow-testset
        ),
        mlflow=MlflowConfig(
            tracking_uri="sqlite:///" + _abs(raw["paths"]["mlflow_db"]).as_posix(),
            artifact_location=_abs(raw["paths"]["mlruns"]).as_uri(),
            experiment=raw["mlflow"]["experiment"],
        ),
        api=ApiConfig(
            model=str(api["model"]),
            max_text_chars=int(api["max_text_chars"]),
            model_version_env=api["model_version_env"],
            host=str(api["host"]),
            port=int(api["port"]),
        ),
    )
