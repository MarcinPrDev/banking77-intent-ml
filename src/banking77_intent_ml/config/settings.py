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
    mlruns: Path  # artefakty runów MLflow (tabele, pliki)
    mlflow_db: Path  # baza SQLite z runami MLflow (parametry, metryki, tagi)


@dataclass(frozen=True)
class SplitConfig:
    val_size: float
    random_state: int
    stratify: bool
    keep_duplicates_together: bool
    dedup_normalization: str
    manifest_path: Path


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    selection_metric: str
    secondary_metrics: tuple[str, ...]
    clean_test_subset_path: Path  # allow-testset
    n_clean_test_expected: int


@dataclass(frozen=True)
class TfidfLrConfig:
    hpo_trials: int


@dataclass(frozen=True)
class MinilmKnnConfig:
    model_id: str
    revision: str
    k_values: tuple[int, ...]
    weights: tuple[str, ...]


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
            mlruns=_abs(raw["paths"]["mlruns"]),
            mlflow_db=_abs(raw["paths"]["mlflow_db"]),
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
        ),
        models=ModelsConfig(
            seed=int(md["seed"]),
            tfidf_lr=TfidfLrConfig(hpo_trials=int(md["tfidf_lr"]["hpo_trials"])),
            minilm_knn=MinilmKnnConfig(
                model_id=md["minilm_knn"]["model_id"],
                revision=str(md["minilm_knn"]["revision"]),
                k_values=tuple(int(k) for k in md["minilm_knn"]["k_values"]),
                weights=tuple(md["minilm_knn"]["weights"]),
            ),
            distilbert=DistilbertConfig(
                model_id=md["distilbert"]["model_id"],
                revision=str(md["distilbert"]["revision"]),
                max_len=int(md["distilbert"]["max_len"]),
                epochs=int(md["distilbert"]["epochs"]),
                lr=float(md["distilbert"]["lr"]),
                batch_size=int(md["distilbert"]["batch_size"]),
                warmup_ratio=float(md["distilbert"]["warmup_ratio"]),
                weight_decay=float(md["distilbert"]["weight_decay"]),
                configs=int(md["distilbert"]["configs"]),
            ),
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
            model_version_env=api["model_version_env"],
            host=str(api["host"]),
            port=int(api["port"]),
        ),
    )
