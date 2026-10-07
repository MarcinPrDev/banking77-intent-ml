"""Testy dymne szkieletu projektu (BANKING77, stan 30.09.2026).

Sprawdzają TYPY i strukturę ustawień, nie konkretne liczby protokołu (te żyją wyłącznie
w `configs/config.yaml`; zakresy, których nie wolno przekroczyć, pilnuje `tests/test_contract.py`).
Wyjątek: stałe wynikające z definicji zadania (77 klas, nazwy plików danych).
"""

import warnings
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import banking77_intent_ml
from banking77_intent_ml.config import load_settings


def test_package_imports() -> None:
    assert banking77_intent_ml.__version__


def test_settings_load() -> None:
    s = load_settings()

    # dane przypięte
    assert s.dataset.name == "banking77"
    assert s.dataset.n_classes == 77
    assert len(s.dataset.commit) == 40
    assert s.dataset.train.filename == "train.csv"
    assert s.dataset.test.filename == "test.csv"
    assert s.dataset.raw_url(s.dataset.train).startswith("https://raw.githubusercontent.com/")
    assert s.dataset.commit in s.dataset.raw_url(s.dataset.test)

    # ścieżki rozwiązane do absolutnych
    assert isinstance(s.paths.raw_dir, Path) and s.paths.raw_dir.is_absolute()
    assert s.split.manifest_path.name == "split_manifest.json"

    # MLflow (decyzja 6.10): runy w SQLite z paths.mlflow_db, artefakty w paths.mlruns — oba
    # jako bezwzględne URI, nie gołe napisy rozwiązywane względem cwd
    assert s.paths.mlruns.is_absolute() and s.paths.mlflow_db.is_absolute()
    assert s.mlflow.tracking_uri == "sqlite:///" + s.paths.mlflow_db.as_posix()
    assert s.mlflow.artifact_location == s.paths.mlruns.as_uri()
    assert s.mlflow.experiment

    # protokół — typy/zakresy, nie wartości (wartości: config; zakres val_size: test kontraktu)
    assert isinstance(s.split.val_size, float) and 0.0 < s.split.val_size < 1.0
    assert isinstance(s.split.random_state, int)
    assert s.split.keep_duplicates_together is True
    assert s.evaluation.primary_metric == "accuracy"
    assert s.evaluation.selection_metric == "macro_f1"

    # hiperparametry startowe — typy (PyYAML potrafi wczytać "4e-5" jako string)
    assert isinstance(s.models.distilbert.lr, float)
    assert isinstance(s.models.distilbert.max_len, int) and s.models.distilbert.max_len > 0
    assert all(isinstance(k, int) for k in s.models.minilm_knn.k_values)
    # TF-IDF + LR (decyzja 7.10): słowniki parametrów sklearn; YAML nie ma krotek
    assert isinstance(s.models.tfidf_lr.tfidf["ngram_range"], tuple)
    assert isinstance(s.models.tfidf_lr.logreg, dict)

    # LLM = stretch, domyślnie wyłączony; model to placeholder do decyzji Marcina (niepusty string)
    assert s.llm.enabled is False
    assert isinstance(s.llm.model, str) and s.llm.model
    assert s.llm.n_eval == s.llm.per_class * s.dataset.n_classes


def test_tfidf_lr_config_is_valid_for_installed_sklearn() -> None:
    """Parametry `models.tfidf_lr` przechodzą walidację sklearn (fit na zabawkowych tekstach).

    Łapie literówkę w kluczu (TypeError), listę zamiast krotki (InvalidParameterError)
    i parametr przestarzały w zainstalowanej wersji biblioteki (FutureWarning -> błąd).
    """
    cfg = load_settings().models.tfidf_lr
    texts = ["top up failed", "top up pending", "top up declined",
             "card not arrived", "card arrived late", "card still not arrived"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        features = TfidfVectorizer(**cfg.tfidf).fit_transform(texts)
        LogisticRegression(**cfg.logreg).fit(features, [0, 0, 0, 1, 1, 1])


def test_settings_root_follows_config_location(tmp_path, monkeypatch) -> None:
    """Config wskazany przez BANKING77_CONFIG: ścieżki względne liczone od JEGO korzenia.

    Scenariusz obrazu Docker: moduł w site-packages, config w /app/configs/config.yaml.
    """
    from banking77_intent_ml.config.settings import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "config.yaml"
    cfg.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    s = load_settings()
    root = tmp_path.resolve()
    assert s.paths.raw_dir == root / "dane" / "banking77"
    assert s.dataset.label_map_path == root / "configs" / "label_map.json"
    assert s.split.manifest_path == root / "configs" / "split_manifest.json"

    # jawny argument wygrywa ze zmienną środowiskową
    assert load_settings(DEFAULT_CONFIG_PATH).paths.raw_dir != s.paths.raw_dir
