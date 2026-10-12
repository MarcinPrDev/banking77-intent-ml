"""Zamrożone embeddingi all-MiniLM-L6-v2 (W2, S2) — liczone raz z przypiętej rewizji, cache .npy.

Model kończy się modułem Normalize (modules.json przypiętej rewizji), więc wektory mają normę 1
i cosinus to iloczyn skalarny. Limit modelu to 256 tokenów (sentence_bert_config.json), a
najdłuższa wiadomość train ma 98 tokenów WordPiece (EDA, sekcja 4) — nic nie jest przycinane.

Cache: `paths.cache_dir/minilm/{train,val}.npy` + `meta.json` (model, rewizja, wersja
sentence-transformers, hashe list id części train i val, czasy, urządzenie, normy). Odczyt
porównuje te pola z bieżącym configiem i splitem; różnica = liczymy od nowa — cache to dane
pochodne, nie zamrożony artefakt. Wiersze w tej samej kolejności co `data.x_train` / `x_val`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import metadata

import numpy as np

from banking77_intent_ml.config import Settings
from banking77_intent_ml.data.datasets import TrainVal
from banking77_intent_ml.data.splits import ids_sha256

CACHE_SUBDIR = "minilm"
_KEY_FIELDS = (
    "model_id", "revision", "sentence_transformers", "sha256_train_ids", "sha256_val_ids",
)


@dataclass(frozen=True)
class SplitEmbeddings:
    train: np.ndarray  # (n_train, wymiar), kolejność jak data.x_train
    val: np.ndarray  # (n_val, wymiar), kolejność jak data.x_val
    meta: dict  # metadane cache (m.in. encode_train_seconds, device)
    from_cache: bool


def encode_splits(data: TrainVal, settings: Settings) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embeddingi train i val jednym modelem z przypiętej rewizji (GPU, jeśli dostępne)."""
    import torch
    from sentence_transformers import SentenceTransformer

    cfg = settings.models.minilm_knn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    model = SentenceTransformer(cfg.model_id, revision=cfg.revision, device=device)
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    train = model.encode(data.x_train.tolist(), convert_to_numpy=True, show_progress_bar=False)
    encode_train_seconds = time.perf_counter() - start
    val = model.encode(data.x_val.tolist(), convert_to_numpy=True, show_progress_bar=False)
    timing = {
        "device": device,
        "load_seconds": load_seconds,
        "encode_train_seconds": encode_train_seconds,
    }
    return train, val, timing


def _library_version() -> str:
    """Wersja sentence-transformers bez importu biblioteki (CI ma tylko extra `dev`)."""
    try:
        return metadata.version("sentence-transformers")
    except metadata.PackageNotFoundError:  # testy z atrapą kodera w CI
        return "brak"


def _key(data: TrainVal, settings: Settings) -> dict:
    cfg = settings.models.minilm_knn
    return {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "sentence_transformers": _library_version(),
        "sha256_train_ids": ids_sha256(data.x_train.index),
        "sha256_val_ids": ids_sha256(data.x_val.index),
    }


def load_or_compute(data: TrainVal, settings: Settings) -> SplitEmbeddings:
    """Embeddingi z cache, jeśli pasują do configu i splitu; inaczej liczone i zapisywane."""
    cache = settings.paths.cache_dir / CACHE_SUBDIR
    meta_path = cache / "meta.json"
    key = _key(data, settings)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if {field: meta.get(field) for field in _KEY_FIELDS} == key:
            train, val = np.load(cache / "train.npy"), np.load(cache / "val.npy")
            return SplitEmbeddings(train, val, meta, from_cache=True)

    train, val, timing = encode_splits(data, settings)
    norms = np.linalg.norm(train, axis=1)
    meta = {
        **key,
        **timing,
        "dim": int(train.shape[1]),
        "n_train": int(train.shape[0]),
        "n_val": int(val.shape[0]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "train.npy", train)
    np.save(cache / "val.npy", val)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return SplitEmbeddings(train, val, meta, from_cache=False)
