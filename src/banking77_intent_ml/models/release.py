"""Artefakt wdrażanego modelu jako asset GitHub Release (W4, ADR-0003 — opcja a).

Archiwum `<model_version>.tar` (`deploy.release_asset`) zawiera niezmienione pliki ocenionego
artefaktu (`deploy.artifact_dir`, pod nazwą katalogu) i `manifest.json`: SHA-256 i rozmiar każdego
pliku, hash mapy etykiet, wersje bibliotek z `metadata.json` artefaktu i id runów MLflow (refit
na 10 003 i ocena na teście). Tar bez kompresji, kolejność plików i metadane ustalone (mtime 0,
uid/gid 0) — te same pliki dają to samo SHA-256 archiwum, więc suma w configu
(`deploy.release_sha256`) jest odtwarzalna z ocenionego artefaktu.

W CI: `gh release download` → `unpack_artifact` (suma archiwum z configu, wersja, mapa etykiet,
potem suma każdego pliku z manifestu) → `docker build` kopiuje katalog do obrazu bez zmian.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from banking77_intent_ml.config import Settings
from banking77_intent_ml.models.persistence import label_map_sha256

MANIFEST_NAME = "manifest.json"
_CHUNK = 1 << 20  # 1 MiB
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LIBRARY_KEYS = ("torch_version", "transformers_version", "numpy_version", "sklearn_version")


class ReleaseArtifactError(RuntimeError):
    """Archiwum albo rozpakowany artefakt niezgodny z configiem lub manifestem."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def final_runs(settings: Settings) -> dict[str, str]:
    """Id runów MLflow wdrażanego modelu: refit na 10 003 i ocena na teście (po jednym)."""
    import mlflow  # tylko przy pakowaniu; rozpakowanie w CI działa bez MLflow

    from banking77_intent_ml.evaluation.tracking import configure_mlflow

    configure_mlflow(settings)
    name = settings.api.model
    runs = {}
    for key, query in (
        ("refit", f"tags.final_refit = '{name}'"),
        ("test", f"tags.split = 'test' and tags.model_family = '{name}'"),
    ):
        found = mlflow.search_runs(
            search_all_experiments=True, filter_string=query, output_format="list"
        )
        if len(found) != 1:
            raise ReleaseArtifactError(f"oczekiwano 1 runu ({query}), jest {len(found)}")
        runs[key] = found[0].info.run_id
    return runs


def build_manifest(
    directory: Path, settings: Settings, mlflow_runs: dict[str, str]
) -> dict[str, Any]:
    """Manifest artefaktu; odmawia, gdy mapa etykiet w configu różni się od mapy artefaktu."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    label_hash = label_map_sha256(settings.dataset.label_map_path)
    if metadata.get("label_map_sha256") != label_hash:
        raise ReleaseArtifactError("mapa etykiet w configu różni się od mapy artefaktu")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return {
        "model_version": settings.deploy.model_version,
        "directory": directory.name,
        "files": {
            p.relative_to(directory).as_posix(): {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in files
        },
        "label_map_sha256": label_hash,
        "libraries": {
            key.removesuffix("_version"): metadata[key] for key in _LIBRARY_KEYS if key in metadata
        },
        "mlflow_runs": mlflow_runs,
    }


def _member(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0  # uid/gid 0 i puste nazwy właściciela to wartości domyślne TarInfo
    return info


def pack_artifact(directory: Path, archive: Path, manifest: dict[str, Any]) -> str:
    """Zapisuje archiwum (manifest + pliki z manifestu pod `<katalog>/`) i zwraca jego SHA-256."""
    if archive.exists():
        raise FileExistsError(f"archiwum już istnieje: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    try:
        with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
            tar.addfile(_member(MANIFEST_NAME, len(payload)), io.BytesIO(payload))
            for name in sorted(manifest["files"]):
                path = directory / name
                with path.open("rb") as fh:
                    tar.addfile(_member(f"{manifest['directory']}/{name}", path.stat().st_size), fh)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return sha256_file(archive)


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ReleaseArtifactError(f"niedozwolona ścieżka w manifeście: {name!r}")
    return name


def unpack_artifact(
    archive: Path, settings: Settings, destination: Path | None = None
) -> dict[str, Any]:
    """Weryfikuje archiwum i rozpakowuje artefakt do `destination` (domyślnie artifact_dir).

    Kolejno: SHA-256 archiwum = `deploy.release_sha256`; wersja i mapa etykiet z manifestu =
    config; zawartość archiwum = manifest + pliki z manifestu (nic więcej); po rozpakowaniu
    SHA-256 każdego pliku = manifest. Katalog docelowy nie może istnieć; przy błędzie jest usuwany.
    """
    expected = settings.deploy.release_sha256
    if not _SHA256.fullmatch(expected):
        raise ReleaseArtifactError(f"deploy.release_sha256 to nie suma SHA-256: {expected!r}")
    got = sha256_file(archive)
    if got != expected:
        raise ReleaseArtifactError(
            f"SHA-256 archiwum {archive.name}: {got}, a w configu "
            f"(deploy.release_sha256): {expected}"
        )
    target = destination or settings.deploy.artifact_dir
    if target.exists():
        raise FileExistsError(f"katalog docelowy już istnieje: {target}")
    with tarfile.open(archive, "r:") as tar:
        manifest = json.load(tar.extractfile(MANIFEST_NAME))
        if manifest["model_version"] != settings.deploy.model_version:
            raise ReleaseArtifactError(
                f"archiwum ma wersję {manifest['model_version']}, "
                f"config: {settings.deploy.model_version}"
            )
        if manifest["label_map_sha256"] != label_map_sha256(settings.dataset.label_map_path):
            raise ReleaseArtifactError("mapa etykiet w configu różni się od mapy artefaktu")
        prefix = manifest["directory"]
        expected_names = {MANIFEST_NAME} | {f"{prefix}/{name}" for name in manifest["files"]}
        if set(tar.getnames()) != expected_names:
            raise ReleaseArtifactError("zawartość archiwum różni się od listy plików w manifeście")
        target.mkdir(parents=True)
        try:
            for name, entry in manifest["files"].items():
                member = tar.getmember(f"{prefix}/{_safe_name(name)}")
                if not member.isfile():
                    raise ReleaseArtifactError(f"{member.name} nie jest zwykłym plikiem")
                out = target / name
                out.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with tar.extractfile(member) as src, out.open("wb") as dst:
                    for chunk in iter(lambda: src.read(_CHUNK), b""):
                        h.update(chunk)
                        dst.write(chunk)
                if h.hexdigest() != entry["sha256"]:
                    raise ReleaseArtifactError(f"SHA-256 pliku {name} różni się od manifestu")
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
    return manifest
