"""Testy artefaktu w GitHub Release (models/release.py) — małe pliki w tmp, bez sieci i MLflow.

Dowodzą: archiwum jest deterministyczne (te same pliki → ta sama suma), rozpakowanie odtwarza
pliki bit w bit, a odmawia przy innej sumie archiwum, pliku niezgodnym z manifestem, dodatkowym
pliku w archiwum, ścieżce wychodzącej poza katalog, istniejącym katalogu docelowym i innej
mapie etykiet. Nie dowodzą, że asset w Release jest tym plikiem — to sprawdza suma w configu.
"""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import replace

import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.models.persistence import label_map_sha256
from banking77_intent_ml.models.release import (
    ReleaseArtifactError,
    build_manifest,
    pack_artifact,
    sha256_file,
    unpack_artifact,
)

RUNS = {"refit": "r" * 32, "test": "t" * 32}


@pytest.fixture
def artifact(tmp_path):
    """(ustawienia z deploy.artifact_dir w tmp, katalog artefaktu) — pliki jak w DistilBERT."""
    settings = load_settings()
    directory = tmp_path / "distilbert"
    (directory / "sub").mkdir(parents=True)
    metadata = {
        "label_map_sha256": label_map_sha256(settings.dataset.label_map_path),
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.13.0",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "config.json").write_text('{"model_type": "distilbert"}', encoding="utf-8")
    (directory / "sub" / "weights.bin").write_bytes(bytes(range(256)) * 64)
    deploy = replace(settings.deploy, artifact_dir=directory)
    return replace(settings, deploy=deploy), directory


def _with_sha(settings, sha):
    return replace(settings, deploy=replace(settings.deploy, release_sha256=sha))


def test_pack_is_deterministic_and_unpack_restores_files(artifact, tmp_path):
    settings, directory = artifact
    manifest = build_manifest(directory, settings, RUNS)
    sha = pack_artifact(directory, tmp_path / "a.tar", manifest)

    assert pack_artifact(directory, tmp_path / "b.tar", manifest) == sha
    target = tmp_path / "restored"
    restored = unpack_artifact(tmp_path / "a.tar", _with_sha(settings, sha), target)
    assert restored == json.loads(json.dumps(manifest))
    assert sorted(restored["files"]) == ["config.json", "metadata.json", "sub/weights.bin"]
    for name in restored["files"]:
        assert (target / name).read_bytes() == (directory / name).read_bytes()
    assert restored["libraries"] == {"torch": "2.11.0+cu128", "transformers": "5.13.0"}


def test_unpack_rejects_other_archive_sha_before_creating_destination(artifact, tmp_path):
    settings, directory = artifact
    pack_artifact(directory, tmp_path / "a.tar", build_manifest(directory, settings, RUNS))

    with pytest.raises(ReleaseArtifactError, match="SHA-256 archiwum"):
        unpack_artifact(tmp_path / "a.tar", _with_sha(settings, "0" * 64), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unpack_rejects_file_not_matching_manifest_and_cleans_up(artifact, tmp_path):
    settings, directory = artifact
    manifest = build_manifest(directory, settings, RUNS)
    manifest["files"]["sub/weights.bin"]["sha256"] = "f" * 64
    sha = pack_artifact(directory, tmp_path / "a.tar", manifest)

    with pytest.raises(ReleaseArtifactError, match="sub/weights.bin"):
        unpack_artifact(tmp_path / "a.tar", _with_sha(settings, sha), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unpack_rejects_member_outside_manifest(artifact, tmp_path):
    settings, directory = artifact
    archive = tmp_path / "a.tar"
    pack_artifact(directory, archive, build_manifest(directory, settings, RUNS))
    extra = tmp_path / "extra.txt"
    extra.write_text("spoza manifestu", encoding="utf-8")
    with tarfile.open(archive, "a") as tar:
        tar.add(extra, arcname="distilbert/extra.txt")
    sha = sha256_file(archive)

    with pytest.raises(ReleaseArtifactError, match="zawartość archiwum"):
        unpack_artifact(archive, _with_sha(settings, sha), tmp_path / "out")


def test_unpack_rejects_path_leaving_destination(artifact, tmp_path):
    settings, directory = artifact
    (tmp_path / "evil").write_text("poza katalogiem", encoding="utf-8")
    manifest = build_manifest(directory, settings, RUNS)
    manifest["files"] = {"../evil": {"sha256": "0" * 64, "bytes": 15}}
    sha = pack_artifact(directory, tmp_path / "a.tar", manifest)

    with pytest.raises(ReleaseArtifactError, match="niedozwolona ścieżka"):
        unpack_artifact(tmp_path / "a.tar", _with_sha(settings, sha), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unpack_refuses_existing_destination(artifact, tmp_path):
    settings, directory = artifact
    sha = pack_artifact(directory, tmp_path / "a.tar", build_manifest(directory, settings, RUNS))

    with pytest.raises(FileExistsError):
        unpack_artifact(tmp_path / "a.tar", _with_sha(settings, sha), directory)


def test_build_manifest_refuses_other_label_map(artifact):
    settings, directory = artifact
    (directory / "metadata.json").write_text('{"label_map_sha256": "inna"}', encoding="utf-8")

    with pytest.raises(ReleaseArtifactError, match="mapa etykiet"):
        build_manifest(directory, settings, RUNS)


def test_release_names_come_from_model_version_and_sha_is_pinned():
    deploy = load_settings().deploy

    assert deploy.release_tag == deploy.model_version
    assert deploy.release_asset == f"{deploy.model_version}.tar"
    assert re.fullmatch(r"[0-9a-f]{64}", deploy.release_sha256)
