"""Artefakt wdrażanego modelu w GitHub Release (W4, ADR-0003 — opcja a); logika: models/release.py.

Uruchomienie:
    uv run python scripts/model_release.py pack
        # dist/<model_version>.tar z `deploy.artifact_dir` + manifest; drukuje SHA-256 do configu
    uv run python scripts/model_release.py unpack dist/<model_version>.tar [--dest KATALOG]
        # weryfikacja (suma z configu, wersja, mapa etykiet, sumy plików) i rozpakowanie do
        # `deploy.artifact_dir` albo do --dest (np. kontrola pobranego assetu)
    uv run python scripts/model_release.py env >> "$GITHUB_ENV"
        # MODEL_VERSION, RELEASE_TAG, RELEASE_ASSET z configu dla workflow (jedno źródło wersji)

Kody wyjścia: 0 = OK, 1 = niezgodność albo istniejące archiwum / katalog docelowy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.models.release import (
    ReleaseArtifactError,
    build_manifest,
    final_runs,
    pack_artifact,
    unpack_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows: polskie znaki w komunikatach
    parser = argparse.ArgumentParser(description="Artefakt modelu w GitHub Release.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pack", help="spakuj deploy.artifact_dir do dist/<model_version>.tar")
    unpack = sub.add_parser("unpack", help="zweryfikuj i rozpakuj archiwum")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("--dest", type=Path, help="katalog docelowy (domyślnie artifact_dir)")
    sub.add_parser("env", help="wypisz MODEL_VERSION, RELEASE_TAG, RELEASE_ASSET (KLUCZ=wartość)")
    args = parser.parse_args()

    settings = load_settings()
    deploy = settings.deploy
    if args.command == "env":
        print(f"MODEL_VERSION={deploy.model_version}")
        print(f"RELEASE_TAG={deploy.release_tag}")
        print(f"RELEASE_ASSET={deploy.release_asset}")
        return 0
    try:
        if args.command == "pack":
            archive = REPO_ROOT / "dist" / deploy.release_asset
            manifest = build_manifest(deploy.artifact_dir, settings, final_runs(settings))
            sha = pack_artifact(deploy.artifact_dir, archive, manifest)
            print(f"[OK] {archive.relative_to(REPO_ROOT)}: {len(manifest['files'])} plików")
            print(f"  SHA-256: {sha}")
            print(f"  runy MLflow: {manifest['mlflow_runs']}")
            if sha != deploy.release_sha256:
                print("  -> wpisz tę sumę do configs/config.yaml: deploy.release_sha256")
            print(f"  Release: tag {deploy.release_tag}, asset {deploy.release_asset}")
        else:
            manifest = unpack_artifact(args.archive, settings, args.dest)
            target = args.dest or deploy.artifact_dir
            print(
                f"[OK] {args.archive.name}: suma archiwum, wersja {manifest['model_version']}, "
                f"mapa etykiet i {len(manifest['files'])} plików zgodne — rozpakowano do {target}"
            )
    except (ReleaseArtifactError, FileExistsError) as exc:
        print(f"[FAIL] MODEL RELEASE: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
