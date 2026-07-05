"""Kontrakt predykcji: cechy post-hoc (znane dopiero PO zdarzeniu) nie wchodzą do modelu.

Szkielet-strażnik (infra): skanuje src/ pod kątem literalnych odwołań do cech
z `configs/config.yaml -> forbidden_features`. Świadome odstępstwo (np. kontrolny
run leakage w W3) oznacza się komentarzem `# allow-posthoc` w tej samej linii.

Do rozszerzenia przez autora w W2+: asercje na realnej liście kolumn wejściowych
pipeline'ów (nie tylko grep po źródłach).
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _forbidden_features() -> list[str]:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return cfg["forbidden_features"]


def test_forbidden_features_defined_in_config():
    feats = set(_forbidden_features())
    assert {"Distance(mi)", "End_Time", "End_Lat", "End_Lng"} <= feats


def test_src_does_not_reference_forbidden_features():
    feats = _forbidden_features()
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if "allow-posthoc" in line:
                continue
            for feat in feats:
                if re.search(re.escape(feat), line):
                    offenders.append(f"{py.relative_to(ROOT)}:{lineno}: {feat}")
    assert not offenders, "Cechy post-hoc w kodzie modelowym:\n" + "\n".join(offenders)
