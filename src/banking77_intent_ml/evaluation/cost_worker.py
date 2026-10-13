"""Proces pomiarowy kosztu CPU (W2 N2) — uruchamia go `evaluation.cost` jako nowy proces Pythona.

Wejście: jeden argument JSON — `mode` (`cold` albo `warm`), `model`, `threads`, `texts` (ścieżka
do listy zdań), opcjonalnie `directory` artefaktu; dla `warm` także `warmup` i `passes`.
Wyjście: jedna linia JSON na stdout.

Na górze tylko biblioteka standardowa: zimny start liczony od importu pakietu obejmuje import
kodu modelu, `load` i pierwszą predykcję; start interpretera i odczyt listy zdań są poza
pomiarem (stałe dla każdej ścieżki). Wątki ustawia środowisko procesu (OMP/MKL/OpenBLAS —
torch czyta je przy imporcie); po `load` worker woła jeszcze `torch.set_num_threads`
i zwraca faktyczną liczbę wątków torch oraz to, czy CUDA była widoczna.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _torch_state(threads: int) -> tuple[int | None, bool]:
    """(wątki torch, CUDA widoczna); ścieżka bez torch (TF-IDF) -> (None, False)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None, False
    torch.set_num_threads(threads)
    return torch.get_num_threads(), torch.cuda.is_available()


def main() -> None:
    job = json.loads(sys.argv[1])
    texts = json.loads(Path(job["texts"]).read_text(encoding="utf-8"))
    directory = Path(job["directory"]) if job.get("directory") else None

    start = time.perf_counter()
    from banking77_intent_ml.config import load_settings
    from banking77_intent_ml.models.factory import load_predictor

    imported = time.perf_counter()
    predictor = load_predictor(job["model"], load_settings(), directory)
    loaded = time.perf_counter()
    torch_threads, cuda = _torch_state(job["threads"])

    if job["mode"] == "cold":
        begin = time.perf_counter()
        predictor.predict_proba(texts[:1])
        result = {
            "import_s": imported - start,
            "load_s": loaded - imported,
            "first_predict_s": time.perf_counter() - begin,
        }
    else:
        for index in range(job["warmup"]):
            predictor.predict_proba([texts[index % len(texts)]])
        timings = []
        for _ in range(job["passes"]):
            for text in texts:
                begin = time.perf_counter_ns()
                predictor.predict_proba([text])
                timings.append(time.perf_counter_ns() - begin)
        result = {"timings_ns": timings}
    print(json.dumps({**result, "torch_threads": torch_threads, "cuda": cuda}))


if __name__ == "__main__":
    main()
