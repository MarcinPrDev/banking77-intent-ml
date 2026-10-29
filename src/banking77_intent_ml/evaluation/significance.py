"""Test istotności różnic między ścieżkami na zapisanych predykcjach testu (dodatek W5, 29–30.10).

Prerejestracja przed obliczeniami (config `evaluation.significance`, decyzje autora 29.10.2026):
pary z `pairs` na teście 3 080, dokładny McNemar na parach rozbieżnych, korekta Holma na wszystkie
pary, `alpha` po korekcie; parowany bootstrap różnicy accuracy — te same wylosowane wiersze dla
obu modeli, `n_bootstrap` losowań, seed `models.seed`, przedział percentylowy `ci_level`.
Na zapisanych predykcjach (`error_analysis.test_artifacts`) — bez treningu, strojenia i wyboru
modelu; wynik nie zmienia zamrożonych liczb ani decyzji z ADR-0003.

Obliczenia korzystają wyłącznie z id i etykiet zapisanych predykcji.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from banking77_intent_ml.config import Settings

Pair = tuple[str, str]


def correct_by_model(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ramka bool (kolumna na ścieżkę: `y_pred == y_true`), wiersze wyrównane po `row_id`.

    `predictions`: ścieżka -> ramka `row_id, y_true, y_pred, score` (jak `predictions_test.csv`).
    Błąd (`ValueError`), gdy ścieżki mają różne zbiory `row_id` albo różne `y_true`,
    puste dane, brakujące wartości lub powtórzone id. Wynik uporządkowany po `row_id`.
    """
    if not predictions:
        raise ValueError("Brak predykcji do porównania")
    reference = None
    correct = {}
    required = ["row_id", "y_true", "y_pred"]
    for name, frame in predictions.items():
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{name}: wymagane kolumny {required}")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{name}: puste dane lub brakujące wartości")
        if frame.row_id.duplicated().any():
            raise ValueError(f"{name}: powtórzone row_id")
        aligned = frame.set_index("row_id").sort_index()
        if reference is None:
            reference = aligned.y_true
        elif not aligned.index.equals(reference.index):
            raise ValueError(f"{name}: różne zbiory row_id")
        elif not aligned.y_true.eq(reference).all():
            raise ValueError(f"{name}: różne y_true dla tych samych row_id")
        correct[name] = aligned.y_pred.eq(aligned.y_true)
    return pd.DataFrame(correct)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """(b, c, p): b — A poprawnie i B błędnie, c — odwrotnie; dokładny test dwustronny.

    p = `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue`; gdy `b + c == 0`, p = 1.0.
    """
    a = np.asarray(correct_a, dtype=bool)
    other = np.asarray(correct_b, dtype=bool)
    if a.ndim != 1 or a.shape != other.shape:
        raise ValueError("Wymagane dwa wektory poprawności tej samej długości")
    b = int(np.count_nonzero(a & ~other))
    c = int(np.count_nonzero(~a & other))
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def holm(p_values: Sequence[float]) -> list[float]:
    """Skorygowane p (Holm, krokowo w dół, monotoniczne, obcięte do 1), w kolejności wejścia."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Wartości p muszą być skończone i należeć do [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def paired_bootstrap_diffs(
    correct: pd.DataFrame,
    pairs: Sequence[Pair],
    *,
    n_bootstrap: int,
    seed: int,
    chunk: int = 1000,
) -> dict[Pair, np.ndarray]:
    """Różnice accuracy A − B w każdym z `n_bootstrap` losowań wierszy z powtórzeniami.

    W jednym losowaniu te same indeksy dla wszystkich par (`np.random.default_rng(seed)`);
    losowania w paczkach po `chunk`, żeby nie trzymać w pamięci macierzy n_bootstrap × n.
    Nieznana ścieżka w `pairs` powoduje `ValueError` z jej nazwą.
    """
    if correct.empty or n_bootstrap <= 0 or chunk <= 0:
        raise ValueError("Wymagane niepuste dane oraz dodatnie n_bootstrap i chunk")
    unknown = sorted({name for pair in pairs for name in pair} - set(correct.columns))
    if unknown:
        raise ValueError(f"Nieznane ścieżki w pairs (brak predykcji): {', '.join(unknown)}")
    rng = np.random.default_rng(seed)
    # Odejmowanie bool nie jest zdefiniowane; int8 wystarcza dla różnic -1, 0, 1.
    differences = {
        (a, b): correct[a].to_numpy(dtype=np.int8) - correct[b].to_numpy(dtype=np.int8)
        for a, b in pairs
    }
    samples = {pair: np.empty(n_bootstrap, dtype=float) for pair in differences}
    n = len(correct)
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        for pair, difference in differences.items():
            samples[pair][start:stop] = difference[indices].mean(axis=1)
    return samples


def significance_table(predictions: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    """Wiersz na parę z `settings.evaluation.significance.pairs`.

    Kolumny: `a`, `b`, `n`, `acc_a`, `acc_b`, `diff_pp` (A − B w p.p.), `b_only`, `c_only`, `p`,
    `p_holm`, `reject` (p_holm <= alpha), `ci_low_pp`, `ci_high_pp` (przedział `ci_level`).
    Przedziały są osobne dla każdej pary; korekta Holma dotyczy tylko wartości p.
    `n` musi odpowiadać `settings.dataset.test.n_rows`; inaczej `ValueError`.
    """
    cfg = settings.evaluation.significance
    correct = correct_by_model(predictions)
    n = len(correct)
    expected_n = settings.dataset.test.n_rows
    if n != expected_n:
        raise ValueError(f"Predykcje: {n} wierszy, pełny test według configu: {expected_n}")
    samples = paired_bootstrap_diffs(
        correct, cfg.pairs, n_bootstrap=cfg.n_bootstrap, seed=settings.models.seed
    )
    tail = (1 - cfg.ci_level) / 2
    accuracy = correct.mean()
    rows = []
    for a, b in cfg.pairs:
        b_only, c_only, p = mcnemar_exact(correct[a].to_numpy(), correct[b].to_numpy())
        low, high = np.quantile(samples[(a, b)], [tail, 1 - tail])
        rows.append({
            "a": a, "b": b, "n": n,
            "acc_a": accuracy[a], "acc_b": accuracy[b],
            "diff_pp": 100 * (accuracy[a] - accuracy[b]),
            "b_only": b_only, "c_only": c_only, "p": p,
            "ci_low_pp": 100 * low, "ci_high_pp": 100 * high,
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p)
    table["reject"] = table.p_holm <= cfg.alpha
    return table[[
        "a", "b", "n", "acc_a", "acc_b", "diff_pp", "b_only", "c_only", "p", "p_holm",
        "reject", "ci_low_pp", "ci_high_pp",
    ]]
