"""Basal zero-shot (dodatek W5): kandydaci na walidacji, ranking, potem raz test — cienkie CLI.

Serwer Basal działa w osobnym środowisku (P:\\Repos\\basal-runtime, transformers 5.17.0), offline
(HF_HUB_OFFLINE=1), z katalogu przypiętej rewizji w cache HF i flagami z `basal.serve_args`:
  basal-serve --model <cache HF>/snapshots/<rewizja> --name <served_name> --mode eager
              --template-prefix off --deterministic --host 127.0.0.1 --port 18090
`--name` = `models.basal.served_name(model)` (repo@7 znaków rewizji) — klient odmawia odpowiedzi
od innego modelu. Uruchomienie (z korzenia repo, środowisko projektu):
  uv run python scripts/basal_zero_shot.py --stage val --candidate basal_4_5b --labels v1
  uv run python scripts/basal_zero_shot.py --stage rank
  uv run python scripts/basal_zero_shot.py --stage test   # po wpisie autora basal.frozen_*
Budżet (prerejestracja 30.10.2026): każdy kandydat raz na walidacji, test raz i tylko zwycięzca
reguły z prerejestracji. Skrypt odmawia przed zapytaniami: puste instrukcje lub opisy, istniejący
run kandydata (val), brak któregoś runu walidacyjnego, `frozen_*` inne niż zwycięzca reguły,
istniejący run testowy Basal. Teksty testu nie są wypisywane. Kod CLI: asystent.
Logika: src/banking77_intent_ml/models/basal.py, src/banking77_intent_ml/evaluation/basal_eval.py.
"""

from __future__ import annotations

import argparse
import sys

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.evaluation.basal_eval import (
    choose,
    count_basal_test_runs,
    count_val_runs,
    instructions,
    label_criteria,
    run_test,
    run_val,
    val_candidates,
)
from banking77_intent_ml.models.basal import HttpBasalClient, served_name


def _fail(message: str) -> None:
    print(f"[FAIL] Basal: {message}")
    sys.exit(1)


def _client(settings, candidate: str) -> HttpBasalClient:
    cfg = settings.basal
    client = HttpBasalClient(cfg.url, cfg.timeout_s, served_name(cfg.models[candidate]))
    try:
        client.health()
    except OSError as exc:
        _fail(f"serwer {cfg.url} nie odpowiada ({exc}) — najpierw basal-serve z --name "
              f"{served_name(cfg.models[candidate])}")
    return client


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="Basal zero-shot: walidacja, ranking, test.")
    parser.add_argument("--stage", choices=("val", "rank", "test"), required=True)
    parser.add_argument("--candidate", help="model z basal.models (tylko --stage val)")
    parser.add_argument("--labels", help="wariant z basal.label_variants (tylko --stage val)")
    args = parser.parse_args()
    settings = load_settings()
    cfg = settings.basal
    if cfg is None:
        _fail("brak sekcji basal w configu")

    if args.stage == "val":
        if args.candidate not in cfg.models or args.labels not in cfg.label_variants:
            _fail(f"--candidate z {sorted(cfg.models)} i --labels z {sorted(cfg.label_variants)}")
        try:
            instructions(settings)
            label_criteria(settings, args.labels)
        except ValueError as exc:
            _fail(str(exc))
        done = count_val_runs(settings, args.candidate, args.labels)
        if done:
            _fail(f"kandydat {args.candidate}/{args.labels} ma już {done} run — budżet: raz")
        client = _client(settings, args.candidate)
        data = load_train_val(settings)
        run_id, result, latency = run_val(args.candidate, args.labels, data, client, settings)
        print(f"[OK] walidacja {args.candidate}/{args.labels} ({int(result.n_samples)} zdań): "
              f"accuracy {result.accuracy:.4f}, macro-F1 {result.macro_f1:.4f}, "
              f"p50 {latency['latency_wall_p50_ms']:.1f} ms — run {run_id}")
        return

    try:
        candidates = val_candidates(settings)
    except ValueError as exc:
        _fail(str(exc))
    winner = choose(candidates, cfg.tie_tolerance)
    if args.stage == "rank":
        for c in candidates:
            mark = "  <- zwycięzca reguły" if c == winner else ""
            print(f"{c.candidate:>14} {c.labels}: macro-F1 {c.macro_f1:.4f}, accuracy "
                  f"{c.accuracy:.4f}, cost_rank {c.cost_rank}, run {c.run_id[:8]}{mark}")
        print(f"Wpis do configu (decyzja autora): basal.frozen_model: {winner.candidate}, "
              f"basal.frozen_labels: {winner.labels}")
        return

    if (cfg.frozen_model, cfg.frozen_labels) != (winner.candidate, winner.labels):
        _fail(f"frozen ({cfg.frozen_model}/{cfg.frozen_labels}) różne od zwycięzcy reguły "
              f"({winner.candidate}/{winner.labels})")
    done = count_basal_test_runs(settings)
    if done:
        _fail(f"w MLflow jest już {done} run testowy Basal — ocena na teście raz")
    client = _client(settings, winner.candidate)
    run_id, results, latency = run_test(client, settings)
    for prefix, result in results.items():
        print(f"[OK] test {prefix} ({int(result.n_samples)}): accuracy {result.accuracy:.4f}, "
              f"macro-F1 {result.macro_f1:.4f}")
    print(f"p50/p95 {latency['latency_wall_p50_ms']:.1f}/{latency['latency_wall_p95_ms']:.1f} ms"
          f" — run {run_id}")


if __name__ == "__main__":
    main()
