"""LLM zero-shot (stretch 2, W4): dobór promptu na walidacji, potem raz test 385 — cienkie CLI.

Środowisko z extra `llm`:  uv sync --extra dev --extra nlp --extra gpu --extra serving --extra llm
Uruchomienie (z korzenia repo; klucz GROQ_API_KEY w .env, poza gitem):
  uv run python scripts/llm_zero_shot.py --stage val --prompt v1
  uv run python scripts/llm_zero_shot.py --stage val --prompt v2
  uv run python scripts/llm_zero_shot.py --stage test        # po wpisie autora llm.frozen_prompt
Budżet (decyzje autora 18.10.2026): każdy wariant raz na walidacji, test raz. Skrypt odmawia przed
zapytaniami do API: pusty szablon, istniejący run wariantu (val), brak `llm.frozen_prompt`, wariant
bez runu walidacyjnego albo istniejący run testowy LLM. Odpowiedzi w cache `paths.cache_dir/llm`
— przerwany przebieg powtarza się bez ponownego kosztu. Teksty testu nie są wypisywane.
Logika: src/banking77_intent_ml/models/llm.py, src/banking77_intent_ml/evaluation/llm_eval.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.data.datasets import load_train_val
from banking77_intent_ml.evaluation.llm_eval import (
    count_llm_test_runs,
    count_val_runs,
    prompt_template,
    run_test,
    run_val,
)
from banking77_intent_ml.models.llm import CachedClient, LiteLLMClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(message: str) -> None:
    print(f"[FAIL] LLM: {message}")
    sys.exit(1)


def _client(settings):
    try:
        from dotenv import load_dotenv  # zależność litellm
    except ImportError:
        pass
    else:
        load_dotenv(REPO_ROOT / ".env")
    safe_model = settings.llm.model.replace("/", "_")
    cache = settings.paths.cache_dir / "llm" / f"{safe_model}.jsonl"
    return CachedClient(LiteLLMClient(settings), cache, settings)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    parser = argparse.ArgumentParser(description="LLM zero-shot: walidacja promptu albo test 385.")
    parser.add_argument("--stage", choices=("val", "test"), required=True)
    parser.add_argument("--prompt", help="wariant z llm.prompts (tylko --stage val)")
    args = parser.parse_args()
    settings = load_settings()
    cfg = settings.llm

    if args.stage == "val":
        if not args.prompt:
            _fail("--stage val wymaga --prompt (np. v1)")
        try:
            prompt_template(settings, args.prompt)
        except ValueError as exc:
            _fail(str(exc))
        done = count_val_runs(settings, args.prompt)
        if done:
            _fail(f"wariant {args.prompt} ma już {done} run(y) na walidacji — budżet: raz")
        data = load_train_val(settings)
        client = _client(settings)
        run_id, result, usage = run_val(args.prompt, data, client, settings)
        scope = f"walidacja ({int(result.n_samples)} zdań)"
    else:
        name = cfg.frozen_prompt
        if not name:
            _fail("brak llm.frozen_prompt — najpierw dobór promptu na walidacji (wpis autora)")
        try:
            prompt_template(settings, name)
        except ValueError as exc:
            _fail(str(exc))
        if not count_val_runs(settings, name):
            _fail(f"wariant {name} nie ma runu walidacyjnego — prompt wybieramy na walidacji")
        done = count_llm_test_runs(settings)
        if done:
            _fail(f"w MLflow jest już {done} run testowy LLM — ocena na teście raz")
        client = _client(settings)
        run_id, result, usage = run_test(client, settings)
        scope = f"test 385 ({int(result.n_samples)} wiadomości), prompt {name}"

    print(f"[OK] LLM {cfg.model}, {scope}: accuracy {result.accuracy:.4f}, "
          f"macro-F1 {result.macro_f1:.4f}, błędy formatu {usage['format_errors']:.0f}")
    print(f"  tokeny {usage['prompt_tokens']:.0f} / {usage['completion_tokens']:.0f}, "
          f"latencja p50 {usage['latency_p50_s']:.2f} s / p95 {usage['latency_p95_s']:.2f} s, "
          f"koszt {usage['cost_usd']:.4f} USD | zapytań do API {client.calls} | "
          f"run {run_id[:8]}")


if __name__ == "__main__":
    main()
