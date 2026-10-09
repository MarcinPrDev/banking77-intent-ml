"""Smoke test środowiska GPU pod fine-tuning DistilBERT (RTX 5080 / Blackwell sm_120) — W1 N3.

Weryfikuje WYŁĄCZNIE środowisko — bez metryk jakości, bez walidacji i testu:
1. torch widzi kartę i ma wkompilowane sm_120 (koła cu128+),
2. tensor przechodzi przez GPU (matmul),
3. DistilBERT z configu (`models.distilbert.model_id` + przypięta `revision`, `num_labels` =
   `dataset.n_classes`): jedna mini-epoka na SMOKE_SENTENCES zdaniach z części treningowej,
   batch i `max_len` z configu, bf16 (autocast), dynamiczny padding — mieści się w VRAM.
   Raport: szczyt VRAM, czas kroku (pierwszy osobno — rozgrzewka), najdłuższy batch w tokenach.

Środowisko:    uv sync --extra dev --extra nlp --extra gpu --extra serving
               (pełna lista extras — `uv sync` usuwa extras pominięte w komendzie)
Uruchomienie:  uv run python scripts/smoke_gpu.py
Oczekiwany finał: "SMOKE TEST GPU: OK". Każdy inny wynik = ścieżka DistilBERT zagrożona
(osobny timebox GPU w playbooku; fallback: TF-IDF + LinearSVC).
"""

import os
import statistics
import sys
import time

SYNC_HINT = "uv sync --extra dev --extra nlp --extra gpu --extra serving"
SMOKE_SENTENCES = 300  # „kilkaset zdań” z planu W1 (N3): ok. 10 kroków przy batch 32


def fail(msg: str) -> None:
    print(f"\n[FAIL] SMOKE TEST GPU: {msg}")
    sys.exit(1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # konsola Windows (cp1250) a polskie znaki
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # jawnie; DataLoader bez wątków

    try:
        import torch
    except ImportError:
        fail(f"brak torch. Zainstaluj: {SYNC_HINT}")

    print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        fail(
            "torch nie widzi CUDA. Najczęstszy powód: koło z PyPI zamiast indeksu cu128 "
            "(sprawdź: uv pip show torch → wersja powinna zawierać '+cu128')."
        )

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} ({total_gb:.1f} GB) | compute capability: {cap} | arch list: {archs}")

    if cap >= (12, 0) and not any("120" in a for a in archs):
        fail(
            "karta to Blackwell (sm_120), ale ten build torch nie ma sm_120 w arch list — "
            f"kernele nie ruszą. Wymuś koła cu128+ (index w pyproject.toml): {SYNC_HINT}"
        )

    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul 1024x1024 na GPU: OK (suma={y:.1f})")

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            set_seed,
        )
    except ImportError:
        fail(f"brak transformers. Zainstaluj: {SYNC_HINT}")

    from banking77_intent_ml.config import load_settings
    from banking77_intent_ml.data.datasets import load_train_val

    settings = load_settings()
    cfg = settings.models.distilbert
    if cfg.revision == "PIN-ME":
        fail("rewizja DistilBERT nieprzypięta w configu (models.distilbert.revision)")
    seed = settings.models.seed
    set_seed(seed)

    data = load_train_val(settings)  # wyłącznie część treningowa — bez walidacji i testu
    sample = data.x_train.sample(n=SMOKE_SENTENCES, random_state=seed)
    labels = data.y_train.loc[sample.index].tolist()

    print(f"Ładuję {cfg.model_id} @ {cfg.revision[:8]} (pierwszy raz ~270 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id, revision=cfg.revision, num_labels=settings.dataset.n_classes
    ).cuda()

    encoded = tokenizer(sample.tolist(), truncation=True, max_length=cfg.max_len)
    features = [
        {"input_ids": ids, "attention_mask": mask, "labels": label}
        for ids, mask, label in zip(
            encoded["input_ids"], encoded["attention_mask"], labels, strict=True
        )
    ]
    loader = torch.utils.data.DataLoader(
        features,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),  # padding do najdłuższego w batchu
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,  # Windows
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_seconds, losses, longest = [], [], 0
    try:
        for batch in loader:
            batch = {key: value.cuda() for key, value in batch.items()}
            longest = max(longest, batch["input_ids"].shape[1])
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                fail(f"loss nie jest skończony: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - start)
            losses.append(loss.item())
    except torch.cuda.OutOfMemoryError:
        fail(
            f"brak VRAM przy batch {cfg.batch_size} i max_len {cfg.max_len} (bf16) — "
            "opcje: mniejszy batch z gradient accumulation albo krótszy max_len"
        )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    rest = statistics.median(step_seconds[1:]) if len(step_seconds) > 1 else float("nan")
    print(
        f"DistilBERT mini-epoka: {len(step_seconds)} kroków × batch {cfg.batch_size} "
        f"(bf16, dynamiczny padding; najdłuższy batch {longest} tokenów, max_len {cfg.max_len})"
    )
    print(
        f"  czas kroku: pierwszy {step_seconds[0]:.3f} s (rozgrzewka), mediana pozostałych "
        f"{rest:.3f} s | loss: pierwszy {losses[0]:.3f}, ostatni {losses[-1]:.3f}"
    )
    print(f"  szczyt VRAM: {peak_gb:.2f} GB z {total_gb:.1f} GB")

    print("\n[OK] SMOKE TEST GPU — środowisko gotowe pod fine-tuning DistilBERT lokalnie.")


if __name__ == "__main__":
    main()
