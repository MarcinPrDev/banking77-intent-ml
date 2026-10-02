"""Smoke test środowiska GPU pod fine-tuning DistilBERT (RTX 5080 / Blackwell sm_120).

Weryfikuje WYŁĄCZNIE środowisko (nie robi żadnego ML na realnych danych):
1. torch widzi kartę i ma wkompilowane sm_120 (koła cu128+),
2. tensor przechodzi przez GPU (matmul),
3. DistilBERT: forward + backward na 8 sztucznych zdaniach mieści się w VRAM.

Środowisko:    uv sync --extra dev --extra nlp --extra gpu --extra serving
               (pełna lista extras — `uv sync` usuwa extras pominięte w komendzie)
Uruchomienie:  uv run python scripts/smoke_gpu.py
Oczekiwany finał: "SMOKE TEST GPU: OK". Każdy inny wynik = ścieżka DistilBERT zagrożona
(osobny timebox GPU w playbooku; fallback: TF-IDF + LinearSVC).
"""

import sys

SYNC_HINT = "uv sync --extra dev --extra nlp --extra gpu --extra serving"


def fail(msg: str) -> None:
    print(f"\n[FAIL] SMOKE TEST GPU: {msg}")
    sys.exit(1)


def main() -> None:
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
    print(f"GPU: {name} | compute capability: {cap} | arch list: {archs}")

    if cap >= (12, 0) and not any("120" in a for a in archs):
        fail(
            "karta to Blackwell (sm_120), ale ten build torch nie ma sm_120 w arch list — "
            f"kernele nie ruszą. Wymuś koła cu128+ (index w pyproject.toml): {SYNC_HINT}"
        )

    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul 1024x1024 na GPU: OK (suma={y:.1f})")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        fail(f"brak transformers. Zainstaluj: {SYNC_HINT}")

    model_id = "distilbert-base-uncased"
    print(f"Pobieram/laduję {model_id} (pierwszy raz ~270 MB)...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=4).cuda()

    sentences = [f"dummy accident description {i}, lane closed due to test" for i in range(8)]
    batch = tok(sentences, return_tensors="pt", padding=True, truncation=True).to("cuda")
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device="cuda")

    model.train()
    out = model(**batch, labels=labels)
    out.loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"DistilBERT fwd+bwd: OK | loss={out.loss.item():.4f} | peak VRAM={peak_gb:.2f} GB")

    print("\n[OK] SMOKE TEST GPU — środowisko gotowe pod fine-tuning DistilBERT lokalnie.")


if __name__ == "__main__":
    main()
