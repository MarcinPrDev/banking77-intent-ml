# Basal ultra na wynajętym GPU — badanie 2 dodatku (31.10.2026)

Prerejestracja: sekcja `basal_ultra` w `configs/config.yaml`. Model `Remek/basal-omni-1.0-ultra`
(31B, Gemma 4 31B, wagi bf16 62,55 GB) nie mieści się na lokalnej karcie RTX 5080 (16 GB), więc
serwer `basal-serve` działa na wynajętej maszynie. Skrypt, dane i MLflow zostają lokalnie; żądania
idą tunelem SSH na lokalny port 18090 (`basal_ultra.url`).

## Wymagania maszyny

- Linux, GPU z co najmniej 80 GB pamięci (A100 80 GB, H100, H200 albo RTX PRO 6000 96 GB).
  Według karty modelu tryb `eager` w bf16 potrzebuje ok. 62 GB na same wagi; na karcie 80 GB
  mieści stany do ok. 10 000 tokenów (prompt v2 ma ok. 3 300).
- Sterownik NVIDIA z obsługą CUDA 12.8 lub nowszej (koła PyTorch `cu128`).
- Dysk co najmniej 150 GB (wagi 62,55 GB + środowisko).
- Dostęp SSH kluczem publicznym.

## 1. Maszyna zdalna

Te same wersje co lokalnie w badaniu 1: Python 3.12, PyTorch 2.11.0 (cu128), `basal` 2.4.1.
Model trafia na duży dysk (`HF_HOME`), bo dysk systemowy poda bywa mniejszy niż 62,55 GB wag.

```bash
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"; }
export HF_HOME=/workspace/hf
uv venv --python 3.12 /root/basal-env && source /root/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.4.1.tar.gz"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"

REV=6d5ba67db149e47ddce26fc02d78c0cadb10f496
SNAP=$(hf download Remek/basal-omni-1.0-ultra --revision "$REV" | tail -1)
(cd "$SNAP" && sha256sum -c SHA256SUMS)   # sumy plików z repozytorium modelu

HF_HUB_OFFLINE=1 nohup basal-serve --model "$SNAP" --name basal-omni-1.0-ultra@6d5ba67 \
  --mode eager --template-prefix off --deterministic --host 127.0.0.1 --port 18090 \
  > /workspace/basal-serve.log 2>&1 &
curl -s http://127.0.0.1:18090/health
```

Przebieg 31.10.2026: RunPod Secure Cloud, 1× H100 SXM 80 GB (EUR-NO-2, $3,99/h), szablon
`runpod-torch-v280` (Ubuntu 24.04, Python 3.12.3, `uv` w obrazie), dyski 50 GB + 150 GB
(`/workspace`), tylko port `22/tcp`; klucz SSH przekazany zmienną `PUBLIC_KEY` poda.
Odcisk klucza hosta porównany z logiem startu poda przed pierwszym połączeniem.

Serwer nasłuchuje tylko na `127.0.0.1` maszyny — z zewnątrz dostęp jest wyłącznie przez tunel.

## 2. Lokalnie (korzeń repo)

Przed pierwszym runem: `basal_ultra.hardware` w configu (nazwa GPU z kroku 1), commit.

```bash
ssh -N -L 18090:127.0.0.1:18090 -p <port> -i ~/.ssh/<klucz> <użytkownik>@<host>
```

Tunel w osobnym terminalu przez cały pomiar. Potem:

```bash
uv run python scripts/basal_zero_shot.py --stage val --study basal_ultra --candidate basal_omni_ultra --labels v1
uv run python scripts/basal_zero_shot.py --stage val --study basal_ultra --candidate basal_omni_ultra --labels v2
uv run python scripts/basal_zero_shot.py --stage rank --study basal_ultra
# wpis autora basal_ultra.frozen_model / frozen_labels + commit, potem raz:
uv run python scripts/basal_zero_shot.py --stage test --study basal_ultra
```

Skrypt odmawia powtórzenia walidacji kandydata, testu przed wpisem `frozen_*` zgodnym z regułą
i drugiego testu w tym badaniu. Zerwane połączenie: do 3 ponowień żądania (`basal_ultra.retries`);
błąd HTTP przerywa run bez zapisu w MLflow.

## 3. Interpretacja i sprzątanie

- Czas klienta (`latency_wall_*`) obejmuje tunel do centrum danych — porównywalny jest tylko
  czas po stronie serwera (`latency_server_*`), i to na innym GPU niż RTX 5080.
- Po teście: zatrzymać i usunąć maszynę (rozliczenie za godziny) oraz klucz SSH u dostawcy.
