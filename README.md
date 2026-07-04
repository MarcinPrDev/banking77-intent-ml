# Accident Severity Intelligence (us-accidents-ml)

Projekt końcowy kursu **AI & ML Engineering** (SAGES/Kodołamacz, gr. 06.12.2025).
Przewidywanie powagi wypadku drogowego (`Severity` 1–4) na danych
[US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — **trzema drogami, które porównuję**:

1. **Tabular** — cechy pogodowe/czasowe/geo → LogisticRegression → RandomForest → **XGBoost + Optuna**, tracking w **MLflow**
2. **Tekst** — kolumna `Description` → TF-IDF → embeddingi zdaniowe → fine-tuning DistilBERT (lokalnie, RTX 5080)
3. **LLM** — klasyfikacja zero-shot opisu (Groq) na próbce, jako baseline porównawczy + endpoint `/explain`

**Pytanie badawcze:** ile sygnału o powadze wypadku niesie tekstowy opis względem danych
strukturalnych i czy fuzja cech poprawia wynik?

Warstwa inżynierska: pakiet `src/` (OOP, pipeline'y, pytest) → **FastAPI** → **Docker** →
**Kubernetes/GCP** → **CI/CD dev→prod**. Przygotowanie danych: siostrzany projekt
[us-accidents-etl](https://github.com/MarcinPrDev/us-accidents-etl) (Spark).

## Quickstart

```bash
uv sync --extra dev          # środowisko + zależności
# dane (~3 GB po rozpakowaniu; nie trzymamy ich w repo):
#   https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents  -> dane/US_Accidents_March23.csv
uv run jupyter lab           # notebooks/01_eda.ipynb
uv run pytest                # testy
uv run ruff check .          # lint
```

Ścieżki i parametry: `configs/config.yaml`.

## Plan etapów (≈ tydzień na etap, powtórka materiału zjazdu w tle)

| Etap | Zakres | Powtórka |
|---|---|---|
| W1 | EDA: braki (i powody), imbalance, outliery, cechy czasowo-geo (DuckDB na pełnym zbiorze) | Z2–Z3 |
| W2–3 | Feature engineering, baseline → RF → XGBoost, Optuna, MLflow | Z3, Z7 |
| W4 | Ścieżka tekstowa: TF-IDF / embeddingi / DistilBERT (GPU lokalnie) | Z8 |
| W5 | LLM-y przez LiteLLM (Groq / Gemini / lokalna Ollama) + Langfuse (obserwowalność, evale, koszty) + fuzja cech + wnioski | Z9 |
| W6 | Refactor do pakietu: pipeline'y, konfiguracja, pytest, DVC | Z5–Z6 |
| W7 | FastAPI (`/predict`, `/predict-text`, `/explain`) + Docker | Z11–Z12 |
| W8 | Deploy: Cloud Run (dev) → GKE Autopilot (prod), CI/CD dev→prod, Secret Manager, WIF | Z13–Z14 |
| W9 | README, diagram, wyniki, feedback (M. Rybiński) | — |

## Chmura (GCP)

- **BigQuery** — wynik ETL + analizy SQL na pełnym zbiorze (W1–W2, free tier)
- **GCS** — remote DVC + artefakty modeli (W6)
- **Dataproc Serverless** — produkcyjny run ETL Spark (W6/W8; workflow w [us-accidents-etl](https://github.com/MarcinPrDev/us-accidents-etl))
- **Artifact Registry · Cloud Run (dev) · GKE Autopilot (prod) · Secret Manager · Workload Identity Federation** (W7–W8)

## Zgodność z krajobrazem 2026 (LLMOps)

- **LiteLLM** — jedna brama do Groq / Gemini / lokalnej Ollamy (routing, koszty)
- **Langfuse** — obserwowalność LLM, wersjonowanie promptów, **evale liczone twardo vs etykiety `Severity`**
- metryka: koszt i latencja / 1000 predykcji per dostawca; structured outputs (Pydantic) w API

## Roadmap (świadomie poza zakresem)

Vertex AI Pipelines/Endpoints · Pub/Sub (streaming) · Cloud Composer/Airflow · agenty i MCP · monitoring dryfu danych

## Status

- [x] Szkielet projektu (pakiet, config, CI, notebook EDA)
- [ ] W1: dane pobrane + EDA
- [ ] W2–3: modele tabular + Optuna + MLflow
- [ ] W4: ścieżka tekstowa
- [ ] W5: LLM + fuzja + wnioski
- [ ] W6: refactor + testy + DVC
- [ ] W7: API + Docker
- [ ] W8: deploy + CI/CD dev→prod
- [ ] W9: dokumentacja + feedback
