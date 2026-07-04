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
| W1 | EDA: braki (i powody), imbalance, outliery, cechy czasowo-geo | Z2–Z3 |
| W2–3 | Feature engineering, baseline → RF → XGBoost, Optuna, MLflow | Z3, Z7 |
| W4 | Ścieżka tekstowa: TF-IDF / embeddingi / DistilBERT (GPU lokalnie) | Z8 |
| W5 | LLM zero-shot (Groq) + fuzja cech + wnioski z porównania | Z9 |
| W6 | Refactor do pakietu: pipeline'y, konfiguracja, pytest, DVC | Z5–Z6 |
| W7 | FastAPI (`/predict`, `/predict-text`, `/explain`) + Docker | Z11–Z12 |
| W8 | Deploy: GKE/Cloud Run + CI/CD dev→prod | Z13–Z14 |
| W9 | README, diagram, wyniki, feedback (M. Rybiński) | — |

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
