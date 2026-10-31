"""Demo BANKING77 na Hugging Face Spaces — model distilbert-v1 w procesie, bez API.

Lokalnie (z korzenia repo, artefakt z `deploy.artifact_dir`):
    MODEL_DIR=models/final/distilbert uv run streamlit run hf_space/app.py
"""

from __future__ import annotations

import streamlit as st
from intent_model import MAX_TEXT_CHARS, IntentModel

MODEL_VERSION = "distilbert-v1"
REPO_URL = "https://github.com/MarcinPrDev/banking77-intent-ml"
DATA_URL = "https://github.com/PolyAI-LDN/task-specific-datasets"
# Zdania kontrolne z configs/smoke_cases.json — napisane do testu dymnego, spoza zbioru BANKING77.
EXAMPLES = (
    "My new card still hasn't arrived, when will it come?",
    "What exchange rate will I get if I pay in euros?",
    "Someone stole my card, please block it right away.",
)


@st.cache_resource
def load_model() -> IntentModel:
    return IntentModel()


def use_example(text: str) -> None:
    st.session_state["text"] = text


st.set_page_config(page_title="BANKING77 — intencje")
st.title("BANKING77 — klasyfikacja intencji")
st.caption(
    "Wiadomość klienta banku (po angielsku) → jedna z 77 intencji. "
    f"Model `{MODEL_VERSION}` (DistilBERT) z projektu końcowego kursu AI & ML Engineering. "
    "Score to wynik modelu dla zwróconej klasy."
)

st.write("Przykłady:")
for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
    column.button(example, on_click=use_example, args=(example,), use_container_width=True)

text = st.text_area(
    "Wiadomość klienta", key="text", max_chars=MAX_TEXT_CHARS,
    placeholder="I still have not received my new card",
)
if st.button("Klasyfikuj", type="primary"):
    if not text.strip():
        st.error("Wpisz wiadomość — pusty tekst albo same spacje nie są klasyfikowane.")
    else:
        best, *others = load_model().top(text)
        st.metric("Intencja", best.intent)
        st.write(f"id {best.intent_id} · score {best.score:.3f} · wersja modelu `{MODEL_VERSION}`")
        st.write("Kolejne klasy: " + ", ".join(f"`{p.intent}` ({p.score:.3f})" for p in others))

st.divider()
st.caption(
    "Model zawsze wybiera jedną z 77 znanych intencji — nie wykrywa spraw spoza zbioru (OOD) "
    "i rozumie tylko angielski. "
    f"Kod i wyniki: [repozytorium projektu]({REPO_URL}). "
    f"Dane: [BANKING77]({DATA_URL}) (PolyAI, CC BY 4.0); model bazowy: distilbert-base-uncased."
)
