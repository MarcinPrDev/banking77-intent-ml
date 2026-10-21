"""UI demo BANKING77 (stretch 3, W4) — Streamlit nad API, bez modelu w UI.

Uruchomienie (API musi działać — README → API lokalnie albo klaster kind):
    uv run streamlit run src/banking77_intent_ml/ui/app.py
Adres API: `ui.api_url` z configu, zmienna BANKING77_API_URL albo pole w panelu bocznym.
"""

from __future__ import annotations

import os

import streamlit as st

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.ui import client

settings = load_settings()
st.set_page_config(page_title="BANKING77 — intencje")
st.title("BANKING77 — klasyfikacja intencji")
st.caption(
    "Wiadomość klienta banku (po angielsku) → API `/predict` → jedna z 77 intencji. "
    "Score to wynik modelu dla zwróconej klasy, nie skalibrowane prawdopodobieństwo."
)

api_url = st.sidebar.text_input(
    "Adres API", value=os.environ.get("BANKING77_API_URL", settings.ui.api_url)
)
request = client.request_factory(api_url)
status = client.health(request)
if isinstance(status, client.ApiError):
    st.sidebar.error(status.message)
else:
    st.sidebar.success(f"API działa: model {status['model']}, wersja {status['model_version']}")

text = st.text_area(
    "Wiadomość klienta",
    max_chars=settings.api.max_text_chars,
    placeholder="I still have not received my new card",
)
if st.button("Klasyfikuj", type="primary"):
    result = client.classify(request, text)
    if isinstance(result, client.ApiError):
        st.error(result.message)
    else:
        st.metric("Intencja", result.intent)
        st.write(
            f"id {result.intent_id} · score {result.score:.3f} · "
            f"wersja modelu `{result.model_version}`"
        )
