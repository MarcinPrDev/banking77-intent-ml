# Space Hugging Face (SDK docker, szablon Streamlit): ten sam Python i te same wersje bibliotek
# co obraz API (uv.lock), torch z indeksu CPU. Model w katalogu model/ repo Space'a.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --user torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false", "--server.fileWatcherType", "none", \
     "--browser.gatherUsageStats", "false"]
