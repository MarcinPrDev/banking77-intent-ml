"""Smoke test bramy LLM pod stretch A — LLM zero-shot (Groq przez LiteLLM).

Weryfikuje WYŁĄCZNIE środowisko i klucz. Celowo NIE dotyka danych projektu i NIE
projektuje promptu klasyfikacyjnego — dobór modelu, promptu i protokołu oceny to
decyzje merytoryczne przy starcie stretchu (README → Protokół ewaluacji, pkt 8).

Co sprawdza:
1. klucz GROQ_API_KEY jest widoczny (z `.env` albo ze zmiennych środowiskowych),
2. LiteLLM dociąga żywy katalog modeli Groq (nazwy modeli zmieniają się co kwartał —
   dlatego lista jest pobierana, a nie zapisana na sztywno),
3. jedno trywialne zapytanie przechodzi: latencja, tokeny, koszt.

Uruchomienie:
    uv sync --extra dev --extra nlp --extra gpu --extra serving --extra llm
    # pełna lista extras: `uv sync` usuwa extras pominięte w komendzie
    # klucz: skopiuj .env.example -> .env i wklej wartość (.env jest poza gitem)
    uv run python scripts/smoke_llm.py [nazwa-modelu]

Oczekiwany finał: "SMOKE TEST LLM: OK". Każdy inny wynik = stretch LLM zablokowany
(rdzeń projektu go nie potrzebuje).
"""

import os
import sys
import time

# modele nie-czatowe w katalogu Groq (transkrypcja, TTS, moderacja) — do smoke testu się nie nadają
_NON_CHAT_HINTS = ("whisper", "tts", "orpheus", "guard", "embed")
# rodziny modeli czatowych — preferowane przy automatycznym wyborze modelu do testu
_CHAT_HINTS = ("gpt-oss", "llama", "qwen", "gemma", "mixtral", "kimi", "deepseek")


def fail(msg: str) -> None:
    print(f"\n[FAIL] SMOKE TEST LLM: {msg}")
    sys.exit(1)


def _load_dotenv() -> None:
    """Wczytuje .env, jeśli python-dotenv jest dostępny (jest zależnością litellm)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        fail(
            "brak GROQ_API_KEY. Załóż klucz na https://console.groq.com (API Keys), "
            "skopiuj .env.example -> .env i wklej wartość. NIE commituj .env."
        )

    try:
        import litellm
    except ImportError:
        fail(
            "brak litellm. Zainstaluj (pełna lista extras — sync usuwa pominięte): "
            "uv sync --extra dev --extra nlp --extra gpu --extra serving --extra llm"
        )

    litellm.drop_params = True  # nieobsługiwane parametry nie mają wywracać wywołania

    try:
        models = litellm.get_valid_models(check_provider_endpoint=True, custom_llm_provider="groq")
    except Exception as exc:  # noqa: BLE001 - diagnostyka smoke testu
        fail(f"nie udało się pobrać katalogu modeli Groq: {type(exc).__name__}: {exc}")

    if not models:
        fail("Groq zwrócił pustą listę modeli — sprawdź uprawnienia klucza.")

    print(f"Katalog modeli Groq ({len(models)}):")
    for m in models:
        print(f"  - {m}")

    if len(sys.argv) > 1:
        model = sys.argv[1] if sys.argv[1].startswith("groq/") else f"groq/{sys.argv[1]}"
    else:
        chat = [m for m in models if not any(h in m.lower() for h in _NON_CHAT_HINTS)]
        preferred = [m for m in chat if any(h in m.lower() for h in _CHAT_HINTS)]
        if not chat:
            fail("w katalogu nie ma modelu czatowego — podaj model argumentem.")
        model = (preferred or chat)[0]
        print(f"\nModel nie podany — biorę pierwszy czatowy z katalogu: {model}")
        print("(wybór modelu pod stretch LLM to osobna decyzja; tu chodzi tylko o test klucza)")

    print(f"\nWywołanie testowe: {model}")
    start = time.perf_counter()
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
            temperature=0,
        )
    except litellm.AuthenticationError as exc:
        fail(f"klucz odrzucony przez Groq (AuthenticationError): {exc}")
    except litellm.RateLimitError as exc:
        fail(f"limit zapytań (RateLimitError) — klucz działa, ale jest zdławiony: {exc}")
    except litellm.APIConnectionError as exc:
        fail(f"brak połączenia z Groq (APIConnectionError): {exc}")
    except litellm.NotFoundError as exc:
        fail(f"model '{model}' nie istnieje w tym koncie (NotFoundError): {exc}")
    except Exception as exc:  # noqa: BLE001 - diagnostyka smoke testu
        fail(f"{type(exc).__name__}: {exc}")
    latency = time.perf_counter() - start

    content = response.choices[0].message.content
    usage = response.usage
    print(f"Odpowiedź: {content!r}")
    print(
        f"Latencja: {latency:.2f} s | tokeny: prompt={usage.prompt_tokens}, "
        f"completion={usage.completion_tokens}, total={usage.total_tokens}"
    )

    try:
        cost = litellm.completion_cost(completion_response=response)
        print(f"Koszt tego wywołania: ${cost:.8f}")
    except Exception as exc:  # noqa: BLE001 - brak cennika nie unieważnia smoke testu
        print(f"Koszt: nieznany ({type(exc).__name__}) — cennik modelu może nie być w LiteLLM.")
    print("(koszt i latencję mierzysz w stretchu A, na zamrożonym promptcie i 385 wiadomościach)")

    print("\n[OK] SMOKE TEST LLM — klucz Groq działa przez LiteLLM, brama gotowa pod stretch A.")


if __name__ == "__main__":
    main()
