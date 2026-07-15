import os
from dotenv import load_dotenv

# Read the .env file so GROQ_API_KEY etc. are available to the code below
load_dotenv()


def get_llm(temperature: float = 0.0):
    """Return a chat model chosen by .env config, so the provider is swappable.

    Change LLM_PROVIDER / LLM_MODEL in your .env to switch models —
    no other code has to change.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "openai/gpt-oss-20b",
            temperature=temperature,
            max_retries=5,          # auto-retry on rate limits (429) with backoff
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1", temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
