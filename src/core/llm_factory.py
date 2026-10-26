"""LLM creation.

Design pattern: FACTORY.
Agents never construct a model themselves. They call get_llm() and receive
"some chat model". Switching Gemini -> OpenAI is a config.yaml change, and
no agent code needs to be touched.
"""
from __future__ import annotations

from src.utils.config import ConfigError, get_config, get_secret
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_llm(temperature: float | None = None):
    cfg = get_config()["llm"]
    provider = str(cfg.get("provider", "")).lower()
    temp = cfg.get("temperature", 0.2) if temperature is None else temperature
    retries = cfg.get("max_retries", 3)
    log.info("Creating LLM: provider=%s model=%s", provider, cfg.get("model"))

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            temperature=temp,
            max_retries=retries,
            google_api_key=get_secret("GOOGLE_API_KEY"),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai

        return ChatOpenAI(
            model=cfg["model"],
            temperature=temp,
            max_retries=retries,
            api_key=get_secret("OPENAI_API_KEY"),
        )

    raise ConfigError(f"Unsupported llm.provider: {provider!r}")
