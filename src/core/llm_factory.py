"""LLM creation.

Design pattern: FACTORY.
Agents never construct a model themselves. They call get_llm() and receive
"some chat model". Switching Gemini -> OpenAI is a config.yaml change, and
no agent code needs to be touched. Example: get_llm(role="router") returns a
chat model configured by llm.roles.router merged over llm.*.
"""
from __future__ import annotations

from typing import Any

from src.core.usage_tracker import get_usage_tracker
from src.utils.config import ConfigError, get_config, get_secret
from src.utils.logger import get_logger

log = get_logger(__name__)


def _resolve_llm_config(cfg: dict[str, Any], role: str | None) -> dict[str, Any]:
    """Merge llm.roles.<role> over llm.*; a missing or unknown role gives the default."""
    base = {k: v for k, v in cfg.items() if k != "roles"}
    if role is None or role == "default":
        return base
    overrides = (cfg.get("roles") or {}).get(role)
    if overrides is None:
        log.warning("Unknown LLM role %r; using default", role)
        return base
    return {**base, **overrides}


def get_llm(temperature: float | None = None, role: str | None = None):
    """Create a chat model for `role` (default role when omitted)."""
    cfg = _resolve_llm_config(get_config()["llm"], role)
    provider = str(cfg.get("provider", "")).lower()
    temp = cfg.get("temperature") if temperature is None else temperature
    retries = cfg.get("max_retries", 3)
    # Some models only accept their default temperature, so pass it only when set.
    extra = {} if temp is None else {"temperature": temp}
    role_name = role or "default"
    callbacks = [get_usage_tracker(role_name, str(cfg.get("model")))]
    log.info("Creating LLM: provider=%s model=%s role=%s", provider, cfg.get("model"), role_name)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            max_retries=retries,
            google_api_key=get_secret("GOOGLE_API_KEY"),
            callbacks=callbacks,
            **extra,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg["model"],
            max_retries=retries,
            api_key=get_secret("OPENAI_API_KEY"),
            callbacks=callbacks,
            **extra,
        )

    raise ConfigError(f"Unsupported llm.provider: {provider!r}")
