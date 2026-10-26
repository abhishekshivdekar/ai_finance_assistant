import pytest

from src.utils.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset the config Singleton so each test starts clean."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()
