import pytest


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch):
    """The agent resolves the OpenAI model on creation, a fake key keeps the tests offline."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
