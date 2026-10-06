import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path / "llm"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
