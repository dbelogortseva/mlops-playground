import pytest


@pytest.fixture(autouse=True)
def disable_database_by_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
