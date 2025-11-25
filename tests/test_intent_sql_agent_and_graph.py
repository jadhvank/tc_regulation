import pytest
from types import SimpleNamespace
from importlib import reload

from src.server.main import app
from src.model import litellm_client
from src.ingestion.sql_store import store_chunks
from src.config import settings as settings_mod
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	monkeypatch.setenv("SQL_AGENT_ENABLED", "true")
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		# Always produce a simple response or basic SELECT for SQL generation
		user = "".join(m.get("content","") for m in messages if m.get("role") == "user").lower()
		if "select" in user or "schema" in user or "rows" in user:
			# SQL generation path
			return SimpleNamespace(
				choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT session_id, file_id, row_index FROM rows WHERE session_id = 'sess'"))]
			)
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_graph_modes(tmp_path):
	client = TestClient(app)
	# Seed minimal chunks into SQLite; skip chroma for speed
	store_chunks(session_id="sess", chunks=[{"text": "alpha", "metadata": {"file": "a.txt", "row_index": 0}, "id": "c1"}])

	# none mode
	resp = client.post("/api/v1/apps/chat/process", json={"query": "just answer plainly", "retrieval_mode": "none"})
	assert resp.status_code == 200

	# sql mode
	resp = client.post("/api/v1/apps/chat/process", json={"query": "show me rows", "retrieval_mode": "sql", "session_id": "sess"})
	assert resp.status_code == 200
	data = resp.json()
	assert data["sources"] is None or any(s.get("source") == "sql" for s in data.get("sources") or [])

	# hybrid mode (no session -> should still succeed without retrieval)
	resp = client.post("/api/v1/apps/chat/process", json={"query": "find alpha", "retrieval_mode": "hybrid"})
	assert resp.status_code == 200

	# both mode
	resp = client.post("/api/v1/apps/chat/process", json={"query": "find alpha and show rows", "retrieval_mode": "both", "session_id": "sess"})
	assert resp.status_code == 200
	data = resp.json()
	# may include both sql and docs depending on chroma population; we at least check not failing
	assert "answer" in data


