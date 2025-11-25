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
		# Emit a simple SQL when asked, otherwise generic
		user = "".join(m.get("content","") for m in messages if m.get("role") == "user").lower()
		sys = "".join(m.get("content","") for m in messages if m.get("role") == "system").lower()
		if "return only sql" in sys or "start with select" in sys:
			return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT row_index FROM rows WHERE session_id = 'sess' LIMIT 5"))])
		return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_default_to_retrieval_when_session_has_data(tmp_path):
	client = TestClient(app)
	# Seed one row so session has data
	store_chunks(session_id="sess", chunks=[{"text": "alpha", "metadata": {"file": "a.txt", "row_index": 0}, "id": "c1"}])

	# No retrieval_mode override; generic query with no keywords -> was 'none', should become 'hybrid'
	resp = client.post("/api/v1/apps/chat/process", json={"query": "hello there", "session_id": "sess"})
	assert resp.status_code == 200
	meta = resp.json().get("meta") or {}
	assert meta.get("intent_mode") in {"hybrid", "sql", "both"}


def test_sql_answer_in_sources(tmp_path):
	client = TestClient(app)
	store_chunks(session_id="sess", chunks=[{"text": "alpha", "metadata": {"file": "a.txt", "row_index": 0}, "id": "c1"}])

	resp = client.post("/api/v1/apps/chat/process", json={"query": "show rows", "session_id": "sess", "retrieval_mode": "sql"})
	assert resp.status_code == 200
	sources = resp.json().get("sources") or []
	assert any(s.get("source") == "sql_answer" for s in sources)


