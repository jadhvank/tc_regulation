import pytest
from importlib import reload
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod
from src.ingestion.sql_store import insert_schema_columns, store_chunks
from src.agents.stats_agent import compute_stats, summarize_stats


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	monkeypatch.setenv("SQL_AGENT_ENABLED", "true")
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	# Keep answers deterministic
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_stats_agent_basic():
	# Seed a tiny dataset
	session_id = "sess-stats"
	insert_schema_columns(session_id, "a.csv", [{"name": "colA", "type": "text", "position": 0}, {"name": "num", "type": "integer", "position": 1}])
	store_chunks(session_id, [
		{"text": "colA: x, num: 1", "metadata": {"file": "a.csv", "row_index": 0}, "id": "c1", "structured": {"colA": "x", "num": "1"}},
		{"text": "colA: y, num: 2", "metadata": {"file": "a.csv", "row_index": 1}, "id": "c2", "structured": {"colA": "y", "num": "2"}},
		{"text": "colA: x, num: 3", "metadata": {"file": "a.csv", "row_index": 2}, "id": "c3", "structured": {"colA": "x", "num": "3"}},
	])
	stats = compute_stats(session_id)
	assert stats["total_rows"] >= 3
	summary = summarize_stats(stats)
	assert "총 행 개수" in summary


def test_stats_routing_e2e(tmp_path):
	client = TestClient(app)
	# Ingest file via API for session
	with open("tests/블록우선순위_v0.03.csv", "rb") as f:
		content = f.read()
	resp = client.post("/api/v1/apps/chat/ingest", files={"files": ("블록우선순위_v0.03.csv", content, "text/csv")})
	assert resp.status_code == 200
	sess = resp.json()["session_id"]

	# Ask a stats query; no override; intent should detect stats -> stats_compute used
	resp2 = client.post("/api/v1/apps/chat/process", json={"query": "모든 블록 정보에 대해 통계 내줘", "session_id": sess})
	assert resp2.status_code == 200
	out = resp2.json()
	sources = out.get("sources") or []
	assert any(s.get("source") == "stats" for s in sources)
*** End Patch***}؟``` ***!

