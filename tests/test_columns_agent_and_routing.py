import pytest
from importlib import reload
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod
from src.agents.columns_agent import get_columns, summarize_columns


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
		return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_columns_routing_and_summary(tmp_path):
	client = TestClient(app)
	with open("tests/블록우선순위_v0.03.csv", "rb") as f:
		content = f.read()
	resp = client.post("/api/v1/apps/chat/ingest", files={"files": ("블록우선순위_v0.03.csv", content, "text/csv")})
	assert resp.status_code == 200
	sess = resp.json()["session_id"]

	# Ask to list columns - expect columns mode to kick in and include columns source
	resp2 = client.post("/api/v1/apps/chat/process", json={"query": "가지고 있는 data의 column들 목록 보여줘", "session_id": sess})
	assert resp2.status_code == 200
	out = resp2.json()
	sources = out.get("sources") or []
	assert any(s.get("source") == "columns" for s in sources)
	# The answer should be non-empty (LLM OK), but we assert routing via sources
*** End Patch
```  ***!

