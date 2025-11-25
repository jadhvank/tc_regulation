from fastapi.testclient import TestClient
from types import SimpleNamespace
import io
import pytest

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod
from importlib import reload


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="CHAT AFTER ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_chat_ingest_then_process():
	client = TestClient(app)
	# Upload a text file for chat
	buf = io.BytesIO(b"this is chat context")
	files = {"files": ("note.txt", buf.getvalue(), "text/plain")}
	resp = client.post("/api/v1/apps/chat/ingest", files=files)
	assert resp.status_code == 200
	data = resp.json()
	assert data["doc_count"] >= 1
	session_id = data["session_id"]

	resp2 = client.post("/api/v1/apps/chat/process", json={"query": "use context", "session_id": session_id})
	assert resp2.status_code == 200
	out = resp2.json()
	assert out["answer"] == "CHAT AFTER ANSWER"
	assert out["sources"] is not None


