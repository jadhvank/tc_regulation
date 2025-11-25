from fastapi.testclient import TestClient
from types import SimpleNamespace
import pytest

from src.server.main import app
from src.model import litellm_client
from src.rag.local import LocalRAG
from src.config import settings as settings_mod
from importlib import reload


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="CHAT ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


@pytest.mark.asyncio
async def test_chat_with_session_sources():
	client = TestClient(app)
	# Build small index first
	rag = LocalRAG()
	session_id = "sess-chat"
	await rag.build_index(session_id, [{"text": "context for chat", "metadata": {"file": "note.txt"}}])

	resp = client.post("/api/v1/apps/chat/process", json={"query": "use context", "session_id": session_id})
	assert resp.status_code == 200
	data = resp.json()
	assert data["answer"] == "CHAT ANSWER"
	assert data["sources"] is not None


