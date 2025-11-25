from fastapi.testclient import TestClient
from types import SimpleNamespace
import pytest

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod
from importlib import reload


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="API-ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_health():
	client = TestClient(app)
	resp = client.get("/api/v1/health")
	assert resp.status_code == 200
	assert resp.json()["status"] == "ok"


def test_chat_process():
	client = TestClient(app)
	resp = client.post("/api/v1/apps/chat/process", json={"query": "Hello?"})
	assert resp.status_code == 200
	data = resp.json()
	assert data["answer"] == "API-ANSWER"
	assert "model_id" in data



