from fastapi.testclient import TestClient
from types import SimpleNamespace
import io
import pandas as pd
import pytest

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod
from importlib import reload


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
	monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="CSV API ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_csv_ingest_and_process():
	client = TestClient(app)
	# create csv in-memory
	df = pd.DataFrame([{"a": 1, "b": "c"}])
	buf = io.StringIO()
	df.to_csv(buf, index=False)
	buf.seek(0)
	files = {"files": ("t.csv", buf.getvalue(), "text/csv")}
	resp = client.post("/api/v1/apps/csv/ingest", files=files)
	assert resp.status_code == 200
	data = resp.json()
	assert data["doc_count"] >= 1
	session_id = data["session_id"]

	resp2 = client.post("/api/v1/apps/csv/process", json={"session_id": session_id, "query": "summarize"})
	assert resp2.status_code == 200
	out = resp2.json()
	assert out["answer"] == "CSV API ANSWER"
	assert out["files"]


