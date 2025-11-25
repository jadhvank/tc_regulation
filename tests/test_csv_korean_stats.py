import io
import pytest
from types import SimpleNamespace
from importlib import reload
from fastapi.testclient import TestClient

from src.server.main import app
from src.model import litellm_client
from src.config import settings as settings_mod


@pytest.fixture(autouse=True)
def _patch_env(tmp_path, monkeypatch):
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	monkeypatch.setenv("SQL_AGENT_ENABLED", "true")
	reload(settings_mod)
	yield


@pytest.fixture(autouse=True)
def _patch_litellm(monkeypatch):
	import re

	def _extract_session_id_from_system(msgs):
		for m in msgs:
			if m.get("role") == "system":
				text = m.get("content", "")
				mm = re.search(r"session_id\s*=\s*'([^']+)'", text)
				if mm:
					return mm.group(1)
		return "sess"

	async def fake_acompletion(model, messages, **kwargs):
		sys_text = "".join(m.get("content","") for m in messages if m.get("role") == "system")
		user_text = "".join(m.get("content","") for m in messages if m.get("role") == "user")
		# SQL generation prompts include strict instructions; generate a GROUP BY over row_kv
		if "return only sql" in sys_text.lower() or "start with select" in sys_text.lower():
			sess = _extract_session_id_from_system(messages)
			sql = f"SELECT col_name, value_text, COUNT(*) AS cnt FROM row_kv WHERE session_id = '{sess}' GROUP BY col_name, value_text ORDER BY cnt DESC LIMIT 50"
			return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=sql))])
		# SQL answer summarization or general chat response
		return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="통계 요약: 각 열/값별 개수를 계산했습니다."))])

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)
	yield


def test_ingest_and_stats_from_korean_csv():
	client = TestClient(app)
	# Load test CSV from repository
	with open("tests/블록우선순위_v0.03.csv", "rb") as f:
		content = f.read()

	# Ingest via chat ingest endpoint (it will parse CSV, store SQLite row_kv, and index Chroma)
	resp = client.post("/api/v1/apps/chat/ingest", files={"files": ("블록우선순위_v0.03.csv", content, "text/csv")})
	assert resp.status_code == 200
	data = resp.json()
	assert data["doc_count"] > 0
	sess = data["session_id"]

	# Ask for statistics; prefer SQL path
	resp2 = client.post("/api/v1/apps/chat/process", json={"query": "모든 블록 정보에 대해 통계 내줘", "session_id": sess, "retrieval_mode": "sql"})
	assert resp2.status_code == 200
	out = resp2.json()
	assert any(s.get("source") == "sql_answer" for s in (out.get("sources") or []))
	# Check that answer includes a Korean statistical cue
	assert "통계" in out["answer"] or "개수" in out["answer"]


