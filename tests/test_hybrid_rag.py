import pytest
from importlib import reload

from src.rag.hybrid import HybridRAG
from src.config import settings as settings_mod


@pytest.mark.asyncio
async def test_hybrid_rag_build_and_search(tmp_path, monkeypatch):
	# isolate both stores
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	monkeypatch.setenv("HYBRID_SEARCH_ENABLED", "true")
	reload(settings_mod)

	rag = HybridRAG()
	session_id = "sess-hybrid"
	chunks = [{"text": "hybrid search context text", "metadata": {"file": "note.txt"}, "id": "h1"}]
	await rag.build_index(session_id=session_id, chunks=chunks)
	res = await rag.search(session_id=session_id, query="context", k=3)
	assert len(res) >= 1
	assert any("context" in r.get("text","") for r in res)


