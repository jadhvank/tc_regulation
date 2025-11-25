import pytest
from src.rag.local import LocalRAG
from src.config.settings import get_settings
from pathlib import Path
from importlib import reload
import os


@pytest.mark.asyncio
async def test_local_rag_build_and_search(tmp_path, monkeypatch):
	# Point CHROMA_DB_DIR to tmp
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	from src.config import settings as settings_mod
	reload(settings_mod)

	rag = LocalRAG()
	session_id = "sess1"
	chunks = [{"text": "alpha beta gamma", "metadata": {"file": "a.txt"}}, {"text": "delta epsilon", "metadata": {"file": "b.txt"}}]
	await rag.build_index(session_id, chunks)
	results = await rag.search(session_id, "alpha", k=1)
	assert len(results) >= 1
	assert "alpha" in results[0]["text"]


