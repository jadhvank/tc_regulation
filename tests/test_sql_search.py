import os
from importlib import reload

from src.ingestion.sql_store import store_chunks, search_fts
from src.config import settings as settings_mod


def test_sql_store_and_search(tmp_path, monkeypatch):
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	reload(settings_mod)

	session_id = "sess-sql"
	chunks = [
		{"text": "hello world keyword", "metadata": {"file": "a.txt", "row_index": 0}, "id": "chunk-1"},
		{"text": "another line without it", "metadata": {"file": "a.txt", "row_index": 1}, "id": "chunk-2"},
	]

	store_chunks(session_id=session_id, chunks=chunks)

	res = search_fts(session_id=session_id, query="keyword", k=3)
	assert len(res) >= 1
	assert "keyword" in res[0]["text"]


