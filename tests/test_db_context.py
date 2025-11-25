from importlib import reload

from src.agents.db_context import refresh_session_profile, get_session_profile, build_db_context
from src.ingestion.sql_store import store_chunks, insert_schema_columns
from src.config import settings as settings_mod


def test_db_context_build_and_store(tmp_path, monkeypatch):
	monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "app.db"))
	monkeypatch.setenv("DB_CONTEXT_MAX_TOKENS", "64")
	reload(settings_mod)

	session_id = "sess-dbctx"
	# seed minimal schema and rows
	insert_schema_columns(session_id, "a.csv", [{"name": "col1", "type": "text", "position": 0}, {"name": "col2", "type": "integer", "position": 1}])
	store_chunks(session_id, [{"text": "col1: v, col2: 1", "metadata": {"file": "a.csv", "row_index": 0}, "id": "z1"}])

	ctx = refresh_session_profile(session_id)
	assert ctx and "a.csv" in ctx

	fetched = get_session_profile(session_id)
	assert fetched and "a.csv" in fetched


