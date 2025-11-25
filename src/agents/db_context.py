from typing import Optional, List, Dict, Any
from datetime import datetime
import sqlite3

from src.config.settings import get_settings


def _get_conn() -> sqlite3.Connection:
	settings = get_settings()
	conn = sqlite3.connect(settings.SQLITE_DB_PATH)
	conn.row_factory = sqlite3.Row
	_init_schema(conn)
	return conn


def _init_schema(conn: sqlite3.Connection) -> None:
	conn.executescript(
		"""
		CREATE TABLE IF NOT EXISTS session_profiles (
			session_id TEXT PRIMARY KEY,
			db_context TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);
		"""
	)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
	row = conn.execute(
		"SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
		(table,),
	).fetchone()
	return row is not None


def _approx_char_budget(tokens: int) -> int:
	# crude 4 chars per token budget
	return max(256, tokens * 4)


def build_db_context(session_id: str, max_tokens: Optional[int] = None) -> str:
	settings = get_settings()
	budget_chars = _approx_char_budget(max_tokens or settings.DB_CONTEXT_MAX_TOKENS)
	conn = _get_conn()
	try:
		# If ingestion tables are not initialized yet, return minimal context
		for required in ("files", "schema_columns", "rows"):
			if not _table_exists(conn, required):
				return f"Session: {session_id}\n(no indexed files yet)"
		# files for session
		files = conn.execute("SELECT id, filename FROM files WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
		lines: List[str] = []
		lines.append(f"Session: {session_id}")
		for f in files:
			file_id = f["id"]
			filename = f["filename"]
			# columns
			cols = conn.execute(
				"SELECT col_name, inferred_type, position FROM schema_columns WHERE session_id = ? AND file_id = ? ORDER BY position",
				(session_id, file_id),
			).fetchall()
			col_parts = [f"{c['col_name']}:{c['inferred_type']}" for c in cols[:20]]
			# row count
			cnt_row = conn.execute("SELECT COUNT(1) AS c FROM rows WHERE session_id = ? AND file_id = ?", (session_id, file_id)).fetchone()
			count = int(cnt_row["c"]) if cnt_row else 0
			lines.append(f"- File {filename} rows={count} cols=[{', '.join(col_parts)}]")
			if sum(len(x) for x in lines) > budget_chars:
				break
		out = "\n".join(lines)
		if len(out) > budget_chars:
			out = out[: budget_chars - 3] + "..."
		return out
	finally:
		conn.close()


def upsert_session_profile(session_id: str, db_context: str) -> None:
	conn = _get_conn()
	try:
		conn.execute(
			"INSERT INTO session_profiles(session_id, db_context, updated_at) VALUES (?, ?, ?) "
			"ON CONFLICT(session_id) DO UPDATE SET db_context = excluded.db_context, updated_at = excluded.updated_at",
			(session_id, db_context, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
		)
		conn.commit()
	finally:
		conn.close()


def get_session_profile(session_id: str) -> Optional[str]:
	conn = _get_conn()
	try:
		row = conn.execute("SELECT db_context FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()
		return row["db_context"] if row else None
	finally:
		conn.close()


def refresh_session_profile(session_id: str, max_tokens: Optional[int] = None) -> str:
	ctx = build_db_context(session_id=session_id, max_tokens=max_tokens)
	upsert_session_profile(session_id=session_id, db_context=ctx)
	return ctx


