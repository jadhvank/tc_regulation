from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sqlite3
import json

from src.config.settings import get_settings


def _get_conn() -> sqlite3.Connection:
	settings = get_settings()
	db_path = Path(settings.SQLITE_DB_PATH)
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	conn.execute("PRAGMA journal_mode=WAL;")
	conn.execute("PRAGMA synchronous=NORMAL;")
	_init_schema(conn)
	return conn


def _init_schema(conn: sqlite3.Connection) -> None:
	conn.executescript(
		"""
		CREATE TABLE IF NOT EXISTS ingestion_sessions (
			session_id TEXT PRIMARY KEY,
			created_at TEXT NOT NULL
		);

		CREATE TABLE IF NOT EXISTS files (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			filename TEXT NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);

		CREATE TABLE IF NOT EXISTS schema_columns (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			file_id INTEGER NOT NULL,
			col_name TEXT NOT NULL,
			inferred_type TEXT NOT NULL,
			position INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_schema_columns_session ON schema_columns(session_id);
		CREATE INDEX IF NOT EXISTS idx_schema_columns_file ON schema_columns(file_id);

		CREATE TABLE IF NOT EXISTS rows (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			file_id INTEGER NOT NULL,
			row_index INTEGER,
			data_json TEXT NOT NULL,
			chunk_id TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_rows_session ON rows(session_id);
		CREATE INDEX IF NOT EXISTS idx_rows_file ON rows(file_id);
		CREATE INDEX IF NOT EXISTS idx_rows_chunk ON rows(chunk_id);

		CREATE VIRTUAL TABLE IF NOT EXISTS fts_rows USING fts5(
			text,
			session_id UNINDEXED,
			file_id UNINDEXED,
			row_index UNINDEXED,
			chunk_id UNINDEXED,
			tokenize = 'porter'
		);
		"""
	)


def ensure_session(session_id: str) -> None:
	conn = _get_conn()
	try:
		conn.execute(
			"INSERT OR IGNORE INTO ingestion_sessions(session_id, created_at) VALUES (?, ?)",
			(session_id, datetime.utcnow().isoformat(timespec="seconds") + "Z"),
		)
		conn.commit()
	finally:
		conn.close()


def _ensure_file(conn: sqlite3.Connection, session_id: str, filename: str) -> int:
	cur = conn.execute(
		"SELECT id FROM files WHERE session_id = ? AND filename = ?",
		(session_id, filename),
	)
	row = cur.fetchone()
	if row:
		return int(row["id"])
	cur = conn.execute(
		"INSERT INTO files(session_id, filename) VALUES (?, ?)",
		(session_id, filename),
	)
	return int(cur.lastrowid)


def insert_schema_columns(session_id: str, filename: str, columns: List[Dict[str, Any]]) -> None:
	conn = _get_conn()
	try:
		ensure_session(session_id)
		file_id = _ensure_file(conn, session_id, filename)
		conn.execute("DELETE FROM schema_columns WHERE session_id = ? AND file_id = ?", (session_id, file_id))
		for col in columns:
			conn.execute(
				"INSERT INTO schema_columns(session_id, file_id, col_name, inferred_type, position) VALUES (?, ?, ?, ?, ?)",
				(session_id, file_id, str(col.get("name", "")), str(col.get("type", "text")), int(col.get("position", 0))),
			)
		conn.commit()
	finally:
		conn.close()


def store_chunks(session_id: str, chunks: List[Dict[str, Any]]) -> int:
	"""
	Store chunked data rows and FTS content. Returns number of rows inserted.
	Requires each chunk to have 'text' and optional metadata including 'file', 'row_index', 'id'.
	"""
	inserted = 0
	conn = _get_conn()
	try:
		ensure_session(session_id)
		for ch in chunks:
			meta = ch.get("metadata", {}) or {}
			filename = str(meta.get("file", "unknown.txt"))
			file_id = _ensure_file(conn, session_id, filename)
			row_index = meta.get("row_index", None)
			chunk_id = ch.get("id", None)
			data_json = json.dumps({"metadata": meta, "text": ch.get("text", "")}, ensure_ascii=False)
			conn.execute(
				"INSERT INTO rows(session_id, file_id, row_index, data_json, chunk_id) VALUES (?, ?, ?, ?, ?)",
				(session_id, file_id, row_index, data_json, chunk_id),
			)
			conn.execute(
				"INSERT INTO fts_rows(text, session_id, file_id, row_index, chunk_id) VALUES (?, ?, ?, ?, ?)",
				(ch.get("text", ""), session_id, file_id, row_index, chunk_id),
			)
			inserted += 1
		conn.commit()
		return inserted
	finally:
		conn.close()


def search_fts(session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
	conn = _get_conn()
	try:
		cur = conn.execute(
			"""
			SELECT rowid, text, session_id, file_id, row_index, chunk_id, bm25(fts_rows) AS score
			FROM fts_rows
			WHERE session_id = ? AND fts_rows MATCH ?
			ORDER BY score LIMIT ?
			""",
			(session_id, query, max(1, k)),
		)
		out: List[Dict[str, Any]] = []
		for r in cur.fetchall():
			out.append(
				{
					"text": r["text"],
					"metadata": {"file_id": r["file_id"], "row_index": r["row_index"]},
					"id": r["chunk_id"],
					"score": r["score"],
				}
			)
		return out
	finally:
		conn.close()


