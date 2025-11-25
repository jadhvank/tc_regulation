from typing import Optional, List, Dict, Any
from datetime import datetime
import sqlite3
from uuid import uuid4

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
		CREATE TABLE IF NOT EXISTS chats (
			chat_id TEXT PRIMARY KEY,
			title TEXT,
			session_id TEXT,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at);
		CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id);

		CREATE TABLE IF NOT EXISTS chat_messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			chat_id TEXT NOT NULL,
			role TEXT NOT NULL,     -- 'user' | 'assistant' | 'system'
			content TEXT NOT NULL,
			created_at TEXT NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id, id);
		"""
	)


def create_chat(session_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
	conn = _get_conn()
	try:
		chat_id = str(uuid4())
		now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
		conn.execute(
			"INSERT INTO chats(chat_id, title, session_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
			(chat_id, title, session_id, now, now),
		)
		conn.commit()
		return {"chat_id": chat_id, "title": title, "session_id": session_id, "created_at": now, "updated_at": now}
	finally:
		conn.close()


def update_chat_session(chat_id: str, session_id: str) -> None:
	conn = _get_conn()
	try:
		now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
		conn.execute(
			"UPDATE chats SET session_id = COALESCE(session_id, ?), updated_at = ? WHERE chat_id = ?",
			(session_id, now, chat_id),
		)
		conn.commit()
	finally:
		conn.close()


def get_chat(chat_id: str) -> Optional[Dict[str, Any]]:
	conn = _get_conn()
	try:
		row = conn.execute("SELECT chat_id, title, session_id, created_at, updated_at FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
		return dict(row) if row else None
	finally:
		conn.close()


def list_chats(limit: int = 100) -> List[Dict[str, Any]]:
	conn = _get_conn()
	try:
		cur = conn.execute(
			"SELECT chat_id, title, session_id, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?",
			(max(1, limit),),
		)
		return [dict(r) for r in cur.fetchall()]
	finally:
		conn.close()


def append_message(chat_id: str, role: str, content: str) -> int:
	conn = _get_conn()
	try:
		now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
		cur = conn.execute(
			"INSERT INTO chat_messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
			(chat_id, role, content, now),
		)
		# bump chat updated_at
		conn.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
		conn.commit()
		return int(cur.lastrowid)
	finally:
		conn.close()


def list_messages(chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
	conn = _get_conn()
	try:
		cur = conn.execute(
			"SELECT id, role, content, created_at FROM chat_messages WHERE chat_id = ? ORDER BY id ASC LIMIT ?",
			(chat_id, max(1, limit)),
		)
		return [dict(r) for r in cur.fetchall()]
	finally:
		conn.close()



