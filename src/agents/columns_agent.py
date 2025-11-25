from typing import Any, Dict, List
import sqlite3

from src.config.settings import get_settings


def _get_conn() -> sqlite3.Connection:
	settings = get_settings()
	conn = sqlite3.connect(settings.SQLITE_DB_PATH)
	conn.row_factory = sqlite3.Row
	return conn


def get_columns(session_id: str) -> Dict[str, List[str]]:
	"""
	Return mapping of filename -> ordered list of column names for the session.
	"""
	conn = _get_conn()
	try:
		# files
		files = conn.execute("SELECT id, filename FROM files WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
		out: Dict[str, List[str]] = {}
		for f in files:
			file_id = f["id"]
			filename = f["filename"]
			cols = conn.execute(
				"SELECT col_name FROM schema_columns WHERE session_id = ? AND file_id = ? ORDER BY position",
				(session_id, file_id),
			).fetchall()
			out[filename] = [c["col_name"] for c in cols]
		return out
	finally:
		conn.close()


def summarize_columns(cols_map: Dict[str, List[str]]) -> str:
	lines: List[str] = []
	for fname, cols in cols_map.items():
		lines.append(f"{fname}: {', '.join(cols)}")
	return "\n".join(lines) if lines else "No columns found for this session."


