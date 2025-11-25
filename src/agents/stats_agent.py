from typing import Any, Dict, List, Optional
import sqlite3

from src.config.settings import get_settings


def _get_conn() -> sqlite3.Connection:
	settings = get_settings()
	conn = sqlite3.connect(settings.SQLITE_DB_PATH)
	conn.row_factory = sqlite3.Row
	return conn


def _get_columns(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
	cur = conn.execute(
		"SELECT DISTINCT file_id, col_name, inferred_type FROM schema_columns WHERE session_id = ? ORDER BY file_id, col_name",
		(session_id,),
	)
	return [dict(r) for r in cur.fetchall()]


def _get_total_rows(conn: sqlite3.Connection, session_id: str) -> int:
	cur = conn.execute("SELECT COUNT(1) AS c FROM rows WHERE session_id = ?", (session_id,))
	row = cur.fetchone()
	return int(row["c"]) if row else 0


def _is_numeric(inferred_type: Optional[str]) -> bool:
	return str(inferred_type or "").lower() in {"integer", "float", "number", "numeric"}


def compute_stats(session_id: str) -> Dict[str, Any]:
	conn = _get_conn()
	try:
		total_rows = _get_total_rows(conn, session_id)
		cols = _get_columns(conn, session_id)
		per_column: Dict[str, Any] = {}
		for c in cols:
			col = c["col_name"]
			file_id = c["file_id"]
			inferred_type = c.get("inferred_type")
			# counts
			non_null = conn.execute(
				"SELECT COUNT(1) AS c FROM row_kv WHERE session_id = ? AND file_id = ? AND col_name = ? AND value_text IS NOT NULL AND value_text <> ''",
				(session_id, file_id, col),
			).fetchone()
			nulls = conn.execute(
				"SELECT COUNT(1) AS c FROM row_kv WHERE session_id = ? AND file_id = ? AND col_name = ? AND (value_text IS NULL OR value_text = '')",
				(session_id, file_id, col),
			).fetchone()
			distinct = conn.execute(
				"SELECT COUNT(DISTINCT value_text) AS c FROM row_kv WHERE session_id = ? AND file_id = ? AND col_name = ?",
				(session_id, file_id, col),
			).fetchone()
			top_vals = conn.execute(
				"SELECT value_text, COUNT(1) AS cnt FROM row_kv WHERE session_id = ? AND file_id = ? AND col_name = ? GROUP BY value_text ORDER BY cnt DESC LIMIT 5",
				(session_id, file_id, col),
			).fetchall()
			item: Dict[str, Any] = {
				"file_id": file_id,
				"inferred_type": inferred_type,
				"non_null_count": int(non_null["c"]) if non_null else 0,
				"null_count": int(nulls["c"]) if nulls else 0,
				"distinct_count": int(distinct["c"]) if distinct else 0,
				"top_values": [{"value": r["value_text"], "count": int(r["cnt"])} for r in top_vals],
			}
			# numeric aggregates
			if _is_numeric(inferred_type):
				num = conn.execute(
					"SELECT MIN(CAST(value_text AS REAL)) AS mn, MAX(CAST(value_text AS REAL)) AS mx, AVG(CAST(value_text AS REAL)) AS av "
					"FROM row_kv WHERE session_id = ? AND file_id = ? AND col_name = ? AND value_text IS NOT NULL AND value_text <> ''",
					(session_id, file_id, col),
				).fetchone()
				item["min"] = num["mn"]
				item["max"] = num["mx"]
				item["avg"] = num["av"]
			per_column[f"{file_id}:{col}"] = item
		return {"total_rows": total_rows, "columns": per_column}
	finally:
		conn.close()


def summarize_stats(stats: Dict[str, Any]) -> str:
	total = stats.get("total_rows", 0)
	lines = [f"총 행 개수: {total}"]
	cols = stats.get("columns", {})
	for key, info in list(cols.items())[:10]:
		col_name = key.split(":", 1)[1] if ":" in key else key
		nn = info.get("non_null_count", 0)
		nu = info.get("null_count", 0)
		ds = info.get("distinct_count", 0)
		top = info.get("top_values", [])[:3]
		top_str = ", ".join([f"{t['value']}({t['count']})" for t in top if t["value"] is not None])
		line = f"- {col_name}: 비결측={nn}, 결측={nu}, 고유값={ds}"
		if top_str:
			line += f", 상위값={top_str}"
		if "avg" in info and info["avg"] is not None:
			line += f", 평균={round(float(info['avg']), 3)}"
		lines.append(line)
	return "\n".join(lines)


