from typing import Any, Dict, List, Tuple
import asyncio
import sqlite3
import re

from src.config.settings import get_settings
from src.model.litellm_client import complete_chat


_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|vacuum|reindex|replace|truncate)\b", re.IGNORECASE)


def _enforce_select_only(sql: str) -> str:
	stmt = sql.strip().strip(";")
	if not _SELECT_RE.match(stmt):
		raise ValueError("Only SELECT statements are allowed")
	if _FORBIDDEN.search(stmt):
		raise ValueError("Forbidden statement detected")
	# basic single statement enforcement
	if ";" in sql.strip()[:-1]:
		raise ValueError("Only one statement is allowed")
	return stmt


def _inject_limit(stmt: str, limit: int) -> str:
	# naive: if no LIMIT present, append one
	if re.search(r"\blimit\s+\d+", stmt, re.IGNORECASE):
		return stmt
	return f"{stmt} LIMIT {int(limit)}"


def _strip_code_fences(text: str) -> str:
	lines = []
	skip = False
	for ln in (text or "").splitlines():
		if ln.strip().startswith("```"):
			# toggle; drop fence lines
			skip = not skip
			continue
		lines.append(ln)
	return "\n".join(lines)


def _extract_select(raw: str) -> str | None:
	if not raw:
		return None
	text = _strip_code_fences(raw).strip()
	m = re.search(r"select\\b[\\s\\S]*", text, re.IGNORECASE)
	if not m:
		return None
	stmt = m.group(0).strip()
	# Stop at first standalone semicolon if present
	semi = stmt.find(";")
	if semi != -1:
		stmt = stmt[:semi]
	return stmt


async def generate_sql(question: str, session_id: str) -> str:
	"""
	Generate a SQLite SELECT for our schema (schema_columns, files, rows, fts_rows).
	Always filter by session_id.
	"""
	system = (
		"You write only safe SQLite SELECT queries for tables: "
		"schema_columns(session_id,file_id,col_name,inferred_type,position), "
		"files(id,session_id,filename), "
		"rows(session_id,file_id,row_index,data_json,chunk_id), "
		"fts_rows(text,session_id,file_id,row_index,chunk_id), "
		"row_kv(session_id,file_id,row_index,col_name,value_text). "
		"For statistics and counts, prefer row_kv with GROUP BY col_name,value_text. "
		"Constraints: Use WHERE session_id = '{session_id}'. No PRAGMA/ATTACH/DDL/DML. Return only SQL, start with SELECT, no prose, no backticks."
	).replace("{session_id}", session_id.replace("'", "''"))
	msgs = [
		{"role": "system", "content": system},
		{"role": "user", "content": question},
	]
	sql = await complete_chat(msgs)
	return sql


def _execute_sql(sql: str) -> Tuple[List[str], List[List[Any]], int]:
	settings = get_settings()
	conn = sqlite3.connect(settings.SQLITE_DB_PATH)
	try:
		conn.row_factory = sqlite3.Row
		# read-only mode best-effort
		try:
			conn.execute("PRAGMA query_only = ON;")
		except Exception:
			pass
		cur = conn.execute(sql)
		rows = cur.fetchall()
		cols = [d[0] for d in cur.description] if cur.description else []
		return cols, [list(r) for r in rows], len(rows)
	finally:
		conn.close()


async def run_sql(question: str, session_id: str) -> Dict[str, Any]:
	settings = get_settings()
	raw = await generate_sql(question=question, session_id=session_id)
	# Try to extract a SELECT statement from the model output
	stmt_candidate = _extract_select(raw)
	if not stmt_candidate:
		return {"sql": None, "error": "no_select_statement_generated"}
	# Enforce safety and limit; wrap in try to avoid raising to graph
	try:
		stmt = _enforce_select_only(stmt_candidate)
		stmt = _inject_limit(stmt, settings.SQL_MAX_ROWS)
	except Exception as e:
		return {"sql": stmt_candidate, "error": f"{e.__class__.__name__}: {e}"}

	async def _run():
		return _execute_sql(stmt)

	try:
		cols, rows, count = await asyncio.wait_for(asyncio.to_thread(_run), timeout=5.0)
	except asyncio.TimeoutError:
		return {"sql": stmt, "error": "timeout"}
	except Exception as e:
		return {"sql": stmt, "error": f"{e.__class__.__name__}: {e}"}

	return {"sql": stmt, "columns": cols, "rows": rows, "row_count": count}


async def summarize_result(result: Dict[str, Any]) -> str:
	if result.get("error"):
		return f"SQL error: {result['error']}"
	cols = result.get("columns", [])
	rows = result.get("rows", [])
	head = rows[:5]
	return f"SQL Rows (showing up to 5): columns={cols}\n{head}"


