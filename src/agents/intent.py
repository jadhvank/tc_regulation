from typing import Literal, Optional

from src.model.litellm_client import complete_chat
from src.ingestion.sql_store import has_session_data


IntentMode = Literal["sql", "hybrid", "none", "both", "stats", "columns"]


_HEUR_KEYS_SQL = [
	"schema", "column", "columns", "table", "tables", "row", "rows", "count", "sum", "avg", "min", "max",
	"group by", "order by", "where", "join", "select", "총", "개수", "열", "행", "스키마",
	"합계", "분포", "비율", "통계", "統計"
]
_HEUR_KEYS_HYBRID = ["search", "find", "context", "내용", "설명", "요약", "상세"]


async def classify_intent(question: str, system_prompt: Optional[str] = None, db_context: Optional[str] = None, session_id: Optional[str] = None) -> IntentMode:
	"""
	Simple heuristic-first classifier; falls back to LLM if ambiguous.
	"""
	q = (question or "").lower()
	sql_score = sum(1 for k in _HEUR_KEYS_SQL if k in q)
	hyb_score = sum(1 for k in _HEUR_KEYS_HYBRID if k in q)
	stats_score = 1 if any(k in q for k in ["통계", "統計", "개수", "합계", "분포", "비율"]) else 0
	columns_score = 1 if any(k in q for k in ["columns", "column", "컬럼", "열 목록", "열이 뭐", "헤더", "schema", "스키마", "필드"]) else 0
	if stats_score and session_id and has_session_data(session_id):
		return "stats"
	if columns_score and session_id and has_session_data(session_id):
		return "columns"
	if sql_score > 0 and hyb_score > 0:
		return "both"
	if sql_score > 0:
		return "sql"
	if hyb_score > 0:
		return "hybrid"
	# LLM fallback (very lightweight)
	sys = "Classify the user question into one of: sql, hybrid, none, both. Reply with exactly one label."
	usr = question
	if system_prompt:
		sys += "\nSystem prompt:\n" + system_prompt
	if db_context:
		sys += "\nDB context:\n" + db_context
	prompt = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
	try:
		resp = await complete_chat(prompt)
		label = (resp or "").strip().lower()
		if label in {"sql", "hybrid", "none", "both"}:
			return label  # type: ignore[return-value]
	except Exception:
		pass
	return "none"


