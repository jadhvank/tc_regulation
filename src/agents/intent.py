from typing import Literal

from src.model.litellm_client import complete_chat


IntentMode = Literal["sql", "hybrid", "none", "both"]


_HEUR_KEYS_SQL = [
	"schema", "column", "columns", "table", "tables", "row", "rows", "count", "sum", "avg", "min", "max",
	"group by", "order by", "where", "join", "select", "총", "개수", "열", "행", "스키마",
]
_HEUR_KEYS_HYBRID = ["search", "find", "context", "내용", "설명", "요약", "상세"]


async def classify_intent(question: str) -> IntentMode:
	"""
	Simple heuristic-first classifier; falls back to LLM if ambiguous.
	"""
	q = (question or "").lower()
	sql_score = sum(1 for k in _HEUR_KEYS_SQL if k in q)
	hyb_score = sum(1 for k in _HEUR_KEYS_HYBRID if k in q)
	if sql_score > 0 and hyb_score > 0:
		return "both"
	if sql_score > 0:
		return "sql"
	if hyb_score > 0:
		return "hybrid"
	# LLM fallback (very lightweight)
	prompt = [
		{"role": "system", "content": "Classify the user question into one of: sql, hybrid, none, both. Reply with exactly one label."},
		{"role": "user", "content": question},
	]
	try:
		resp = await complete_chat(prompt)
		label = (resp or "").strip().lower()
		if label in {"sql", "hybrid", "none", "both"}:
			return label  # type: ignore[return-value]
	except Exception:
		pass
	return "none"


