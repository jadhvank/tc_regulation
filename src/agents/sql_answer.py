from typing import Any, Dict, Optional

from src.model.litellm_client import complete_chat


async def answer_from_sql(question: str, sql_result: Dict[str, Any], db_context: Optional[str] = None) -> str:
	"""
	Produce a concise NL answer from a SQL result.
	"""
	if not sql_result or sql_result.get("error"):
		err = sql_result.get("error") or "unknown"
		return f"SQL returned an error or empty result: {err}"

	cols = sql_result.get("columns", [])
	rows = sql_result.get("rows", [])[:10]
	row_count = sql_result.get("row_count", 0)

	sections = []
	if db_context:
		sections.append(f"[DB]\n{db_context}")
	sections.append(f"[SQL_RESULT]\ncolumns={cols}\nrows(sample)={rows}\nrow_count={row_count}")
	sections.append(f"Question:\n{question}")
	user_content = "\n\n".join(sections)

	messages = [
		{
			"role": "system",
			"content": (
				"You are a precise analyst. Write a concise answer using the SQL result. "
				"Summarize key findings, include simple calculations if helpful, and reference columns or files when relevant."
			),
		},
		{"role": "user", "content": user_content},
	]

	text = await complete_chat(messages)
	return text


