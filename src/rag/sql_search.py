from typing import List, Dict, Any

from src.ingestion.sql_store import search_fts


class SqlSearch:
	def __init__(self) -> None:
		pass

	async def search(self, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
		# synchronous sqlite; call directly (simple and fast for local use)
		return search_fts(session_id=session_id, query=query, k=k)


