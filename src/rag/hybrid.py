from typing import List, Dict, Any, Tuple
import asyncio

from src.rag.local import LocalRAG
from src.rag.sql_search import SqlSearch
from src.ingestion.sql_store import store_chunks


def _rrf(scores: List[Any], k: int = 60) -> Dict[str, float]:
	"""
	Compute Reciprocal Rank Fusion contributions for a ranked list of items.
	Items must have unique stable keys 'key' in each element.
	Returns mapping key -> contribution.
	"""
	out: Dict[str, float] = {}
	for rank, item in enumerate(scores, start=1):
		key = item["key"]
		out[key] = out.get(key, 0.0) + 1.0 / (k + rank)
	return out


def _make_key(item: Dict[str, Any]) -> str:
	if item.get("id"):
		return str(item["id"])
	meta = item.get("metadata", {}) or {}
	return f"{meta.get('file','unknown')}::{meta.get('row_index','')}::{(item.get('text') or '')[:64]}"


class HybridRAG:
	def __init__(self):
		self._vec = LocalRAG()
		self._sql = SqlSearch()

	async def build_index(self, session_id: str, chunks: List[Dict[str, Any]]) -> str:
		# Write to SQL store as well for robustness when called directly
		if chunks:
			store_chunks(session_id=session_id, chunks=chunks)
		return await self._vec.build_index(session_id=session_id, chunks=chunks)

	async def search(self, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
		# Run in parallel
		vec_task = asyncio.create_task(self._vec.search(session_id=session_id, query=query, k=k))
		sql_task = asyncio.create_task(self._sql.search(session_id=session_id, query=query, k=k))
		vec_res, sql_res = await asyncio.gather(vec_task, sql_task)

		# Build ranked keys
		vec_ranked = [{"key": _make_key(r), "item": r} for r in vec_res]
		sql_ranked = [{"key": _make_key(r), "item": r} for r in sql_res]

		score_map: Dict[str, float] = {}
		for m in (_rrf(vec_ranked), _rrf(sql_ranked)):
			for k_, v in m.items():
				score_map[k_] = score_map.get(k_, 0.0) + v

		# Merge by key; prefer vector item when duplicate to keep distance field
		merged: Dict[str, Dict[str, Any]] = {}
		for r in vec_ranked + sql_ranked:
			if r["key"] not in merged:
				merged[r["key"]] = r["item"]

		# Sort by fused score desc
		results = sorted(merged.values(), key=lambda x: score_map.get(_make_key(x), 0.0), reverse=True)
		return results[:k]


