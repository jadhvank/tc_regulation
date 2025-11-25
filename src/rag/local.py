from typing import List, Dict, Any
from pathlib import Path
import hashlib
import asyncio

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings


def _hash_text(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LocalRAG:
	def __init__(self):
		settings = get_settings()
		self._client = chromadb.PersistentClient(
			path=str(Path(settings.CHROMA_DB_DIR)),
			settings=ChromaSettings(anonymized_telemetry=False),
		)

	async def build_index(self, session_id: str, chunks: List[Dict[str, Any]]) -> str:
		if not chunks:
			# Nothing to index; ensure collection exists and return
			self._client.get_or_create_collection(name=session_id, metadata={"session_id": session_id})
			return session_id

		collection = self._client.get_or_create_collection(name=session_id, metadata={"session_id": session_id})

		ids: List[str] = []
		texts: List[str] = []
		metas: List[Dict[str, Any]] = []

		for idx, ch in enumerate(chunks):
			raw_text = ch.get("text", "")
			text = (raw_text or "").strip()
			if not text:
				# Skip empty/whitespace-only chunks to avoid Chroma upsert validation errors
				continue
			meta = ch.get("metadata", {}) or {}
			doc_id = f"{idx}-{_hash_text(text)[:12]}"
			ids.append(doc_id)
			texts.append(text)
			metas.append(meta)

		# Upsert can be CPU-bound; run in a thread to avoid blocking the loop if needed
		def _upsert():
			# If everything filtered out, ensure collection exists and skip upsert
			if not ids:
				return
			collection.upsert(ids=ids, documents=texts, metadatas=metas)

		await asyncio.to_thread(_upsert)
		return session_id

	async def search(self, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
		collection = self._client.get_or_create_collection(name=session_id)

		def _query():
			return collection.query(query_texts=[query], n_results=max(1, k))

		result = await asyncio.to_thread(_query)
		out: List[Dict[str, Any]] = []
		# result shape: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
		for i, text in enumerate(result.get("documents", [[]])[0]):
			out.append(
				{
					"text": text,
					"metadata": (result.get("metadatas", [[]])[0] or [{}])[i],
					"id": (result.get("ids", [[]])[0] or [""])[i],
					"distance": (result.get("distances", [[]])[0] or [None])[i],
				}
			)
		return out


