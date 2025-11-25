from typing import List, Dict
from abc import ABC, abstractmethod


class RAGAdapter(ABC):
	@abstractmethod
	async def build_index(self, session_id: str, chunks: List[Dict]) -> str: ...

	@abstractmethod
	async def search(self, session_id: str, query: str, k: int = 5) -> List[Dict]: ...


