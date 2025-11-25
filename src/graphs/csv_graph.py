from typing import Any, Dict, List, TypedDict, Optional
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, END

from src.rag.local import LocalRAG
from src.rag.hybrid import HybridRAG
from src.model.litellm_client import complete_chat
from src.config.settings import get_settings


class CSVState(TypedDict, total=False):
	session_id: str
	query: str
	k: int
	model_id: str
	retrieved: List[Dict[str, Any]]
	answer: str
	messages: List[Dict[str, str]]
	output_paths: List[str]


async def retrieve_node(state: CSVState) -> CSVState:
	settings = get_settings()
	rag = HybridRAG() if settings.HYBRID_SEARCH_ENABLED else LocalRAG()
	docs = await rag.search(session_id=state["session_id"], query=state["query"], k=state.get("k", 5))
	return {**state, "retrieved": docs}


async def generate_node(state: CSVState) -> CSVState:
	context = "\n".join([d.get("text", "") for d in state.get("retrieved", [])])
	messages = [
		{"role": "system", "content": "You are a helpful analyst. Use the provided context where relevant."},
		{"role": "user", "content": f"{context}\n\nQuestion:\n{state['query']}"},
	]
	answer = await complete_chat(messages, model_id=state.get("model_id"))
	return {**state, "answer": answer, "messages": messages + [{"role": "assistant", "content": answer}]}


async def write_node(state: CSVState) -> CSVState:
	settings = get_settings()
	session_dir = Path(settings.OUTPUT_DIR) / state["session_id"]
	session_dir.mkdir(parents=True, exist_ok=True)
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	md_path = session_dir / f"result_{ts}.md"
	sources = state.get("retrieved", [])
	with md_path.open("w", encoding="utf-8") as f:
		f.write(f"# Answer\n\n{state.get('answer','')}\n\n")
		if sources:
			f.write("## Sources\n\n")
			for i, s in enumerate(sources, 1):
				src = s.get("metadata", {}).get("file", "unknown")
				f.write(f"- [{i}] {src}\n")
	return {**state, "output_paths": [str(md_path)]}


def build_csv_graph():
	graph = StateGraph(CSVState)
	graph.add_node("retrieve", retrieve_node)
	graph.add_node("generate", generate_node)
	graph.add_node("write", write_node)
	graph.set_entry_point("retrieve")
	graph.add_edge("retrieve", "generate")
	graph.add_edge("generate", "write")
	graph.add_edge("write", END)
	return graph.compile()


