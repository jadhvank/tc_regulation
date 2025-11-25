from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END

from src.model.litellm_client import complete_chat
from src.rag.local import LocalRAG
from src.rag.hybrid import HybridRAG
from src.config.settings import get_settings
from src.agents.intent import classify_intent
from src.agents.sql_agent import run_sql, summarize_result


class ChatState(TypedDict, total=False):
	query: str
	system_prompt: str
	model_id: str
	k: int
	session_id: str
	retrieval_mode: str  # override: none | sql | hybrid | both
	intent_mode: str     # resolved: none | sql | hybrid | both
	retrieved: List[Dict[str, Any]]
	sql_result: Dict[str, Any]
	sql_summary: str
	answer: str
	messages: List[Dict[str, str]]


async def intent_node(state: ChatState) -> ChatState:
	mode = (state.get("retrieval_mode") or "").strip().lower()
	settings = get_settings()
	if mode in {"none", "sql", "hybrid", "both"}:
		return {**state, "intent_mode": mode}
	# decide automatically
	intent = await classify_intent(state.get("query", ""))
	# if SQL agent disabled, downgrade sql/both intents
	if not settings.SQL_AGENT_ENABLED and intent in {"sql", "both"}:
		intent = "hybrid"
	return {**state, "intent_mode": intent}


async def sql_search_node(state: ChatState) -> ChatState:
	settings = get_settings()
	if state.get("intent_mode") not in {"sql", "both"} or not settings.SQL_AGENT_ENABLED:
		return state
	if not state.get("session_id"):
		# Without session, we cannot scope DB; skip
		return state
	result = await run_sql(question=state.get("query", ""), session_id=state["session_id"])
	summary = await summarize_result(result)
	return {**state, "sql_result": result, "sql_summary": summary}


async def hybrid_search_node(state: ChatState) -> ChatState:
	if state.get("intent_mode") not in {"hybrid", "both"}:
		return state
	# Optional retrieval if session_id present
	if not state.get("session_id"):
		return state
	k = state.get("k", 5)
	settings = get_settings()
	rag = HybridRAG() if settings.HYBRID_SEARCH_ENABLED else LocalRAG()
	docs = await rag.search(session_id=state["session_id"], query=state.get("query", ""), k=k)
	return {**state, "retrieved": docs}


async def generate_node(state: ChatState) -> ChatState:
	system_prompt = state.get("system_prompt", "You are a helpful assistant.")
	prompt = state.get("query", "")
	parts: List[str] = []
	# prefer SQL summary first if present
	if state.get("sql_summary"):
		parts.append(f"[SQL]\n{state['sql_summary']}")
	if state.get("retrieved"):
		texts = "\n\n".join([d.get("text", "") for d in state["retrieved"]])
		parts.append(f"[DOCUMENTS]\n{texts}")
	context = "\n\n".join([p for p in parts if p])
	user_content = f"{context}\n\nQuestion:\n{prompt}" if context else prompt
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content},
	]
	answer = await complete_chat(messages, model_id=state.get("model_id"))
	return {**state, "messages": messages + [{"role": "assistant", "content": answer}], "answer": answer}


def build_chat_graph():
	graph = StateGraph(ChatState)
	graph.add_node("intent", intent_node)
	graph.add_node("sql_search", sql_search_node)
	graph.add_node("hybrid_search", hybrid_search_node)
	graph.add_node("generate", generate_node)
	graph.set_entry_point("intent")
	graph.add_edge("intent", "sql_search")
	graph.add_edge("sql_search", "hybrid_search")
	graph.add_edge("hybrid_search", "generate")
	graph.add_edge("generate", END)
	return graph.compile()


