from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END

from src.model.litellm_client import complete_chat
from src.rag.local import LocalRAG
from src.rag.hybrid import HybridRAG
from src.config.settings import get_settings
from src.agents.intent import classify_intent
from src.agents.sql_agent import run_sql, summarize_result
from src.agents.db_context import get_session_profile, refresh_session_profile
from src.ingestion.sql_store import has_session_data
from src.agents.sql_answer import answer_from_sql
from src.agents.stats_agent import compute_stats, summarize_stats


class ChatState(TypedDict, total=False):
	query: str
	system_prompt: str
	model_id: str
	k: int
	session_id: str
	retrieval_mode: str  # override: none | sql | hybrid | both
	intent_mode: str     # resolved: none | sql | hybrid | both
	db_context: str
	retrieved: List[Dict[str, Any]]
	sql_result: Dict[str, Any]
	sql_summary: str
	sql_answer_text: str
	stats_result: Dict[str, Any]
	stats_summary: str
	answer: str
	messages: List[Dict[str, str]]
async def load_db_context_node(state: ChatState) -> ChatState:
	settings = get_settings()
	if not settings.DB_CONTEXT_ENABLED or not state.get("session_id"):
		return {}
	# try load profile, if absent rebuild
	ctx = get_session_profile(state["session_id"])
	if not ctx:
		try:
			ctx = refresh_session_profile(state["session_id"], settings.DB_CONTEXT_MAX_TOKENS)
		except Exception:
			ctx = None
	if ctx:
		return {"db_context": ctx}
	return {}



async def intent_node(state: ChatState) -> ChatState:
	mode = (state.get("retrieval_mode") or "").strip().lower()
	settings = get_settings()
	if mode in {"none", "sql", "hybrid", "both"}:
		return {"intent_mode": mode}
	# decide automatically
	intent = await classify_intent(
		state.get("query", ""),
		system_prompt=state.get("system_prompt", None),
		db_context=state.get("db_context", None),
		session_id=state.get("session_id", None),
	)
	# if SQL agent disabled, downgrade sql/both intents
	if not settings.SQL_AGENT_ENABLED and intent in {"sql", "both"}:
		intent = "hybrid"
	# If session has data and intent is none, default to hybrid
	if intent == "none" and state.get("session_id") and has_session_data(state["session_id"]):
		intent = "hybrid"
	return {"intent_mode": intent}


async def sql_search_node(state: ChatState) -> ChatState:
	settings = get_settings()
	if state.get("intent_mode") not in {"sql", "both"} or not settings.SQL_AGENT_ENABLED:
		return {}
	if not state.get("session_id"):
		# Without session, we cannot scope DB; skip
		return {}
	result = await run_sql(question=state.get("query", ""), session_id=state["session_id"])
	summary = await summarize_result(result)
	return {"sql_result": result, "sql_summary": summary}


async def sql_answer_node(state: ChatState) -> ChatState:
	if not state.get("sql_result"):
		return {}
	text = await answer_from_sql(
		question=state.get("query", ""),
		sql_result=state.get("sql_result", {}),
		db_context=state.get("db_context", None),
	)
	return {"sql_answer_text": text or ""}


async def stats_compute_node(state: ChatState) -> ChatState:
	if state.get("intent_mode") != "stats" or not state.get("session_id"):
		return {}
	res = compute_stats(session_id=state["session_id"])
	summary = summarize_stats(res)
	return {"stats_result": res, "stats_summary": summary}


async def hybrid_search_node(state: ChatState) -> ChatState:
	if state.get("intent_mode") not in {"hybrid", "both"}:
		return {}
	# Optional retrieval if session_id present
	if not state.get("session_id"):
		return {}
	k = state.get("k", 5)
	settings = get_settings()
	rag = HybridRAG() if settings.HYBRID_SEARCH_ENABLED else LocalRAG()
	docs = await rag.search(session_id=state["session_id"], query=state.get("query", ""), k=k)
	return {"retrieved": docs}


async def generate_node(state: ChatState) -> ChatState:
	system_prompt = state.get("system_prompt", "You are a helpful assistant. Use the provided context sections ([DB], [STATS], [SQL], [DOCUMENTS]) strictly. Do not suggest opening files or external tools. If no context and no session_id, ask the user to ingest the CSV or provide session_id.")
	prompt = state.get("query", "")
	parts: List[str] = []
	# include DB profile first if available
	if state.get("db_context"):
		parts.append(f"[DB]\n{state['db_context']}")
	# include stats if present
	if state.get("stats_summary"):
		parts.append(f"[STATS]\n{state['stats_summary']}")
	# include SQL natural-language answer if present
	if state.get("sql_answer_text"):
		parts.append(f"[SQL_ANSWER]\n{state['sql_answer_text']}")
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
	return {"messages": messages + [{"role": "assistant", "content": answer}], "answer": answer}


def build_chat_graph():
	graph = StateGraph(ChatState)
	graph.add_node("load_db_context", load_db_context_node)
	graph.add_node("intent", intent_node)
	graph.add_node("sql_search", sql_search_node)
	graph.add_node("sql_answer", sql_answer_node)
	graph.add_node("stats_compute", stats_compute_node)
	graph.add_node("hybrid_search", hybrid_search_node)
	graph.add_node("generate", generate_node)
	graph.set_entry_point("load_db_context")
	graph.add_edge("load_db_context", "intent")
	# route intent
	graph.add_edge("intent", "sql_search")
	graph.add_edge("intent", "stats_compute")
	graph.add_edge("sql_search", "sql_answer")
	graph.add_edge("sql_answer", "hybrid_search")
	graph.add_edge("stats_compute", "generate")
	graph.add_edge("hybrid_search", "generate")
	graph.add_edge("generate", END)
	return graph.compile()


