from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END

from src.model.litellm_client import complete_chat
from src.rag.local import LocalRAG


class ChatState(TypedDict, total=False):
	query: str
	system_prompt: str
	model_id: str
	k: int
	session_id: str
	retrieved: List[Dict[str, Any]]
	answer: str
	messages: List[Dict[str, str]]


async def retrieve_node(state: ChatState) -> ChatState:
	# Optional retrieval if session_id present
	if not state.get("session_id"):
		return state
	k = state.get("k", 5)
	rag = LocalRAG()
	docs = await rag.search(session_id=state["session_id"], query=state.get("query", ""), k=k)
	return {**state, "retrieved": docs}


async def generate_node(state: ChatState) -> ChatState:
	system_prompt = state.get("system_prompt", "You are a helpful assistant.")
	prompt = state.get("query", "")
	context = ""
	if state.get("retrieved"):
		context = "\n\n".join([d.get("text", "") for d in state["retrieved"]])
	user_content = f"{context}\n\nQuestion:\n{prompt}" if context else prompt
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content},
	]
	answer = await complete_chat(messages, model_id=state.get("model_id"))
	return {**state, "messages": messages + [{"role": "assistant", "content": answer}], "answer": answer}


def build_chat_graph():
	graph = StateGraph(ChatState)
	graph.add_node("retrieve", retrieve_node)
	graph.add_node("generate", generate_node)
	graph.set_entry_point("retrieve")
	graph.add_edge("retrieve", "generate")
	graph.add_edge("generate", END)
	return graph.compile()


