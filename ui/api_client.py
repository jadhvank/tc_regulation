import os
from typing import Any, Dict, List, Optional, Tuple

import httpx


def _get_api_base() -> str:
	# Prefer Streamlit secrets if available; fallback to env; else default
	try:
		import streamlit as st  # type: ignore

		if "API_BASE_URL" in st.secrets:
			return str(st.secrets["API_BASE_URL"])
	except Exception:
		pass
	return os.getenv("API_BASE_URL", "http://localhost:8000")


def _client() -> httpx.Client:
	return httpx.Client(base_url=_get_api_base(), timeout=60)


def chat_ingest(files: List[Tuple[str, Tuple[str, bytes, str]]] | None = None, folder_zip: Optional[Tuple[str, bytes, str]] = None) -> Dict[str, Any]:
	multipart: List[Tuple[str, Tuple[str, bytes, str]]] = []
	for item in files or []:
		# item should be ("files", (filename, content_bytes, mime))
		multipart.append(item)
	if folder_zip:
		multipart.append(("folder_zip", folder_zip))
	with _client() as c:
		resp = c.post("/api/v1/apps/chat/ingest", files=multipart or None)
		resp.raise_for_status()
		return resp.json()


def chat_process(query: str, session_id: Optional[str] = None, k: Optional[int] = 5, system_prompt: Optional[str] = None, model_id: Optional[str] = None, chat_id: Optional[str] = None) -> Dict[str, Any]:
	payload: Dict[str, Any] = {"query": query}
	if session_id:
		payload["session_id"] = session_id
	if chat_id:
		payload["chat_id"] = chat_id
	if k is not None:
		payload["k"] = k
	if system_prompt:
		payload["system_prompt"] = system_prompt
	if model_id:
		payload["model_id"] = model_id
	with _client() as c:
		resp = c.post("/api/v1/apps/chat/process", json=payload)
		resp.raise_for_status()
		return resp.json()


def csv_ingest(files: List[Tuple[str, Tuple[str, bytes, str]]] | None = None, folder_zip: Optional[Tuple[str, bytes, str]] = None) -> Dict[str, Any]:
	multipart: List[Tuple[str, Tuple[str, bytes, str]]] = []
	for item in files or []:
		multipart.append(item)
	if folder_zip:
		multipart.append(("folder_zip", folder_zip))
	with _client() as c:
		resp = c.post("/api/v1/apps/csv/ingest", files=multipart or None)
		resp.raise_for_status()
		return resp.json()


def csv_process(session_id: str, query: str, k: Optional[int] = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
	payload: Dict[str, Any] = {"session_id": session_id, "query": query}
	if k is not None:
		payload["k"] = k
	if model_id:
		payload["model_id"] = model_id
	with _client() as c:
		resp = c.post("/api/v1/apps/csv/process", json=payload)
		resp.raise_for_status()
		return resp.json()


def chats_create(session_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
	payload: Dict[str, Any] = {}
	if session_id:
		payload["session_id"] = session_id
	if title:
		payload["title"] = title
	with _client() as c:
		resp = c.post("/api/v1/chats", json=payload or {})
		resp.raise_for_status()
		return resp.json()


def chats_list() -> List[Dict[str, Any]]:
	with _client() as c:
		resp = c.get("/api/v1/chats")
		resp.raise_for_status()
		data = resp.json() or {}
		return list(data.get("chats", []))


def chats_messages(chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
	with _client() as c:
		resp = c.get(f"/api/v1/chats/{chat_id}/messages", params={"limit": limit})
		resp.raise_for_status()
		data = resp.json() or {}
		return list(data.get("messages", []))


def get_config() -> Dict[str, Any]:
	with _client() as c:
		resp = c.get("/api/v1/config")
		resp.raise_for_status()
		return resp.json()


def update_config(llm_model_id: Optional[str] = None, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
	payload: Dict[str, Any] = {}
	if llm_model_id:
		payload["llm_model_id"] = llm_model_id
	if openai_api_key:
		payload["openai_api_key"] = openai_api_key
	if anthropic_api_key:
		payload["anthropic_api_key"] = anthropic_api_key
	with _client() as c:
		resp = c.post("/api/v1/config", json=payload or {})
		resp.raise_for_status()
		return resp.json()


