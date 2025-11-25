import io
from typing import List, Tuple
import os
import sys

# Ensure project root is on sys.path so 'ui' package is importable when Streamlit runs this file directly
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from ui.api_client import chat_ingest, chat_process, csv_ingest, csv_process, chats_list, chats_create, chats_messages
from ui.components import render_answer, render_sources


st.set_page_config(page_title="Agent Server UI", layout="wide")
st.title("Agent Server UI")

if "chat_session_id" not in st.session_state:
	st.session_state["chat_session_id"] = None
if "chat_id" not in st.session_state:
	st.session_state["chat_id"] = None
if "csv_session_id" not in st.session_state:
	st.session_state["csv_session_id"] = None

# Sidebar: Data ingest (top) and Chat History (bottom)
with st.sidebar:
	st.subheader("Data Ingest")
	with st.expander("Chat Ingest", expanded=False):
		up_files_sb = st.file_uploader("Upload .txt/.md/.csv for RAG", accept_multiple_files=True, type=["txt", "md", "csv"], key="chat_uploads")
		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Ingest to Chat Index", key="btn_ingest_chat"):
				files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
				for f in up_files_sb or []:
					files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
				with st.spinner("Ingesting..."):
					resp = chat_ingest(files=files_payload or None)
				st.session_state["chat_session_id"] = resp["session_id"]
				st.success(f"session_id={resp['session_id']}, chunks={resp['doc_count']}")
		with col_b:
			if st.button("Reset Chat Session", key="btn_reset_chat_session"):
				st.session_state["chat_session_id"] = None
				st.info("Chat session reset.")

	with st.expander("CSV/Folder Ingest", expanded=False):
		csv_files_sb = st.file_uploader("Upload CSV/TXT/MD files", accept_multiple_files=True, type=["csv", "txt", "md"], key="csv_files_sb")
		zip_file_sb = st.file_uploader("Or upload a folder ZIP", type=["zip"], key="zip_upload_sb")
		if st.button("Ingest CSV/Folder", key="btn_ingest_csv"):
			files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
			for f in csv_files_sb or []:
				files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
			zip_payload = None
			if zip_file_sb is not None:
				zip_payload = (zip_file_sb.name, zip_file_sb.getvalue(), zip_file_sb.type or "application/zip")
			with st.spinner("Ingesting..."):
				resp = csv_ingest(files=files_payload or None, folder_zip=zip_payload)
			st.session_state["csv_session_id"] = resp["session_id"]
			st.success(f"CSV session_id={resp['session_id']}, chunks={resp['doc_count']}")

	st.subheader("Chat History")
	chats = []
	try:
		chats = chats_list()
	except Exception:
		st.warning("Failed to load chat list.")
	chat_labels = [(c.get("title") or c.get("chat_id")) for c in chats]
	selected_idx = None
	if chats:
		try:
			selected_idx = st.selectbox("Select a conversation", options=list(range(len(chats))), format_func=lambda i: chat_labels[i], index=0 if st.session_state.get("chat_id") is None else next((i for i, c in enumerate(chats) if c.get("chat_id") == st.session_state.get("chat_id")), 0))
			st.session_state["chat_id"] = chats[selected_idx].get("chat_id")
		except Exception:
			pass
	col_new1, col_new2 = st.columns([1, 1])
	with col_new1:
		if st.button("New Chat"):
			try:
				created = chats_create(session_id=st.session_state.get("chat_session_id"), title=None)
				st.session_state["chat_id"] = created.get("chat_id")
				st.success(f"Created chat {st.session_state['chat_id']}")
			except Exception:
				st.error("Failed to create chat.")
	with col_new2:
		if st.button("Refresh Chats"):
			st.experimental_rerun()

tab_chat, tab_csv = st.tabs(["Chat", "CSV/Folder"])

with tab_chat:
	st.subheader("Chat")
	chat_id = st.session_state.get("chat_id")
	if not chat_id:
		st.info("Select a conversation from the sidebar or create a new chat.")
	else:
		# Load and render messages
		msgs = []
		try:
			msgs = chats_messages(chat_id=chat_id, limit=200)
		except Exception:
			st.warning("Failed to load messages.")
		# Render messages
		for m in msgs:
			role = (m.get("role") or "").lower()
			content = m.get("content") or ""
			if hasattr(st, "chat_message"):
				with st.chat_message(role if role in {"user", "assistant"} else "assistant"):
					st.markdown(content)
			else:
				st.markdown(f"**{(role or 'assistant').title()}**: {content}")

		# Single input field; submit on Enter (no Ask button, no extra fields)
		user_text = st.chat_input("메시지를 입력하세요")
		if user_text and user_text.strip():
			with st.spinner("Thinking..."):
				_ = chat_process(
					query=user_text.strip(),
					session_id=st.session_state.get("chat_session_id"),
					chat_id=st.session_state.get("chat_id"),
				)
			if hasattr(st, "rerun"):
				st.rerun()
			elif hasattr(st, "experimental_rerun"):
				st.experimental_rerun()

with tab_csv:
	st.subheader("CSV / Folder Pipeline")
	st.info("Use the sidebar to ingest CSV/Folder data. Then run processing below.")

	with st.form("csv_form", clear_on_submit=False):
		query2 = st.text_area("Task / Question", height=120)
		c1, c2 = st.columns([1, 1])
		with c1:
			k2 = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1, key="k2")
		with c2:
			model2 = st.text_input("Model Override", value="", key="model2")
		run = st.form_submit_button("Process")

	if run and query2.strip():
		if not st.session_state["csv_session_id"]:
			st.error("Please ingest files first to create a CSV session.")
		else:
			with st.spinner("Processing..."):
				resp = csv_process(session_id=st.session_state["csv_session_id"], query=query2, k=int(k2), model_id=model2 or None)
			render_answer(resp.get("answer", ""))
			render_sources(resp.get("sources"))
			file_urls = resp.get("file_urls") or []
			files = resp.get("files", [])
			if file_urls or files:
				st.markdown("#### Downloads")
				if file_urls:
					from ui.api_client import _get_api_base  # reuse base
					base = _get_api_base().rstrip("/")
					for u in file_urls:
						full = f"{base}{u}"
						name = u.split("/")[-1] if "/" in u else u
						st.link_button(f"Download {name}", full)
				else:
					# fallback to local read if URLs not provided
					for p in files:
						try:
							with open(p, "rb") as fh:
								st.download_button(label=f"Download {p}", data=fh.read(), file_name=p.split("/")[-1])
						except Exception:
							st.write(f"- {p}")


