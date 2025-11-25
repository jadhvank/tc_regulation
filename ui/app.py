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

from ui.api_client import chat_ingest, chat_process, csv_ingest, csv_process
from ui.components import render_answer, render_sources


st.set_page_config(page_title="Agent Server UI", layout="wide")
st.title("Agent Server UI")

if "chat_session_id" not in st.session_state:
	st.session_state["chat_session_id"] = None
if "chat_history" not in st.session_state:
	st.session_state["chat_history"] = []  # list of {"role","content"}
if "csv_session_id" not in st.session_state:
	st.session_state["csv_session_id"] = None

tab_chat, tab_csv = st.tabs(["Chat", "CSV/Folder"])

with tab_chat:
	st.subheader("Chat with optional RAG")
	up_files = st.file_uploader("Upload .txt/.md/.csv for RAG (optional)", accept_multiple_files=True, type=["txt", "md", "csv"])
	col_ing1, col_ing2 = st.columns([1, 4])
	with col_ing1:
		if st.button("Ingest to Chat Index"):
			files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
			for f in up_files or []:
				files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
			with st.spinner("Ingesting..."):
				resp = chat_ingest(files=files_payload or None)
			st.session_state["chat_session_id"] = resp["session_id"]
			st.success(f"Ingested {resp['doc_count']} chunks. session_id={resp['session_id']}")
	with col_ing2:
		if st.button("Reset Chat Session"):
			st.session_state["chat_session_id"] = None
			st.info("Chat session reset.")

	with st.form("chat_form", clear_on_submit=False):
		system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)
		query = st.text_area("Your Question", height=120)
		c1, c2, c3 = st.columns([1, 1, 2])
		with c1:
			k = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
		with c2:
			model_id = st.text_input("Model Override", value="")
		submitted = st.form_submit_button("Ask")

	if submitted and query.strip():
		with st.spinner("Thinking..."):
			resp = chat_process(
				query=query,
				session_id=st.session_state["chat_session_id"],
				k=int(k),
				system_prompt=system_prompt or None,
				model_id=model_id or None,
			)
		# append to history
		st.session_state["chat_history"].append({"role": "user", "content": query})
		st.session_state["chat_history"].append({"role": "assistant", "content": resp.get("answer", "")})
		render_answer(resp.get("answer", ""))
		render_sources(resp.get("sources"))

	if st.session_state["chat_history"]:
		with st.expander("Conversation History", expanded=False):
			for msg in st.session_state["chat_history"][-20:]:
				st.markdown(f"**{msg['role'].title()}**: {msg['content']}")
	if st.button("Clear Conversation"):
		st.session_state["chat_history"] = []
		st.success("Conversation cleared.")

with tab_csv:
	st.subheader("CSV / Folder Pipeline")
	cu, zu = st.columns([3, 2])
	with cu:
		csv_files = st.file_uploader("Upload CSV/TXT/MD files", accept_multiple_files=True, type=["csv", "txt", "md"], key="csv_files")
	with zu:
		zip_file = st.file_uploader("Or upload a folder ZIP", type=["zip"], key="zip_upload")

	ing_col1, _ = st.columns([1, 5])
	with ing_col1:
		if st.button("Ingest CSV/Folder"):
			files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
			for f in csv_files or []:
				files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
			zip_payload = None
			if zip_file is not None:
				zip_payload = (zip_file.name, zip_file.getvalue(), zip_file.type or "application/zip")
			with st.spinner("Ingesting..."):
				resp = csv_ingest(files=files_payload or None, folder_zip=zip_payload)
			st.session_state["csv_session_id"] = resp["session_id"]
			st.success(f"Ingested {resp['doc_count']} chunks. session_id={resp['session_id']}")

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


