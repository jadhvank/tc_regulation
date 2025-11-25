from typing import List, Dict, Any
import streamlit as st


def render_sources(sources: List[Dict[str, Any]] | None) -> None:
	if not sources:
		return
	with st.expander("Sources", expanded=False):
		for i, s in enumerate(sources, 1):
			src = s.get("source") or s.get("metadata", {}).get("file")
			text = s.get("text", "")
			st.markdown(f"- [{i}] {src or 'unknown'}")
			if text:
				st.code(text[:1000])


def render_answer(answer: str) -> None:
	st.markdown("### Answer")
	st.write(answer or "")


