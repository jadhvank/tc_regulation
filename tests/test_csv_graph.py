import pytest
from types import SimpleNamespace
from src.graphs.csv_graph import build_csv_graph
from src.rag.local import LocalRAG
from src.model import litellm_client
from src.config import settings as settings_mod
from importlib import reload


@pytest.mark.asyncio
async def test_csv_graph_end_to_end(tmp_path, monkeypatch):
	# isolate chroma dir
	monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
	reload(settings_mod)

	# mock LLM
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="CSV ANSWER"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

	rag = LocalRAG()
	session_id = "sess-x"
	chunks = [{"text": "row about sales and revenue", "metadata": {"file": "x.csv"}}]
	await rag.build_index(session_id, chunks)

	app = build_csv_graph()
	out = await app.ainvoke({"session_id": session_id, "query": "summarize sales", "k": 3, "model_id": "m"})
	assert out["answer"] == "CSV ANSWER"
	assert out.get("output_paths")


