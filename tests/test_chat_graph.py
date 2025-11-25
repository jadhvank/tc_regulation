import pytest
from types import SimpleNamespace

from src.graphs.chat_graph import build_chat_graph
from src.model import litellm_client


@pytest.mark.asyncio
async def test_chat_graph_generate(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="A1"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

	app = build_chat_graph()
	state = {"query": "Q1", "system_prompt": "S1", "model_id": "m1"}
	out = await app.ainvoke(state)
	assert out["answer"] == "A1"
	assert out["messages"][-1]["content"] == "A1"


