import types
import asyncio
from types import SimpleNamespace

import pytest

from src.model import litellm_client


@pytest.mark.asyncio
async def test_complete_chat_monkeypatched(monkeypatch):
	async def fake_acompletion(model, messages, **kwargs):
		return SimpleNamespace(
			choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))]
		)

	monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

	out = await litellm_client.complete_chat(
		[{"role": "system", "content": "You are a bot."}, {"role": "user", "content": "hi"}],
		model_id="test-model",
		temperature=0,
	)
	assert out == "hello world"


