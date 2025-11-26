import os
import json
import pytest
import httpx
from typing import Any

from src.config.settings import get_settings
from src.model import hchat_client


@pytest.mark.asyncio
async def test_hchat_complete_chat_nonstream(monkeypatch: Any):
	# Configure env
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	# Mock transport for non-stream
	def handler(request: httpx.Request) -> httpx.Response:
		assert request.url.path.endswith("/claude/messages")
		body = json.loads(request.content.decode())
		assert body["model"] == "claude-3-5-sonnet-v2"
		assert body.get("stream") is False
		result = {"content": [{"type": "text", "text": "Hello from Claude"}]}
		return httpx.Response(200, json=result)

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[
				{"role": "system", "content": "You are helpful."},
				{"role": "user", "content": "Hi"},
			],
			model_id=None,
			client=client,
		)
		assert text == "Hello from Claude"


@pytest.mark.asyncio
async def test_hchat_stream_chat_streaming(monkeypatch: Any):
	# Configure env
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	# Prepare SSE-like chunked content
	chunk1 = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}
	chunk2 = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "World"}}
	data_bytes = (
		f"data: {json.dumps(chunk1)}\n\n"
		f"data: {json.dumps(chunk2)}\n\n"
		f"data: [DONE]\n\n"
	).encode()

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.url.path.endswith("/claude/messages")
		body = json.loads(request.content.decode())
		assert body.get("stream") is True
		return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data_bytes)

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		collected = []
		async for part in hchat_client.stream_chat(
			messages=[
				{"role": "system", "content": "You are helpful."},
				{"role": "user", "content": "Hi"},
			],
			model_id=None,
			client=client,
		):
			collected.append(part)
		assert "".join(collected) == "Hello World"


