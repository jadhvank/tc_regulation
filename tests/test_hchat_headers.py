import os
import json
import pytest
import httpx
from typing import Any

from src.config.settings import get_settings
from src.model import hchat_client


def _build_mock_response_asserting_headers(expected: dict[str, Any]):
	def handler(request: httpx.Request) -> httpx.Response:
		# Verify endpoint path
		assert request.url.path.endswith("/claude/messages")
		# Verify headers per expectation (presence and exact value if provided)
		for k, v in expected.items():
			assert k in request.headers
			if v is not None:
				assert request.headers[k] == v
		# Also ensure model is forwarded
		body = json.loads(request.content.decode())
		assert "model" in body
		return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
	return handler


@pytest.mark.asyncio
async def test_headers_default_sends_both_bearer_and_api_key(monkeypatch: Any):
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.delenv("HCHAT_AUTH_STYLE", raising=False)
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	handler = _build_mock_response_asserting_headers({
		"Authorization": "Bearer test-key",
		"api-key": "test-key",
	})
	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[{"role": "user", "content": "hi"}],
			client=client,
		)
		assert text == "ok"


@pytest.mark.asyncio
async def test_headers_style_bearer_only(monkeypatch: Any):
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("HCHAT_AUTH_STYLE", "bearer")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.headers.get("Authorization") == "Bearer test-key"
		assert "api-key" not in request.headers
		return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[{"role": "user", "content": "hi"}],
			client=client,
		)
	assert text == "ok"


@pytest.mark.asyncio
async def test_headers_style_api_key_only(monkeypatch: Any):
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("HCHAT_AUTH_STYLE", "api-key")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.headers.get("api-key") == "test-key"
		auth = request.headers.get("Authorization")
		assert not auth or auth == ""
		return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[{"role": "user", "content": "hi"}],
			client=client,
		)
	assert text == "ok"


@pytest.mark.asyncio
async def test_headers_style_raw_authorization(monkeypatch: Any):
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("HCHAT_AUTH_STYLE", "raw-authorization")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.headers.get("Authorization") == "test-key"
		assert "api-key" not in request.headers
		return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[{"role": "user", "content": "hi"}],
			client=client,
		)
	assert text == "ok"


@pytest.mark.asyncio
async def test_model_id_forwarding_override(monkeypatch: Any):
	monkeypatch.setenv("HCHAT_ENABLED", "true")
	monkeypatch.setenv("HCHAT_BASE_URL", "https://example.com/v2/api")
	monkeypatch.setenv("HCHAT_API_KEY", "test-key")
	monkeypatch.setenv("LLM_MODEL_ID", "claude-3-5-sonnet-v2")
	get_settings.cache_clear()

	def handler(request: httpx.Request) -> httpx.Response:
		body = json.loads(request.content.decode())
		assert body["model"] == "claude-haiku-4-5"
		return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

	transport = httpx.MockTransport(handler)
	async with httpx.AsyncClient(transport=transport) as client:
		text = await hchat_client.complete_chat(
			messages=[{"role": "user", "content": "hi"}],
			model_id="claude-haiku-4-5",
			client=client,
		)
	assert text == "ok"


