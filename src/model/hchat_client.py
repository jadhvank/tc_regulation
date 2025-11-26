from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import json
import httpx

from src.config.settings import get_settings


def _filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
	allowed = {"temperature", "max_tokens", "top_p", "stop", "presence_penalty", "frequency_penalty", "timeout"}
	return {k: v for k, v in params.items() if k in allowed and v is not None}


def _extract_system_and_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
	system_prompt: Optional[str] = None
	remaining: List[Dict[str, str]] = []
	for idx, m in enumerate(messages):
		role = m.get("role", "")
		if idx == 0 and role == "system":
			system_prompt = m.get("content", "") or None
			continue
		if role in ("user", "assistant"):
			remaining.append({"role": role, "content": m.get("content", "")})
		# silently ignore additional/unexpected roles
	return system_prompt, remaining


def _build_headers(api_key: str) -> Dict[str, str]:
	settings = get_settings()
	style = (settings.HCHAT_AUTH_STYLE or "").strip().lower()
	# Default: send both bearer and api-key for maximum compatibility
	if not style:
		return {
			"Authorization": f"Bearer {api_key}",
			"api-key": api_key,
			"Content-Type": "application/json",
		}
	if style == "bearer":
		return {
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
		}
	if style == "api-key":
		return {
			"api-key": api_key,
			"Content-Type": "application/json",
		}
	if style == "raw-authorization":
		return {
			"Authorization": api_key,
			"Content-Type": "application/json",
		}
	# Fallback to both
	return {
		"Authorization": f"Bearer {api_key}",
		"api-key": api_key,
		"Content-Type": "application/json",
	}


async def complete_chat(
	messages: List[Dict[str, str]],
	model_id: Optional[str] = None,
	client: Optional[httpx.AsyncClient] = None,
	**params: Any,
) -> str:
	"""
	Call H Chat Claude messages API (non-streaming) and return assistant text.
	"""
	settings = get_settings()
	if not settings.HCHAT_API_KEY:
		raise ValueError("HCHAT_API_KEY is not configured")
	if not settings.HCHAT_BASE_URL:
		raise ValueError("HCHAT_BASE_URL is not configured")

	model = model_id or settings.LLM_MODEL_ID
	base_url = settings.HCHAT_BASE_URL.rstrip("/")
	url = f"{base_url}/claude/messages"
	system_prompt, other_messages = _extract_system_and_messages(messages)

	payload: Dict[str, Any] = {
		"model": model,
		"messages": other_messages,
		"stream": False,
	}
	if system_prompt:
		payload["system"] = system_prompt
	payload.update(_filter_params(params))

	headers = _build_headers(settings.HCHAT_API_KEY)
	timeout = params.get("timeout", 30.0)

	_close_client = False
	if client is None:
		client = httpx.AsyncClient(timeout=timeout)
		_close_client = True
	try:
		resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
		resp.raise_for_status()
		data = resp.json()
		# Expected shape: {'content': [{'type':'text', 'text': '...'}], ...}
		content = data.get("content")
		if isinstance(content, list) and content:
			first = content[0] or {}
			text = first.get("text")
			if isinstance(text, str):
				return text
		# Fallbacks
		if isinstance(content, str):
			return content
		raise ValueError("Unexpected response shape from H Chat API")
	finally:
		if _close_client:
			await client.aclose()


async def stream_chat(
	messages: List[Dict[str, str]],
	model_id: Optional[str] = None,
	client: Optional[httpx.AsyncClient] = None,
	**params: Any,
) -> AsyncGenerator[str, None]:
	"""
	Stream text chunks from H Chat Claude messages API.
	Parses SSE-like 'data:' lines, emitting text deltas when present.
	"""
	settings = get_settings()
	if not settings.HCHAT_API_KEY:
		raise ValueError("HCHAT_API_KEY is not configured")
	if not settings.HCHAT_BASE_URL:
		raise ValueError("HCHAT_BASE_URL is not configured")

	model = model_id or settings.LLM_MODEL_ID
	base_url = settings.HCHAT_BASE_URL.rstrip("/")
	url = f"{base_url}/claude/messages"
	system_prompt, other_messages = _extract_system_and_messages(messages)

	payload: Dict[str, Any] = {
		"model": model,
		"messages": other_messages,
		"stream": True,
	}
	if system_prompt:
		payload["system"] = system_prompt
	payload.update(_filter_params(params))

	headers = _build_headers(settings.HCHAT_API_KEY)
	timeout = params.get("timeout", 30.0)

	_close_client = False
	if client is None:
		client = httpx.AsyncClient(timeout=timeout)
		_close_client = True
	try:
		async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as r:
			r.raise_for_status()
			async for line in r.aiter_lines():
				if not line:
					continue
				# Expecting SSE format: "data: {...}"
				if line.startswith("data:"):
					raw = line[len("data:"):].strip()
					if raw in ("[DONE]", ""):
						continue
					try:
						event = json.loads(raw)
					except Exception:
						continue
					# Try Anthropic-like stream events
					# content_block_delta -> delta.text
					delta = None
					if isinstance(event, dict):
						if event.get("type") == "content_block_delta":
							inner = event.get("delta") or {}
							delta = inner.get("text")
						elif event.get("type") == "message_delta":
							# sometimes full delta text is present
							delta = event.get("text")
						elif "content" in event and isinstance(event["content"], list):
							first = event["content"][0] or {}
							delta = first.get("text")
					if isinstance(delta, str) and delta:
						yield delta
	finally:
		if _close_client:
			await client.aclose()


