from typing import Any, Dict, List, Optional
from litellm import acompletion

from src.config.settings import get_settings
from src.model.hchat_client import complete_chat as hchat_complete_chat


def _filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
	allowed = {"temperature", "max_tokens", "top_p", "stop", "presence_penalty", "frequency_penalty", "timeout"}
	return {k: v for k, v in params.items() if k in allowed and v is not None}


async def complete_chat(messages: List[Dict[str, str]], model_id: Optional[str] = None, **params: Any) -> str:
	"""
	Thin wrapper over litellm.acompletion for chat models.
	Returns assistant content string.
	"""
	settings = get_settings()
	if settings.HCHAT_ENABLED:
		return await hchat_complete_chat(messages, model_id=model_id, **_filter_params(params))
	model = model_id or settings.LLM_MODEL_ID
	resp = await acompletion(model=model, messages=messages, **_filter_params(params))
	# litellm returns OpenAI-like response object
	return resp.choices[0].message.content  # type: ignore[attr-defined]


