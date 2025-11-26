from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatProcessRequest(BaseModel):
	query: str = Field(..., description="User question or message")
	k: Optional[int] = Field(default=None, description="Top-k documents for retrieval (future)")
	system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
	model_id: Optional[str] = Field(default=None, description="Model override")
	session_id: Optional[str] = Field(default=None, description="If provided, use LocalRAG with this session index")
	retrieval_mode: Optional[str] = Field(default=None, description="Override: one of none, sql, hybrid, both")
	chat_id: Optional[str] = Field(default=None, description="Existing chat to append and use history")


class SourceItem(BaseModel):
	source: Optional[str] = None
	section: Optional[str] = None
	chunk_index: Optional[int] = None
	text: Optional[str] = None


class ChatProcessResponse(BaseModel):
	answer: str
	model_id: str
	sources: Optional[List[SourceItem]] = None
	meta: Optional[Dict[str, Any]] = None
	chat_id: Optional[str] = None


class CSVIngestResponse(BaseModel):
	session_id: str
	doc_count: int


class ChatIngestResponse(BaseModel):
	session_id: str
	doc_count: int


class CSVProcessRequest(BaseModel):
	session_id: str
	query: str
	k: Optional[int] = 5
	model_id: Optional[str] = None


class CSVProcessResponse(BaseModel):
	answer: str
	model_id: str
	files: List[str] = []
	sources: Optional[List[SourceItem]] = None
	file_urls: Optional[List[str]] = None


class ChatCreateRequest(BaseModel):
	session_id: Optional[str] = None
	title: Optional[str] = None


class ChatCreateResponse(BaseModel):
	chat_id: str
	session_id: Optional[str] = None
	title: Optional[str] = None
	created_at: str
	updated_at: str


class ChatListItem(BaseModel):
	chat_id: str
	title: Optional[str] = None
	session_id: Optional[str] = None
	updated_at: str


class ChatListResponse(BaseModel):
	chats: List[ChatListItem]


class ChatMessage(BaseModel):
	role: str
	content: str
	created_at: str


class ChatMessagesResponse(BaseModel):
	chat_id: str
	messages: List[ChatMessage]


class ConfigGetResponse(BaseModel):
	llm_model_id: str
	openai_key_set: bool
	anthropic_key_set: bool


class ConfigUpdateRequest(BaseModel):
	llm_model_id: Optional[str] = None
	openai_api_key: Optional[str] = None
	anthropic_api_key: Optional[str] = None


class ConfigUpdateResponse(BaseModel):
	llm_model_id: str
	openai_key_set: bool
	anthropic_key_set: bool


