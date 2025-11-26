from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path


class Settings(BaseSettings):
	OPENAI_API_KEY: str | None = Field(default=None)
	OPENAI_BASE: str | None = Field(default=None)
	ANTHROPIC_API_KEY: str | None = Field(default=None)

	# H Chat (Claude) integration
	HCHAT_ENABLED: bool = Field(default=False)
	HCHAT_BASE_URL: str | None = Field(default=None)  # e.g. https://h-chat-api.autoever.com/v2/api
	HCHAT_API_KEY: str | None = Field(default=None)
	HCHAT_PROVIDER: str = Field(default="claude")
	HCHAT_AUTH_STYLE: str | None = Field(default=None)  # bearer | api-key | raw-authorization | None -> both

	LLM_MODEL_ID: str = Field(default="gpt-4o-mini")

	LOG_DIR: str = Field(default="./logs")
	OUTPUT_DIR: str = Field(default="./outputs")
	DATA_DIR: str = Field(default="./data")
	CHROMA_DB_DIR: str = Field(default="./data/indices/chroma")
	SQLITE_DB_PATH: str = Field(default="./data/indices/sqlite/app.db")
	HYBRID_SEARCH_ENABLED: bool = Field(default=True)
	SQL_AGENT_ENABLED: bool = Field(default=True)
	SQL_MAX_ROWS: int = Field(default=200)
	DB_CONTEXT_ENABLED: bool = Field(default=True)
	DB_CONTEXT_MAX_TOKENS: int = Field(default=512)
	CORS_ORIGINS: str = Field(default="*")  # comma-separated or '*'

	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

	def ensure_runtime_dirs(self) -> None:
		"""
		Ensure that runtime directories exist at startup.
		"""
		for path in [self.LOG_DIR, self.OUTPUT_DIR, self.DATA_DIR, self.CHROMA_DB_DIR]:
			Path(path).mkdir(parents=True, exist_ok=True)
		# ensure sqlite directory exists
		try:
			sqlite_parent = Path(self.SQLITE_DB_PATH).parent
			sqlite_parent.mkdir(parents=True, exist_ok=True)
		except Exception:
			# best-effort; don't crash app startup
			pass


@lru_cache
def get_settings() -> Settings:
	"""
	Returns a cached Settings instance loaded from environment/.env.
	Also ensures runtime directories are present.
	"""
	settings = Settings()  # type: ignore[call-arg]
	settings.ensure_runtime_dirs()
	return settings


