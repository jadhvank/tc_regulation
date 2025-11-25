import os
from importlib import reload

from src.config import settings as settings_mod


def test_settings_defaults(tmp_path, monkeypatch):
	monkeypatch.delenv("LLM_MODEL_ID", raising=False)
	reload(settings_mod)
	s = settings_mod.get_settings()
	assert s.LLM_MODEL_ID == "gpt-4o-mini"
	assert s.LOG_DIR == "./logs"
	assert s.OUTPUT_DIR == "./outputs"
	assert s.DATA_DIR == "./data"


def test_settings_env_override(monkeypatch):
	monkeypatch.setenv("LLM_MODEL_ID", "gpt-4o")
	reload(settings_mod)
	s = settings_mod.get_settings()
	assert s.LLM_MODEL_ID == "gpt-4o"


