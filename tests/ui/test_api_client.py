import types
import io
import json
import pytest

import ui.api_client as api


class FakeResponse:
	def __init__(self, status_code=200, payload=None):
		self.status_code = status_code
		self._payload = payload or {}

	def raise_for_status(self):
		if not (200 <= self.status_code < 300):
			raise RuntimeError("HTTP error")

	def json(self):
		return self._payload


class FakeClient:
	def __init__(self, *args, **kwargs):
		self.last = {}

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def post(self, path, files=None, json=None):
		self.last = {"path": path, "files": files, "json": json}
		# Return different payloads per endpoint
		if path.endswith("/chat/ingest"):
			return FakeResponse(payload={"session_id": "s1", "doc_count": 2})
		if path.endswith("/chat/process"):
			return FakeResponse(payload={"answer": "A", "model_id": "m"})
		if path.endswith("/csv/ingest"):
			return FakeResponse(payload={"session_id": "s2", "doc_count": 3})
		if path.endswith("/csv/process"):
			return FakeResponse(payload={"answer": "B", "model_id": "m", "files": []})
		return FakeResponse()


def test_chat_ingest_and_process(monkeypatch):
	monkeypatch.setattr(api, "_client", lambda: FakeClient())

	files = [("files", ("a.txt", b"hello", "text/plain"))]
	out = api.chat_ingest(files=files)
	assert out["session_id"] == "s1"

	p = api.chat_process(query="Q", session_id="s1", k=3, system_prompt="S", model_id="m")
	assert p["answer"] == "A"


def test_csv_ingest_and_process(monkeypatch):
	monkeypatch.setattr(api, "_client", lambda: FakeClient())
	files = [("files", ("x.csv", b"a,b\n1,2\n", "text/csv"))]
	out = api.csv_ingest(files=files)
	assert out["session_id"] == "s2"

	p = api.csv_process(session_id="s2", query="Q2", k=5, model_id="m")
	assert p["answer"] == "B"


