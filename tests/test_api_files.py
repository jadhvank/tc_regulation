from fastapi.testclient import TestClient
from pathlib import Path
from importlib import reload
import pytest

from src.server.main import app
from src.config import settings as settings_mod


def test_files_endpoint_serves_output(tmp_path, monkeypatch):
	monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
	reload(settings_mod)

	# Create sample output file
	session_id = "sess-t"
	base = Path(settings_mod.get_settings().OUTPUT_DIR) / session_id
	base.mkdir(parents=True, exist_ok=True)
	target = base / "x.md"
	target.write_text("# hello", encoding="utf-8")

	client = TestClient(app)
	resp = client.get(f"/api/v1/files/{session_id}/x.md")
	assert resp.status_code == 200
	assert "text/markdown" in resp.headers.get("content-type", "") or "text/plain" in resp.headers.get("content-type", "")
	assert resp.text.startswith("# hello")


