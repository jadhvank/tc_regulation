# Agent Server (MVP)

LangGraph + LiteLLM agent server with FastAPI. Provides:
- Chat app (optional LocalRAG retrieval via session_id)
- CSV/Folder analysis app (ingest → index → retrieve → generate → write)

## Prerequisites
- Python 3.11
- uv (recommended) or pip

## Setup (uv)
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
uv pip install -e .[dev]
cp env.example .env  # fill secrets
```

## Run
```bash
uv run uvicorn src.server.main:app --reload
```

## UI (Streamlit)
```bash
# optional: set API_BASE_URL in .streamlit/secrets.toml or env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

## Remote Access
- Start API on 0.0.0.0: `uv run uvicorn src.server.main:app --host 0.0.0.0 --port 8000`
- Start UI on 0.0.0.0: `uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501`
- From a remote machine:
  - Open `http://<SERVER_IP>:8501` (UI)
  - Set UI `API_BASE_URL` to `http://<SERVER_IP>:8000` in `.streamlit/secrets.toml`
- File downloads:
  - The API exposes `GET /api/v1/files/{session_id}/{filepath}` for any file under `OUTPUT_DIR/{session_id}`.
  - CSV process responses include `file_urls` (relative). The UI links to them using your `API_BASE_URL`.

### CORS
- Configure allowed origins via env: `CORS_ORIGINS=*` (default) or `http://localhost:8501,http://<SERVER_IP>:8501`

### Optional public tunneling
- Cloudflare: `cloudflared tunnel --url http://localhost:8501`
- Ngrok: `ngrok http 8501`

## Environment (.env)
See `env.example`.

Key variables:
- OPENAI_API_KEY, OPENAI_BASE
- LLM_MODEL_ID (default: gpt-4o-mini)
- LOG_DIR, OUTPUT_DIR, DATA_DIR, CHROMA_DB_DIR

## API
- GET `/api/v1/health`
- POST `/api/v1/apps/chat/process`
  - body: `{ query, system_prompt?, model_id?, session_id?, k? }`
- POST `/api/v1/apps/chat/ingest`
  - form-data: `files=[UploadFile]*` or `folder_zip`
  - returns `{ session_id, doc_count }`
- POST `/api/v1/apps/csv/ingest`
  - form-data: `files=[UploadFile]*` or `folder_zip`
  - returns `{ session_id, doc_count }`
- POST `/api/v1/apps/csv/process`
  - body: `{ session_id, query, k?, model_id? }`
  - returns `{ answer, files, sources?, model_id }`

## Tests
```bash
uv run pytest -q
```


