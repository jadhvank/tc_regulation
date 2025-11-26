from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from typing import List, Optional
from uuid import uuid4
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from starlette.status import HTTP_404_NOT_FOUND

from src.config.settings import get_settings
from src.graphs.chat_graph import build_chat_graph
from src.graphs.csv_graph import build_csv_graph
from src.schemas.api import ChatProcessRequest, ChatProcessResponse, CSVIngestResponse, ChatIngestResponse, CSVProcessRequest, CSVProcessResponse
from src.utils.logging import get_logger
from src.ingestion.csv_ingestor import csv_to_chunks
from src.ingestion.fs_ingestor import unzip_to_folder, folder_to_chunks
from src.rag.local import LocalRAG
from src.ingestion.sql_store import store_chunks
from src.ingestion.analyze import analyze_and_store_schema
from src.agents.db_context import refresh_session_profile
from src.history.store import create_chat, list_chats as db_list_chats, list_messages as db_list_messages, append_message as db_append_message, get_chat as db_get_chat, update_chat_session as db_update_chat_session
from src.config.secure_store import get_secret as get_app_secret, set_secret as set_app_secret, is_set as is_secret_set
import os
from functools import lru_cache

app = FastAPI(title="Agent Server (MVP)", version="0.1.0")
logger = get_logger(__name__)
settings = get_settings()

origins = ["*"] if settings.CORS_ORIGINS.strip() == "*" else [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health():
	return {"status": "ok"}


chat_app = build_chat_graph()
csv_app = build_csv_graph()


@router.post("/apps/chat/process", response_model=ChatProcessResponse)
async def process_chat(req: ChatProcessRequest):
	# Resolve or create chat
	chat_id = req.chat_id
	chat_row = None
	if chat_id:
		chat_row = db_get_chat(chat_id)
	if not chat_row:
		created = create_chat(session_id=req.session_id or None, title=None)
		chat_id = created["chat_id"]
		chat_row = created
	# If session_id provided and chat lacks one, persist it
	if req.session_id and not (chat_row or {}).get("session_id"):
		db_update_chat_session(chat_id, req.session_id)
		chat_row = db_get_chat(chat_id) or chat_row
	# Load history messages (without system)
	history = [{"role": m["role"], "content": m["content"]} for m in db_list_messages(chat_id=chat_id, limit=100)]
	# Build state
	state = {
		"query": req.query,
		"system_prompt": req.system_prompt or "You are a helpful assistant.",
		"model_id": req.model_id or settings.LLM_MODEL_ID,
		"session_id": (req.session_id or (chat_row or {}).get("session_id")) or None,
		"k": req.k or 5,
		"retrieval_mode": req.retrieval_mode or None,
		"history_messages": history,
	}
	logger.info({"event": "chat_process_start", "model_id": state["model_id"]})
	# Persist incoming user message
	try:
		db_append_message(chat_id=chat_id, role="user", content=req.query)
	except Exception:
		logger.exception("chat_history_append_user_failed")
	result = await chat_app.ainvoke(state)
	logger.info({"event": "chat_process_end"})
	# Persist assistant message
	try:
		db_append_message(chat_id=chat_id, role="assistant", content=result.get("answer", ""))
	except Exception:
		logger.exception("chat_history_append_assistant_failed")
	# Build sources from both hybrid docs and SQL summary (if any)
	sources = [{"source": d.get("metadata", {}).get("file"), "text": d.get("text")} for d in result.get("retrieved", [])]
	if result.get("sql_summary"):
		sources.append({"source": "sql", "text": result.get("sql_summary")})
	if result.get("sql_answer_text"):
		sources.append({"source": "sql_answer", "text": result.get("sql_answer_text")})
	if result.get("stats_summary"):
		sources.append({"source": "stats", "text": result.get("stats_summary")})
	if result.get("columns_summary"):
		sources.append({"source": "columns", "text": result.get("columns_summary")})
	return ChatProcessResponse(
		answer=result.get("answer", ""),
		model_id=state["model_id"],
		sources=sources or None,
		meta={"tokens": None, "intent_mode": result.get("intent_mode")},
		chat_id=chat_id,
	)


@router.post("/apps/chat/ingest", response_model=ChatIngestResponse)
async def ingest_chat(files: Optional[List[UploadFile]] = File(default=None), folder_zip: Optional[UploadFile] = File(default=None)):
	try:
		session_id = str(uuid4())
		upload_dir = Path(settings.DATA_DIR) / "uploads" / session_id / "chat"
		upload_dir.mkdir(parents=True, exist_ok=True)

		chunks = []
		csv_paths: List[Path] = []
		if files:
			for f in files:
				dest = upload_dir / f.filename
				content = await f.read()
				dest.write_bytes(content)
				if dest.suffix.lower() == ".csv":
					chunks.extend(csv_to_chunks(dest))
					csv_paths.append(dest)
				elif dest.suffix.lower() in {".txt", ".md"}:
					chunks.append({"text": dest.read_text(encoding="utf-8", errors="ignore"), "metadata": {"file": dest.name}})
		if folder_zip:
			zip_dest = upload_dir / folder_zip.filename
			zip_dest.write_bytes(await folder_zip.read())
			folder = unzip_to_folder(zip_dest, upload_dir / "unzipped")
			chunks.extend(folder_to_chunks(folder))
			# collect csv paths for schema analysis
			for p in folder.rglob("*.csv"):
				csv_paths.append(p)

		# write to SQLite (rows + FTS)
		if chunks:
			store_chunks(session_id=session_id, chunks=chunks)
		# analyze CSV schema and store
		for p in csv_paths:
			try:
				analyze_and_store_schema(session_id=session_id, file_path=p)
			except Exception:
				logger.exception("schema_analysis_failed")
		# refresh session DB profile
		try:
			refresh_session_profile(session_id=session_id)
		except Exception:
			logger.exception("db_context_refresh_failed")
		rag = LocalRAG()
		if chunks:
			await rag.build_index(session_id=session_id, chunks=chunks)
		return ChatIngestResponse(session_id=session_id, doc_count=len(chunks))
	except Exception as e:
		logger.exception("chat_ingest_failed")
		return JSONResponse({"detail": f"ingest failed: {e.__class__.__name__}: {e}"}, status_code=500)


@router.post("/apps/csv/ingest", response_model=CSVIngestResponse)
async def ingest_csv(files: Optional[List[UploadFile]] = File(default=None), folder_zip: Optional[UploadFile] = File(default=None)):
	try:
		session_id = str(uuid4())
		upload_dir = Path(settings.DATA_DIR) / "uploads" / session_id
		upload_dir.mkdir(parents=True, exist_ok=True)

		chunks = []
		csv_paths: List[Path] = []
		if files:
			for f in files:
				dest = upload_dir / f.filename
				content = await f.read()
				dest.write_bytes(content)
				if dest.suffix.lower() == ".csv":
					chunks.extend(csv_to_chunks(dest))
					csv_paths.append(dest)
				elif dest.suffix.lower() in {".txt", ".md"}:
					chunks.append({"text": dest.read_text(encoding="utf-8", errors="ignore"), "metadata": {"file": dest.name}})
		if folder_zip:
			zip_dest = upload_dir / folder_zip.filename
			zip_dest.write_bytes(await folder_zip.read())
			folder = unzip_to_folder(zip_dest, upload_dir / "unzipped")
			chunks.extend(folder_to_chunks(folder))
			for p in folder.rglob("*.csv"):
				csv_paths.append(p)

		# write to SQLite (rows + FTS)
		if chunks:
			store_chunks(session_id=session_id, chunks=chunks)
		# analyze CSV schema
		for p in csv_paths:
			try:
				analyze_and_store_schema(session_id=session_id, file_path=p)
			except Exception:
				logger.exception("csv_schema_analysis_failed")
		# refresh session DB profile
		try:
			refresh_session_profile(session_id=session_id)
		except Exception:
			logger.exception("db_context_refresh_failed")
		rag = LocalRAG()
		if chunks:
			await rag.build_index(session_id=session_id, chunks=chunks)
		return CSVIngestResponse(session_id=session_id, doc_count=len(chunks))
	except Exception as e:
		logger.exception("csv_ingest_failed")
		return JSONResponse({"detail": f"ingest failed: {e.__class__.__name__}: {e}"}, status_code=500)


@router.post("/apps/csv/process", response_model=CSVProcessResponse)
async def process_csv(req: CSVProcessRequest):
	state = {
		"session_id": req.session_id,
		"query": req.query,
		"k": req.k or 5,
		"model_id": req.model_id or settings.LLM_MODEL_ID,
	}
	result = await csv_app.ainvoke(state)
	files = result.get("output_paths", [])
	# Build file URLs relative to API base
	base = (Path(settings.OUTPUT_DIR) / state["session_id"]).resolve()
	file_urls = []
	for p in files:
		try:
			rel = Path(p).resolve().relative_to(base).as_posix()
			file_urls.append(f"/api/v1/files/{state['session_id']}/{rel}")
		except Exception:
			continue
	# Prepare sources
	sources = []
	for d in result.get("retrieved", []):
		src = d.get("metadata", {}).get("file", None)
		sources.append({"source": src, "text": d.get("text", None)})
	return CSVProcessResponse(
		answer=result.get("answer", ""),
		model_id=state["model_id"],
		files=files,
		file_urls=file_urls or None,
		sources=sources or None,
	)


@router.get("/files/{session_id}/{filepath:path}")
async def get_file(session_id: str, filepath: str):
	base = Path(settings.OUTPUT_DIR).resolve() / session_id
	target = (base / filepath).resolve()
	# prevent path traversal
	if not str(target).startswith(str(base)):
		return JSONResponse({"detail": "Not found"}, status_code=HTTP_404_NOT_FOUND)
	if not target.exists() or not target.is_file():
		return JSONResponse({"detail": "Not found"}, status_code=HTTP_404_NOT_FOUND)
	return FileResponse(str(target))


# ----- Chat history endpoints -----
from src.schemas.api import ChatCreateRequest, ChatCreateResponse, ChatListResponse, ChatListItem, ChatMessagesResponse, ChatMessage
from src.schemas.api import ConfigGetResponse, ConfigUpdateRequest, ConfigUpdateResponse


@router.post("/chats", response_model=ChatCreateResponse)
async def create_chat_api(req: ChatCreateRequest):
	row = create_chat(session_id=req.session_id or None, title=req.title or None)
	return ChatCreateResponse(**row)


@router.get("/chats", response_model=ChatListResponse)
async def list_chats_api():
	items = [ChatListItem(**r) for r in db_list_chats(limit=200)]
	return ChatListResponse(chats=items)


@router.get("/chats/{chat_id}/messages", response_model=ChatMessagesResponse)
async def get_chat_messages_api(chat_id: str, limit: int = 100):
	rows = db_list_messages(chat_id=chat_id, limit=limit)
	msgs = [ChatMessage(role=r["role"], content=r["content"], created_at=r["created_at"]) for r in rows]
	return ChatMessagesResponse(chat_id=chat_id, messages=msgs)


# ----- Config endpoints -----
@router.get("/config", response_model=ConfigGetResponse)
async def get_config_api():
	cur = get_settings()
	return ConfigGetResponse(
		llm_model_id=cur.LLM_MODEL_ID,
		openai_key_set=is_secret_set("OPENAI_API_KEY") or bool(os.getenv("OPENAI_API_KEY")),
		anthropic_key_set=is_secret_set("ANTHROPIC_API_KEY") or bool(os.getenv("ANTHROPIC_API_KEY")),
	)


@router.post("/config", response_model=ConfigUpdateResponse)
async def update_config_api(req: ConfigUpdateRequest):
	# Update model id
	if req.llm_model_id:
		os.environ["LLM_MODEL_ID"] = req.llm_model_id
	# Update secrets (store encrypted; also export to env for runtime)
	if req.openai_api_key:
		set_app_secret("OPENAI_API_KEY", req.openai_api_key)
		os.environ["OPENAI_API_KEY"] = req.openai_api_key
	if req.anthropic_api_key:
		set_app_secret("ANTHROPIC_API_KEY", req.anthropic_api_key)
		os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key
	# Clear cached settings to pick up env overrides
	get_settings.cache_clear()  # type: ignore[attr-defined]
	cur = get_settings()
	return ConfigUpdateResponse(
		llm_model_id=cur.LLM_MODEL_ID,
		openai_key_set=is_secret_set("OPENAI_API_KEY") or bool(os.getenv("OPENAI_API_KEY")),
		anthropic_key_set=is_secret_set("ANTHROPIC_API_KEY") or bool(os.getenv("ANTHROPIC_API_KEY")),
	)


app.include_router(router)


