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
	state = {
		"query": req.query,
		"system_prompt": req.system_prompt or "You are a helpful assistant.",
		"model_id": req.model_id or settings.LLM_MODEL_ID,
		"session_id": req.session_id or None,
		"k": req.k or 5,
	}
	logger.info({"event": "chat_process_start", "model_id": state["model_id"]})
	result = await chat_app.ainvoke(state)
	logger.info({"event": "chat_process_end"})
	return ChatProcessResponse(
		answer=result.get("answer", ""),
		model_id=state["model_id"],
		sources=[{"source": d.get("metadata", {}).get("file"), "text": d.get("text")} for d in result.get("retrieved", [])] or None,
		meta={"tokens": None},
	)


@router.post("/apps/chat/ingest", response_model=ChatIngestResponse)
async def ingest_chat(files: Optional[List[UploadFile]] = File(default=None), folder_zip: Optional[UploadFile] = File(default=None)):
	try:
		session_id = str(uuid4())
		upload_dir = Path(settings.DATA_DIR) / "uploads" / session_id / "chat"
		upload_dir.mkdir(parents=True, exist_ok=True)

		chunks = []
		if files:
			for f in files:
				dest = upload_dir / f.filename
				content = await f.read()
				dest.write_bytes(content)
				if dest.suffix.lower() == ".csv":
					chunks.extend(csv_to_chunks(dest))
				elif dest.suffix.lower() in {".txt", ".md"}:
					chunks.append({"text": dest.read_text(encoding="utf-8", errors="ignore"), "metadata": {"file": dest.name}})
		if folder_zip:
			zip_dest = upload_dir / folder_zip.filename
			zip_dest.write_bytes(await folder_zip.read())
			folder = unzip_to_folder(zip_dest, upload_dir / "unzipped")
			chunks.extend(folder_to_chunks(folder))

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
		if files:
			for f in files:
				dest = upload_dir / f.filename
				content = await f.read()
				dest.write_bytes(content)
				if dest.suffix.lower() == ".csv":
					chunks.extend(csv_to_chunks(dest))
				elif dest.suffix.lower() in {".txt", ".md"}:
					chunks.append({"text": dest.read_text(encoding="utf-8", errors="ignore"), "metadata": {"file": dest.name}})
		if folder_zip:
			zip_dest = upload_dir / folder_zip.filename
			zip_dest.write_bytes(await folder_zip.read())
			folder = unzip_to_folder(zip_dest, upload_dir / "unzipped")
			chunks.extend(folder_to_chunks(folder))

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


app.include_router(router)


