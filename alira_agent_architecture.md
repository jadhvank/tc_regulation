## ALiRA Agent Architecture — Analysis and Simplification Guide

### 1) Overview and design rationale

- **Goal**: Provide a modular “agent platform” that can run domain-specific apps (documentation, chatbot, model analysis) on top of LLMs with optional RAG and tracing.
- **Layering**
  - **model**: Abstracts LLM providers behind a consistent interface (LiteLLM), handles model lifecycle and inference params.
  - **agent**: Encapsulates task logic and orchestration per capability (e.g., chat, ReAct, stateflow description). Uses `AgentState` and optional tools/RAG.
  - **app**: Product-level workflows composed of multiple agents and post-processing; manages instances, knowledge, and exposes process endpoints.
- **Why split?**
  - **Separation of concerns**: inference vs orchestration vs product flow.
  - **Replaceability**: swap providers (LiteLLM/OpenAI/vLLM), swap orchestrator (custom → LangGraph), swap app workflows with minimal cross-impact.
  - **Traceability**: App-level and Agent-level process methods are wrapped with AsyncMLflowTracker for MLflow async tracking to capture runs and metrics.


### 2) Layer responsibilities and boundaries

- **model layer (`agent/src/model`)**
  - Manages model instances, loading, status, and invocation via LiteLLM.
  - Normalizes inference params per backend (`inference_config.py` with param filtering/defaults).
  - Exposed via service routes under `/api/v1/models`: `/load`, `/status`, `/chat`.

- **agent layer (`agent/src/agent`)**
  - Base `Agent` and `AgentState` define common state schema and lifecycle.
  - Concrete agents implement `process(state)`; examples:
    - `general/chat_agent.py`: RAG-enabled chat (search, compose prompt, call LLM).
    - `general/react_agent.py`: tool-augmented iterative calls (ReAct style loop).
    - Documentation/stateflow agents under `documentation/requirement_generation/*`.
  - Agents call into model layer (`model_manager.complete_chat/…`) and optional RAG search.

- **app layer (`agent/src/app`)**
  - Base `App` adds app metadata, instance id, knowledge/history, and method tracing (MLflow) for `process` / `process_v2`.
  - Concrete apps orchestrate multiple agents for a productized flow, e.g.:
    - `model_description/*`: parse model JSON/CSV, build graph, delegate to specialized agents for blocks/stateflow/MATLAB functions, post-process, format output.
    - `arch_doc/arch_doc.py`: multi-agent flow (router, component analyzer, summarizer) to produce architecture docs.
    - `chatbot/chatbot.py`: wraps chat flow using `ChatAgent` with history and RAG.
  - `app_manager` registers apps and manages their instances; `app_service` exposes `/api/v1/apps/init`, `/api/v1/apps/process`, `/api/v1/apps/add_knowledge`.


### 3) Runtime flow (end-to-end)

1. Client calls `/api/v1/models/load` to create a model instance (maps high-level ids like `ALiRA` → provider id and base URL; sets API key/base).
2. Client calls `/api/v1/apps/init` with `model_instance_id` and target `app_id` to create an app instance.
3. Client optionally adds knowledge (documents/index id) to the app instance.
4. Client calls `/api/v1/apps/process` with query/payload and optional `extra_params`/`inference_params`.
5. App orchestrates: builds internal inputs → calls agents (possibly in sequence) → agents perform RAG/tooling → call model manager to get LLM outputs.
6. App post-processes outputs (e.g., summaries, formatting) and returns normalized response(s).
7. MLflow async tracker wraps `process` to record runs/metadata; state/history optionally updated.


### 4) RAG / Knowledge pipeline

- Two paths appear in codebase:
  - **External core service** (`core/knowledge_manager.py`): asynchronous client to a Core server that builds and serves indices (returns `index_id`). Apps/agents query by `index_id` and `k`.
  - **Local embeddings** (`agent/general/embedding.py`): on-demand HF embeddings via LangChain for local vectorization (e.g., for simple search).
- Agents performing RAG (e.g., `ChatAgent`) follow:
  - Read user query and optional `index_id`/`k` from `AgentState`.
  - Retrieve relevant documents → compose prompt with context → call model → parse and attach sources into response.
- This design allows swapping between managed Core indices and local vector stores with minimal changes to agent code (treat retrieval as a service/API).
  - Core index build endpoint: `POST {CORE_SERVER_URL}/api/documents/build?alias=<alias>` returns `task_id` and `index_id`; the resulting `index_id` is stored and propagated through app/agent state to enable retrieval.


### 5) API surface (selected)

- Model
  - `POST /api/v1/models/load`: resolve `model_id` (e.g., map `ALiRA` → hosted vLLM), set API base/key, create instance.
  - `GET /api/v1/models/status?model_instance_id=`: return load/progress info and readiness.
  - `POST /api/v1/models/chat`: temporary chat completion using an existing instance.
- App
  - `POST /api/v1/apps/init`: create app instance with `model_instance_id` and `app_id`.
  - `POST /api/v1/apps/process`: run end-to-end processing; returns normalized list of `{response, meta_data, model_id}` or app-specific schema.
  - `POST /api/v1/apps/add_knowledge`: attach knowledge/index to an app instance for subsequent RAG.
  - Note: additional experimental routers exist (`*_v2`) and may expose alternative or evolving endpoints.


### 6) Key modules map

- `model/model_manager.py`: instance table, LiteLLM integration (`generate_text`, `complete_chat`), parameter filtering/defaults.
- `model/inference_config.py`: provider/model param defaults and whitelist/blacklist.
- `service/model_service.py`: model load/status/chat APIs (maps friendly ids; uses env like `SLM_SERVER_URL`).
- `agent/agent.py`: `AgentState` schema and abstract `Agent` base.
- `agent/general/chat_agent.py`: retrieval + prompt composition + chat completion; returns with sources.
- `agent/general/react_agent.py`: tool call loop with `tool_choice=auto` and tool schemas.
- `agent/documentation/.../stateflow/*`: specialized stateflow description/generation; uses templates and chained agents.
- `app/app.py`: `App` base, MLflow async tracing, knowledge/codebase lists, instance-scoped data path and history.
- `app/app_manager.py`: app registry and instance lifecycle; `register_app` decorator.
- `app/.../model_description/*`: model graph building, filtering target nodes, calling agents for block/stateflow/matlab; post-processing and schema build.
- `app/chatbot/chatbot.py`: wraps chat agent and history; normalizes response.
- `core/knowledge_manager.py`: Core server client to build indices and manage `knowledge_table`.
- `state/state_manager.py`: in-memory state/metadata for processes; commonly used by services/managers.
- `mlflow_async/*`, `mlflow_trace/*`: async tracing utilities used to decorate/trace app methods.
- `service/app_service.py`: `/app/init`, `/app/process`, `/app/add_knowledge`; MCP session wrapper; formatting responses.
- `main.py`: server startup/shutdown, env loading, restoration of app instances from DB.


### 7) State, tracing, logging

- **State**: `AgentState` dict passed through agents; `App` keeps history per instance; `state_manager` centralizes some process/global status.
- **Tracing**: `App.__init_subclass__` wraps `process` / `process_v2` with `AsyncMLflowTracker.trace(...)` to emit runs without changing app code. `Agent.__init_subclass__` also wraps `process` for MLflow tracing at the agent level.
- **Logging**: `logger` module and structured events in services/agents; errors surfaced via `exception_handler` decorators in FastAPI routes.


### 8) Simplified implementation blueprint (LangGraph + LiteLLM + CSV/Folder ingestion)

- **Objective**: A lighter, composable agent supporting:
  - Input: CSV file(s) or a folder with various files.
  - Steps: ingest → index → retrieve → generate → write outputs.
  - Infra: LangGraph for orchestration, LiteLLM for LLM calls, pluggable RAG (local or external), FastAPI for API, Streamlit (or similar) for UI.

- **Component mapping**
  - ALiRA `model_manager` → LiteLLM thin wrapper.
  - ALiRA `agent/*` orchestration → LangGraph nodes and edges.
  - ALiRA `core/knowledge_manager` → RAG adapter interface; choose external Core or local vector DB (Chroma/Qdrant/PGVector).
  - ALiRA apps → a single “CSV/Folder Analysis App” graph with configurable tools.

- **LangGraph state and nodes (example)**

```python
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from litellm import acompletion

class PipelineState(TypedDict, total=False):
    inputs: Dict[str, Any]           # paths, user options
    chunks: List[Dict[str, Any]]     # {id, text, metadata}
    index_id: str
    query: str
    retrieved: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    outputs: Dict[str, Any]

async def ingest_node(state: PipelineState) -> PipelineState:
    # Load CSV or walk folder, extract text and metadata
    chunks = await extract_chunks(state["inputs"])
    return {**state, "chunks": chunks}

async def index_node(state: PipelineState) -> PipelineState:
    # Use local vector DB or call Core to build index; return index_id
    index_id = await rag_adapter.build_index(state["chunks"])
    return {**state, "index_id": index_id}

async def retrieve_node(state: PipelineState) -> PipelineState:
    docs = await rag_adapter.search(state["query"], index_id=state["index_id"], k=state["inputs"].get("k", 5))
    return {**state, "retrieved": docs}

async def generate_node(state: PipelineState) -> PipelineState:
    context = "\n".join([d.get("text","") for d in state.get("retrieved", [])])
    messages = [
        {"role": "system", "content": state["inputs"].get("system_prompt","You are a helpful assistant.")},
        {"role": "user", "content": f"{context}\n\nQuestion:\n{state['query']}"},
    ]
    resp = await acompletion(model=state["inputs"]["model_id"], messages=messages, stream=False)
    answer = resp.choices[0].message.content
    return {**state, "messages": messages+[{"role":"assistant","content":answer}]}

async def write_node(state: PipelineState) -> PipelineState:
    outputs = await write_outputs(state)
    return {**state, "outputs": outputs}

graph = StateGraph(PipelineState)
graph.add_node("ingest", ingest_node)
graph.add_node("index", index_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("write", write_node)
graph.set_entry_point("ingest")
graph.add_edge("ingest", "index")
graph.add_edge("index", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "write")
graph.add_edge("write", END)
app = graph.compile()
```

- **RAG adapter interface**

```python
class RAGAdapter:
    async def build_index(self, chunks: list[dict]) -> str: ...
    async def search(self, query: str, index_id: str, k: int = 5) -> list[dict]: ...

class CoreRAG(RAGAdapter):
    ... # HTTP to Core service

class LocalRAG(RAGAdapter):
    ... # Chroma/Qdrant embeddings + metadata store
```

- **API sketch (FastAPI)**

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/ingest")
async def ingest(files: list[UploadFile] | None = None, folder_zip: UploadFile | None = None):
    # Extract chunks and return a temporary index_id or session_id
    ...

@app.post("/process")
async def process(payload: dict):
    # Run the compiled graph with inputs: model_id, system_prompt, k, query, output options
    ...

@app.get("/result/{session_id}")
async def result(session_id: str):
    ...
```

- **UI wiring (Streamlit or similar)**
  - Upload CSV/folder → call `/ingest` → show doc counts.
  - Enter question(s)/tasks → call `/process` and stream tokens (optional).
  - Download generated files (e.g., summaries, regenerated CSVs/docs).

- **Data contracts (example)**
  - Request `/process`
    - `model_id`: string (LiteLLM model alias)
    - `query`: string or list of tasks
    - `k`: int (top-k)
    - `system_prompt`: optional
    - `rag_backend`: `"core"` | `"local"`
    - `output`: `{ format: 'md|csv|json', path: '/desired/output/dir' }`
  - Response
    - `{ answer, sources: [{source, section, chunk_index}], files: [paths], session_id }`


### 9) MVP checklist and risks

- **Checklist**
  - LiteLLM configured with API keys and timeouts.
  - RAG adapter implemented; start with LocalRAG (Chroma/Qdrant) and simple CSV/folder ingestor.
  - LangGraph pipeline compiled; nodes isolated and testable.
  - FastAPI endpoints for ingest/process; optional streaming endpoint (`/process/stream` with SSE or websockets).
  - Minimal Streamlit page: upload, ask, view, download.
  - Observability: basic logs + timing; add tracing later as needed.

- **Risks/Trade-offs**
  - Provider diffs (token limits, function calling) → normalize via LiteLLM params.
  - Large folders: ingestion/chunking memory; consider async IO and backpressure.
  - Index/cache invalidation when inputs change; include content hashes in index keys.
  - Output consistency: define stable schemas for app responses and file manifests.


---

This document aligns current ALiRA structure with a slimmer, LangGraph-based blueprint suitable for CSV/folder analysis + RAG + LiteLLM inference, exposing clean APIs and a simple UI.***

