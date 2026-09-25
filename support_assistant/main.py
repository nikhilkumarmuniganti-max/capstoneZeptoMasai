"""FastAPI wrapper around the LangGraph support assistant.

Run locally (from the support_assistant folder):
    uvicorn main:app --port 7860
Then:
    curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" \
         -d '{"query": "How much is the delivery fee?"}'
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import mock_llm_enabled
from graph import ask, get_collection, get_graph
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up once at startup: load MiniLM, build/load the ChromaDB index, compile the graph.
    get_collection()
    get_graph()
    yield


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG over Zepto's policy documents (ChromaDB + LangGraph). "
                "Runs a deterministic mock LLM unless MOCK_LLM=0.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": mock_llm_enabled(),
            "indexed_chunks": get_collection().count()}
