"""LangGraph flow: classify_intent -> (retrieve_and_answer | direct_answer) -> END.

                     ┌───────────────────┐
     query ────────> │  classify_intent  │
                     └─────────┬─────────┘
          policy_question      │      general_question
              ┌────────────────┴────────────────┐
              v                                 v
   ┌─────────────────────┐           ┌─────────────────────┐
   │ retrieve_and_answer │           │    direct_answer    │
   │ (ChromaDB top-3)    │           │    (no retrieval)   │
   └──────────┬──────────┘           └──────────┬──────────┘
              └──────────────> END <────────────┘

Each node's *generation* step branches on MOCK_LLM (see config.mock_llm_enabled):
  * mock (default, graded): deterministic code, no LLM call, no network call
  * MOCK_LLM=0 (optional): the same step is done by the LLM in llm.py
The routing edge itself never depends on MOCK_LLM, and retrieval always runs for real.
"""

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import SNIPPET_CHARS, TOP_K, mock_llm_enabled
from ingest import build_index, retrieve
from prompts import ANSWER_PROMPT, CLASSIFY_PROMPT, GENERAL_PROMPT, format_context
from schemas import AskResponse

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel",
                   "gift card", "support hours"]
GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."


class AssistantState(TypedDict, total=False):
    query: str
    intent: str             # "policy_question" | "general_question"
    retrieved: list[dict]   # top-k chunks (policy path only)
    answer: str
    sources: list[str]
    confidence: float


@lru_cache(maxsize=1)
def get_collection():
    return build_index()


# ---------------------------------------------------------------- nodes

def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    if mock_llm_enabled():
        # Mock (graded): keyword heuristic, no LLM call.
        q = query.lower()
        intent = "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0: let the LLM classify.
        from llm import chat
        label = chat([{"role": "user", "content": CLASSIFY_PROMPT.format(question=query)}])
        intent = "policy_question" if "policy_question" in label else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    # Retrieval always runs for real: local MiniLM embedding + ChromaDB cosine search.
    chunks = retrieve(get_collection(), state["query"], k=TOP_K)
    top = chunks[0]

    if mock_llm_enabled():
        # Mock (graded): canned template built from the single most similar chunk.
        snippet = top["text"]
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        response = AskResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=[c["id"] for c in chunks],
            confidence=round(min(max(top["similarity"], 0.0), 1.0), 3),
        )
    else:
        # Optional MOCK_LLM=0: grounded generation from the structured prompt template.
        from llm import generate_validated
        prompt = ANSWER_PROMPT.format(context=format_context(chunks), question=state["query"])
        response = generate_validated(prompt)

    return {"retrieved": chunks, **response.model_dump()}


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_llm_enabled():
        # Mock (graded): fixed canned string, no LLM call, no retrieval.
        response = AskResponse(answer=GENERAL_ANSWER, sources=[], confidence=1.0)
    else:
        # Optional MOCK_LLM=0: answer directly with the LLM, still without retrieval.
        from llm import generate_validated
        response = generate_validated(GENERAL_PROMPT.format(question=state["query"]))
        response.sources = []
    return response.model_dump()


def route_by_intent(state: AssistantState) -> str:
    return state["intent"]


# ---------------------------------------------------------------- graph

def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run(query: str) -> AssistantState:
    """Run the full graph and return the final state (includes intent and retrieved chunks)."""
    return get_graph().invoke({"query": query})


def ask(query: str) -> AskResponse:
    """Run the graph and return the validated answer/sources/confidence JSON."""
    state = run(query)
    return AskResponse(answer=state["answer"], sources=state.get("sources", []),
                       confidence=state["confidence"])
