"""Checks the graded mock-mode behaviour end to end (python check_assistant.py).

1. All 8 documents are embedded and queryable in ChromaDB.
2. classify_intent routes keyword queries to policy_question and others to general_question.
3. The conditional edge sends them to retrieve_and_answer / direct_answer.
4. Retrieval returns chunks from the correct source document.
5. Mock outputs follow the canned templates, and the Pydantic schema is populated.
6. No network call is made while answering: sockets are blocked during the checks.
"""

import os
import socket
import sys

os.environ.pop("MOCK_LLM", None)  # graded default: MOCK_LLM unset -> mock mode

from graph import GENERAL_ANSWER, ask, get_collection, get_graph, run  # noqa: E402
from schemas import AskResponse  # noqa: E402

POLICY_CASES = [
    # (query, document that must supply the top chunk)
    ("How much is the delivery fee on a small order?", "doc_01"),
    ("How long does a refund take to reach my account?", "doc_02"),
    ("What does the Zepto Pass+ membership include?", "doc_03"),
    ("My order tracking shows no movement for 30 minutes, what should I do?", "doc_04"),
    ("Can I cancel an order after it has been packed?", "doc_05"),
    ("Can I combine two gift card balances in one order?", "doc_07"),
    ("What are your support hours, and is there phone support?", "doc_08"),
]
GENERAL_CASES = ["What is the capital of France?", "Tell me a joke about bananas."]


def block_network():
    def refuse(*args, **kwargs):
        raise RuntimeError("Network access attempted during a mock-mode answer")
    socket.socket.connect = refuse
    socket.create_connection = refuse


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    collection = get_collection()   # loads the cached model + local ChromaDB
    get_graph()
    stored = collection.get(include=["metadatas"])
    docs_indexed = sorted({m["doc_id"] for m in stored["metadatas"]})
    print(f"[1] ChromaDB collection '{collection.name}': {collection.count()} chunks "
          f"from {len(docs_indexed)} documents {docs_indexed}")
    assert len(docs_indexed) == 8

    block_network()
    print("[6] Network blocked for the rest of the checks (any connection attempt would fail)\n")

    failures = 0
    for query, expected_doc in POLICY_CASES:
        state = run(query)
        top = state["retrieved"][0]
        ok = (state["intent"] == "policy_question"
              and top["doc_id"] == expected_doc
              and state["answer"].startswith("Based on the retrieved context: ")
              and state["sources"] == [c["id"] for c in state["retrieved"]]
              and len(state["sources"]) == 3)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  policy  | {query}")
        print(f"      intent={state['intent']} -> retrieve_and_answer | top chunk {top['id']} "
              f"({top['title']}, cosine {top['similarity']}) expected {expected_doc}")
        print(f"      sources={state['sources']} confidence={state['confidence']}")

    for query in GENERAL_CASES:
        state = run(query)
        ok = (state["intent"] == "general_question" and state["answer"] == GENERAL_ANSWER
              and state["sources"] == [] and "retrieved" not in state)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  general | {query}")
        print(f"      intent={state['intent']} -> direct_answer | answer={state['answer']!r}")

    response = ask(POLICY_CASES[0][0])
    AskResponse.model_validate(response.model_dump())   # schema round-trip
    print(f"\n[5] Validated API response: {response.model_dump_json()}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
