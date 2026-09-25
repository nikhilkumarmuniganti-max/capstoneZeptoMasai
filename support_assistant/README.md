# Module 3 — Support Assistant (`/support_assistant`)

A small GenAI service that answers questions about Zepto's policies, grounded in Zepto's own documents.
It uses local **sentence-transformers** embeddings in **ChromaDB**, a **LangGraph** intent router, a
**Pydantic**-enforced JSON answer, and a **FastAPI** `POST /ask` endpoint, packaged with a **Dockerfile**.

**It runs on the deterministic mock LLM.** With `MOCK_LLM` unset (or `MOCK_LLM=1`), no LLM is called at all:
there is no signup, no API key, and no network request to any LLM provider. This default is the graded
baseline, and it is the only mode this submission was run in. The real-LLM branch (`MOCK_LLM=0`) exists in
the code as the brief requires, but it was **not** used or tested.

## How to run

From the repository root, inside the virtual environment described in the [root README](../README.md):

```bash
pip install -r support_assistant/requirements.txt
cd support_assistant

python ingest.py               # optional: (re)build the ChromaDB index; the API also builds it on startup
python check_assistant.py      # runs the acceptance checks in mock mode, with network access blocked
uvicorn main:app --port 7860   # serve the API on http://127.0.0.1:7860  (docs: /docs)
```

The first run downloads `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face once. After that the model is
loaded from the local cache, and everything runs offline.

### Docker

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query": "How much is the delivery fee?"}'
```

What the Dockerfile does:

1. Starts from `python:3.12-slim` and installs `requirements.txt`, using CPU-only PyTorch.
2. Copies the code and the 8 documents.
3. Runs as a non-root user (uid 1000).
4. Runs `python ingest.py` **at build time**, which downloads MiniLM and builds the ChromaDB index into the
   image. The container then starts quickly and needs no network access at run time (`HF_HUB_OFFLINE=1`).
5. Serves the app with `uvicorn main:app --host 0.0.0.0 --port 7860`.
6. Sets `MOCK_LLM=1` explicitly.

> Docker was not installed on the development machine, so `docker build` was not executed there. As a check
> of the install step, `pip install --dry-run --platform manylinux_2_28_x86_64 --python-version 3.12
> --only-binary=:all: -r requirements.txt` confirmed that a pre-built Linux wheel exists for every dependency
> (including `torch-2.9.1+cpu`). The same app code was run with uvicorn, and its recorded responses are below.
> Deploying to Hugging Face Spaces is an optional, ungraded stretch goal and was not attempted.

## Files

| File | Role |
|---|---|
| `docs/doc_01.txt` … `doc_08.txt` | The 8 policy documents, copied verbatim from the brief |
| `config.py` | Paths, model name, `TOP_K = 3`, and the `mock_llm_enabled()` toggle |
| `ingest.py` | **Ingestion + embedding**: load, chunk, embed with MiniLM, store in ChromaDB. Also `retrieve()` |
| `prompts.py` | The structured prompt templates (used only when `MOCK_LLM=0`) |
| `schemas.py` | Pydantic `AskRequest` and `AskResponse` (answer / sources / confidence) |
| `graph.py` | The LangGraph `StateGraph`: 3 nodes plus the conditional edge |
| `llm.py` | Optional Groq client and the retry-on-invalid-JSON logic (`MOCK_LLM=0` only) |
| `main.py` | FastAPI app: `POST /ask` and `GET /health` |
| `check_assistant.py` | Mock-mode acceptance checks (routing, retrieval, templates, schema, no network) |
| `Dockerfile`, `.dockerignore` | Container build |

## Architecture — the RAG pipeline, stage by stage

```
docs/*.txt ──> [1 Ingestion] ──> [2 Embedding] ──> ChromaDB "zepto_policies"
                ingest.py          MiniLM (384-d)    (cosine, 26 chunks)
                                                              │
POST /ask ──> classify_intent ──policy_question──> retrieve_and_answer ──> AskResponse JSON
 (main.py)     (graph.py)    │                    [3 Retrieval] top-3     [4 Generation]
                             │
                             └─general_question──> direct_answer ────────> AskResponse JSON
                                                   (no retrieval)          [4 Generation]
```

1. **Ingestion** (`ingest.py`: `load_documents()` → `chunk_document()`). The 8 files in `docs/` are read and
   split into **sentence-level chunks**, one sentence per chunk, giving 26 chunks.
   - **Why one sentence per chunk:** each policy is only 3–5 sentences long, and each sentence states one
     rule (a fee, a time limit, an exception). With single-sentence chunks, the top match *is* the rule that
     answers the question.
   - **Chunk ids:** `doc_XX#n` (document id plus chunk number). They carry `doc_id`/`title` metadata and are
     what the API returns as `sources`.
2. **Embedding** (`ingest.py`: `embed()` and `build_index()`).
   - **Model:** each chunk is encoded locally with **sentence-transformers `all-MiniLM-L6-v2`** into a
     normalized 384-dimensional vector.
   - **Storage:** the vectors go into the persistent **ChromaDB collection `zepto_policies`** (folder
     `chroma_db/`), created with `hnsw:space = cosine`.
   - **When it runs:** `main.py` builds or loads the index once at startup; the Docker image builds it at
     build time.
3. **Retrieval** (the `retrieve_and_answer` node in `graph.py`, which calls `ingest.retrieve()`). For a
   `policy_question`, the query is embedded with the same model and ChromaDB returns the **top-3 chunks by
   cosine similarity** (reported as `1 − cosine distance`). This stage **always runs for real**, in both
   modes, because it needs no API key and no network.
4. **Generation** (the final step inside `retrieve_and_answer` and `direct_answer`). This is the stage that
   branches on `MOCK_LLM`, and its output is validated as `schemas.AskResponse`.

**Routing.** `classify_intent` is the graph's first node. A conditional edge (`route_by_intent`) sends
`policy_question` to `retrieve_and_answer` and `general_question` to `direct_answer`, and both then go to
`END`. The routing logic itself does not depend on `MOCK_LLM`.

**Graph state.** The state is a `TypedDict` (`AssistantState`: `query`, `intent`, `retrieved`, `answer`,
`sources`, `confidence`). Each node returns the keys it fills.

### What changes with `MOCK_LLM`

| Stage / node | Default: `MOCK_LLM` unset or `1` (graded, used) | Optional: `MOCK_LLM=0` (in code, not used) |
|---|---|---|
| `classify_intent` | Keyword heuristic. If the lower-cased query contains `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card` or `support hours` → `policy_question`, otherwise `general_question` | The LLM classifies using `CLASSIFY_PROMPT` |
| Retrieval | MiniLM + ChromaDB top-3, run for real | Same, run for real |
| `retrieve_and_answer` generation | `"Based on the retrieved context: {snippet}"`, where the snippet is the first ~200 characters of the top chunk. `sources` = ids of the 3 retrieved chunks. `confidence` = cosine similarity of the top chunk (deterministic, in 0–1) | Groq LLM with `ANSWER_PROMPT`, grounded in the 3 chunks |
| `direct_answer` generation | Fixed string: `"I can only answer questions about Zepto policies right now."`, with `sources = []` and `confidence = 1.0` | Groq LLM with `GENERAL_PROMPT`, no retrieval |
| Schema enforcement | `AskResponse` is filled from code, so there is no LLM output that could fail | LLM JSON is validated against `AskResponse`. On failure it retries **up to 2 more times** with a corrective instruction, then returns a response marked `[ERROR]` (`llm.generate_validated`) |

The mock path makes **no network call of any kind**. `check_assistant.py` proves this. After loading the cached model and
the local index, it replaces `socket.connect` with a function that raises an error, then answers every test
query, and every check still passes.

### The structured prompt template

`ANSWER_PROMPT` in [`prompts.py`](prompts.py) is complete text, not a description. It follows the
**role → context → task → format → length** skeleton:

- **Role:** Zepto's customer-support assistant.
- **Context:** the retrieved chunks, each tagged with its chunk id.
- **Task:** answer from the excerpts, quote numbers exactly, and cite the chunk ids used.
- **Format:** return a single JSON object with `answer` / `sources` / `confidence`.
- **Length:** at most 2 sentences, under 60 words.
- **Negative constraints:** *"Do NOT answer using any information that is not present in the provided
  context"* and *"Do NOT guess, invent policies, or rely on general knowledge"*, plus a fixed fallback reply
  when the context does not contain the answer.
- **Few-shot example:** one worked example (a question about a INR 120 order, with its JSON output).

## Example calls (recorded with `MOCK_LLM` at its default)

The server was started with `uvicorn main:app --host 127.0.0.1 --port 7860` and these are the raw responses.

**Policy question — routed to `retrieve_and_answer`.** The query contains "delivery":

```text
POST /ask  {"query": "How much is the delivery fee on a small order?"}
```
```json
{"answer":"Based on the retrieved context: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.","sources":["doc_01#1","doc_05#0","doc_05#2"],"confidence":0.681}
```

The top chunk `doc_01#1` comes from **doc_01 — Delivery Policy**, and it states exactly the fee asked about.

**Policy question — routed to `retrieve_and_answer`.** The query contains "refund":

```text
POST /ask  {"query": "How long does a refund take?"}
```
```json
{"answer":"Based on the retrieved context: Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit.","sources":["doc_02#1","doc_02#0","doc_06#1"],"confidence":0.619}
```

The top chunk comes from **doc_02 — Returns & Refunds**.

**General question — routed to `direct_answer`.** No keyword matches, so there is no retrieval:

```text
POST /ask  {"query": "What is the capital of France?"}
```
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

**Health check, and a rejected request** (the Pydantic request model enforces a non-empty `query`):

```text
GET /health
{"status":"ok","mock_llm":true,"indexed_chunks":26}

POST /ask  {"query": ""}      -> HTTP 422
{"detail":[{"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character","input":"","ctx":{"min_length":1}}]}
```

## Acceptance checks (`python check_assistant.py`)

The script runs with `MOCK_LLM` unset and with sockets blocked. Output:

```text
[1] ChromaDB collection 'zepto_policies': 26 chunks from 8 documents ['doc_01', 'doc_02', 'doc_03', 'doc_04', 'doc_05', 'doc_06', 'doc_07', 'doc_08']
[6] Network blocked for the rest of the checks (any connection attempt would fail)

PASS  policy  | How much is the delivery fee on a small order?
      intent=policy_question -> retrieve_and_answer | top chunk doc_01#1 (Delivery Policy, cosine 0.6807) expected doc_01
PASS  policy  | How long does a refund take to reach my account?
      intent=policy_question -> retrieve_and_answer | top chunk doc_02#1 (Returns & Refunds, cosine 0.6047) expected doc_02
PASS  policy  | What does the Zepto Pass+ membership include?
      intent=policy_question -> retrieve_and_answer | top chunk doc_03#0 (Membership Tiers, cosine 0.7363) expected doc_03
PASS  policy  | My order tracking shows no movement for 30 minutes, what should I do?
      intent=policy_question -> retrieve_and_answer | top chunk doc_04#2 (Order Tracking, cosine 0.676) expected doc_04
PASS  policy  | Can I cancel an order after it has been packed?
      intent=policy_question -> retrieve_and_answer | top chunk doc_05#0 (Order Cancellation Policy, cosine 0.7756) expected doc_05
PASS  policy  | Can I combine two gift card balances in one order?
      intent=policy_question -> retrieve_and_answer | top chunk doc_07#2 (Gift Cards, cosine 0.8346) expected doc_07
PASS  policy  | What are your support hours, and is there phone support?
      intent=policy_question -> retrieve_and_answer | top chunk doc_08#3 (Customer Support Hours, cosine 0.5844) expected doc_08
PASS  general | What is the capital of France?
      intent=general_question -> direct_answer | answer='I can only answer questions about Zepto policies right now.'
PASS  general | Tell me a joke about bananas.
      intent=general_question -> direct_answer | answer='I can only answer questions about Zepto policies right now.'

ALL CHECKS PASSED
```

(`sources` and `confidence` lines are omitted above for brevity; the script prints them.)

## Design decisions

- **Sentence-level chunking** rather than one chunk per document. It gives more precise retrieval and a
  snippet that actually contains the answer. Without it, "How long does a refund take?" would quote the
  document's opening sentence about return windows instead of the "3–5 business days" rule.
- **Confidence = top-chunk cosine similarity.** It is still fully deterministic, but it tells the caller how
  closely the best chunk matched, rather than a constant. `direct_answer` uses 1.0, because its canned reply
  is certain by construction.
- **`sources` = all top-3 chunk ids**, in similarity order, exactly as the brief's mock specification states.
- **Keyword heuristic implemented literally**, with exactly the 8 phrases from the brief. As a consequence, a
  policy-sounding question without any of them (e.g. "Is phone support available?") is routed to
  `direct_answer`. That is the specified mock behaviour; the optional LLM classifier is what would handle
  such cases.
- **Model loaded from the local cache** (`local_files_only=True`), so after the first download nothing
  contacts Hugging Face. ChromaDB runs with telemetry disabled.
- **`truststore`** makes the one-time model download verify HTTPS against the OS certificate store. This was
  needed on the development machine, where antivirus software re-signs HTTPS traffic.
