"""Ingestion + embedding: docs/*.txt -> chunks -> all-MiniLM-L6-v2 vectors -> ChromaDB.

Chunking scheme: sentence-level chunks (SENTENCES_PER_CHUNK = 1). Each policy
document is only 3-5 sentences long and every sentence states one policy point
(a fee, a time limit, an exception). One sentence per chunk means the top retrieved
chunk is the specific rule that answers the question, so the mock answer's
~200-character snippet quotes that rule rather than a document's opening line.

Chunk ids look like "doc_02#1" (document id + chunk number); these are the
`sources` returned by the API.
"""

import re
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from config import (CHROMA_DIR, COLLECTION_NAME, DOC_TITLES, DOCS_DIR, EMBEDDING_MODEL,
                    use_os_certificate_store)

SENTENCES_PER_CHUNK = 1


def load_documents() -> dict[str, str]:
    """Read every docs/doc_XX.txt file -> {"doc_01": text, ...}."""
    docs = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(DOCS_DIR.glob("doc_*.txt"))}
    if len(docs) != 8:
        raise RuntimeError(f"Expected 8 policy documents in {DOCS_DIR}, found {len(docs)}")
    return docs


def chunk_document(doc_id: str, text: str) -> list[dict]:
    """Split one document into chunks of SENTENCES_PER_CHUNK sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        chunks.append({
            "id": f"{doc_id}#{i // SENTENCES_PER_CHUNK}",
            "text": " ".join(sentences[i:i + SENTENCES_PER_CHUNK]),
            "metadata": {"doc_id": doc_id, "title": DOC_TITLES.get(doc_id, doc_id),
                         "chunk": i // SENTENCES_PER_CHUNK},
        })
    return chunks


def build_chunks() -> list[dict]:
    return [c for doc_id, text in load_documents().items() for c in chunk_document(doc_id, text)]


@lru_cache(maxsize=1)
def get_embedder():
    """Load all-MiniLM-L6-v2 once per process (runs locally, no API key)."""
    use_os_certificate_store()
    from sentence_transformers import SentenceTransformer
    try:
        # Use the locally cached model without contacting the Hugging Face Hub
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL)  # first run: download once, then cached


def embed(texts: list[str]) -> list[list[float]]:
    # Normalized vectors, so cosine similarity = dot product; Chroma is also told to use cosine.
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def build_index(rebuild: bool = False):
    """Create (or reuse) the ChromaDB collection holding every chunk's embedding."""
    client = get_client()
    chunks = build_chunks()

    if not rebuild:
        try:
            existing = client.get_collection(COLLECTION_NAME)
            if existing.count() == len(chunks):
                return existing
        except Exception:
            pass  # collection does not exist yet

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # similarity search by cosine distance
        embedding_function=None,             # we always pass our own MiniLM vectors
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
    )
    return collection


def retrieve(collection, query: str, k: int = 3) -> list[dict]:
    """Embed the query and return the top-k chunks by cosine similarity."""
    result = collection.query(query_embeddings=embed([query]), n_results=k,
                              include=["documents", "metadatas", "distances"])
    return [
        {"id": cid, "text": text, "doc_id": meta["doc_id"], "title": meta["title"],
         "similarity": round(1.0 - dist, 4)}   # cosine distance -> cosine similarity
        for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                         result["metadatas"][0], result["distances"][0])
    ]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    col = build_index(rebuild=True)
    print(f"Indexed {col.count()} chunks from 8 documents into ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR.name}/")
    for item in col.get(include=["documents"])["ids"]:
        print(" ", item)
