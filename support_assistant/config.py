"""Shared settings for the Zepto support assistant."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
SNIPPET_CHARS = 200

DOC_TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}

# Optional real-LLM extension (MOCK_LLM=0): Groq's OpenAI-compatible endpoint.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_SCHEMA_RETRIES = 2  # extra attempts after the first, if the LLM output fails validation


def mock_llm_enabled() -> bool:
    """MOCK_LLM unset or "1" -> deterministic mock (the graded default).

    Only an explicit MOCK_LLM=0 switches on the optional real-LLM calls.
    """
    return os.getenv("MOCK_LLM", "1").strip() != "0"


def use_os_certificate_store() -> None:
    """Verify HTTPS against the OS certificate store (for the one-time model download).

    Needed on machines where antivirus software re-signs HTTPS traffic; harmless elsewhere.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
