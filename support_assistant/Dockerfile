# Zepto support assistant: FastAPI + LangGraph + ChromaDB, mock LLM by default.
#   docker build -t zepto-support-assistant .
#   docker run --rm -p 7860:7860 zepto-support-assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    MOCK_LLM=1

# Non-root user (uid 1000, the user Hugging Face Spaces runs containers as)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Dependencies first, so this layer is cached across code changes.
# requirements.txt pulls CPU-only PyTorch from the PyTorch wheel index.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code and the 8 policy documents
COPY --chown=appuser:appuser . .
RUN chown -R appuser:appuser /app
USER appuser

# Download all-MiniLM-L6-v2 and build the ChromaDB index at build time, so the
# container starts quickly and needs no network access at run time.
RUN python ingest.py
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
