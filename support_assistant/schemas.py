"""Pydantic models: the API request and the enforced JSON answer schema."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The assistant's answer")
    sources: list[str] = Field(default_factory=list,
                               description="Chunk ids used (e.g. 'doc_02#1'); empty for general questions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0 and 1")
