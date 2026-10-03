"""
schemas.py
==========
Pydantic models for the /ask endpoint's request and response.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Chunk/document IDs used to answer. Empty for general_question answers.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
