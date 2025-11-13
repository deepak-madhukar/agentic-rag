from pydantic import BaseModel
from typing import Optional, List


class AskRequest(BaseModel):
    query: str
    user_role: str = "Admin"
    filters: Optional[dict] = None
    retrieval_depth: int = 10


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieval_depth: int
    hallucination_score: float
    latency_ms: int


class ValidateAccessRequest(BaseModel):
    user_role: str
    document_type: str


class ValidateAccessResponse(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    redaction_hints: List[str] = []


class HealthResponse(BaseModel):
    status: str
    message: str


class TraceResponse(BaseModel):
    request_id: str
    query: str
    planner_decision: dict
    retriever_stats: dict
    synthesizer_stats: dict
    total_time_ms: int
