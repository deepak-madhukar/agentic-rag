from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import uuid
import time
from app.models.schemas import AskRequest, AskResponse, Citation
from app.core.dependecies import get_dependencies, Dependencies
from app.service.trace_store import get_trace_store

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask_query(
    request: AskRequest,
    deps: Dependencies = Depends(get_dependencies),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    if not deps.is_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "System not ready. Run ingestion pipeline first."},
        )

    synthesis_result, trace = deps.rag_pipeline.process_query(
        request_id=request_id,
        query=request.query,
        user_role=request.user_role,
        retrieval_depth=request.retrieval_depth,
    )

    trace_dict = {
        "request_id": request_id,
        "query": request.query,
        "planner_decision": trace.planner_decision,
        "retriever_stats": trace.retriever_stats,
        "synthesizer_stats": trace.synthesizer_stats,
        "total_time_ms": trace.total_time_ms,
        "timestamp": time.time(),
    }
    get_trace_store().store(request_id, trace_dict)

    latency_ms = int((time.time() - start_time) * 1000)

    citations = [
        Citation(
            doc_id=c.doc_id,
            chunk_id=c.chunk_id,
            score=c.score,
            metadata={},
        )
        for c in synthesis_result.citations
    ]

    return AskResponse(
        answer=synthesis_result.answer,
        citations=citations,
        retrieval_depth=request.retrieval_depth,
        hallucination_score=synthesis_result.hallucination_score,
        latency_ms=latency_ms,
    )