import logging
import os
import uuid
import time
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.logging_config import setup_logging
from app.health import router as health_router
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ValidateAccessRequest,
    ValidateAccessResponse,
    TraceResponse,
)
from app.deps import get_dependencies, Dependencies
from app.trace_store import get_trace_store

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API server starting...")
    yield
    logger.info("RAG API server shutting down...")


app = FastAPI(
    title="Unified Advanced RAG System",
    description="Agentic RAG with hybrid retrieval, KG, and ACL",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)


@app.post("/ask", response_model=AskResponse)
async def ask_query(
    request: AskRequest,
    deps: Dependencies = Depends(get_dependencies),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"Request {request_id}: {request.query[:100]}")

    if not deps.is_ready():
        logger.error("Pipeline not ready")
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


@app.post("/validate-access", response_model=ValidateAccessResponse)
async def validate_access(
    request: ValidateAccessRequest,
    deps: Dependencies = Depends(get_dependencies),
):
    allowed = deps.acl_checker.can_access(request.user_role, request.document_type)

    reason = None
    if not allowed:
        reason = f"User role '{request.user_role}' cannot access '{request.document_type}' documents"

    redaction_hints = []
    redaction_rules = deps.acl_checker.get_redaction_rules(request.user_role)
    if redaction_rules:
        redaction_hints = [rule.get("pattern", "") for rule in redaction_rules]

    return ValidateAccessResponse(
        allowed=allowed,
        reason=reason,
        redaction_hints=redaction_hints,
    )


@app.get("/debug/trace")
async def get_latest_trace(deps: Dependencies = Depends(get_dependencies)):
    trace = get_trace_store().get_latest()

    if not trace:
        return JSONResponse(
            status_code=404,
            content={"detail": "No trace available yet"},
        )

    return trace


@app.post("/evaluate")
async def run_evaluation(deps: Dependencies = Depends(get_dependencies)):
    if not deps.is_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "System not ready"},
        )

    try:
        import asyncio

        from rag_pipeline.evaluator import RAGEvaluator
        from pathlib import Path

        evaluator = RAGEvaluator(deps.rag_pipeline, Path("rag_pipeline/evaluation_results"))
        results = await evaluator.evaluate()

        return {
            "status": "completed",
            "results_file": "rag_pipeline/evaluation_results/results.json",
            "chart_files": [
                "rag_pipeline/evaluation_results/faithfulness_by_query.png",
                "rag_pipeline/evaluation_results/latency_histogram.png",
            ],
            "summary": results.get("summary", {}),
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Evaluation failed: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
