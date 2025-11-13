import logging
import os
import uuid
import time
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.utils.logging_config import setup_logging
from app.router.health import router as health_router
from app.models.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ValidateAccessRequest,
    ValidateAccessResponse,
    TraceResponse,
)
from app.core.deps import get_dependencies, Dependencies
from app.service.trace_store import get_trace_store
from app.router.ask_route import router as ask_router
from app.router.debug_trace_route import router as debug_trace_router
from app.router.evaluate_route import router as evaluate_router
from app.router.health import router as health_router
from app.router.validate_access_route import router as validate_access_router

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API server starting...")
    yield
    logger.info("RAG API server shutting down...")


app = FastAPI(
    title="Advanced Agentic RAG System",
    description="Agentic RAG with hybrid retrieval, KG, and ACL",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(ask_router)
app.include_router(debug_trace_router)
app.include_router(evaluate_router)
app.include_router(health_router)
app.include_router(validate_access_router)
