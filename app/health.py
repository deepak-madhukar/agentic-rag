import logging
from fastapi import APIRouter, Depends
from app.schemas import HealthResponse
from app.deps import get_dependencies, Dependencies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health(deps: Dependencies = Depends(get_dependencies)) -> HealthResponse:
    status = "healthy" if deps.llm_client else "degraded"
    message = "System is operational" if status == "healthy" else "Some components missing"

    return HealthResponse(status=status, message=message)


@router.get("/ready", response_model=HealthResponse)
async def readiness(deps: Dependencies = Depends(get_dependencies)) -> HealthResponse:
    if deps.is_ready():
        return HealthResponse(status="ready", message="All systems initialized")
    else:
        return HealthResponse(
            status="not_ready",
            message="Index or dependencies not loaded. Run ingestion pipeline first.",
        )
